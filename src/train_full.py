"""
Full-Dataset Training Pipeline for Amazon ML Challenge 2026 Business Entity Resolution.
Supports the complete 2.2M Source 1 population using:
1. Country-partitioned target indexing for strict memory safety (< 1 GB RAM).
2. Streaming chunked processing of Source 1 (chunk size = 50,000).
3. Ground-truth positive extraction + hard negative mining from realistic blocking collisions.
4. Cumulative global float32 feature matrix persistence (no naive model overwrite).
5. Single global XGBoost hist model trained across all training entities.
6. Strict entity-level validation isolation (50,000 holdout S1 entities).
7. Threshold optimization for official entity-level Macro F0.5 (including singletons).
"""

import gc
import json
import os
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src import config
from src.blocking import BlockingIndex, generate_blocking_keys
from src.data_loader import stream_tsv_chunks
from src.evaluate import compute_macro_f05
from src.features import extract_pairwise_features
from src.model import EntityMatchingModel

def get_peak_ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return usage / 1024.0
    return usage / (1024.0 * 1024.0)

def partition_target_files_by_country(
    source2_path: Path,
    source3_path: Path,
    out_dir: Path
) -> Set[str]:
    """
    Stream Source 2 and Source 3 line by line and partition into per-country TSV files.
    Peak RAM is negligible (< 30 MB).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    open_handles: Dict[str, any] = {}
    countries_seen: Set[str] = set()

    usecols = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]

    print(f"  Streaming and partitioning target records from S2 and S3 into {out_dir}...")
    for label, path in [("Source 2", source2_path), ("Source 3", source3_path)]:
        count = 0
        for chunk in stream_tsv_chunks(path, chunk_size=100000, usecols=usecols):
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                norm_c = (str(country or "")).strip().upper() or "UNKNOWN"
                if norm_c not in open_handles:
                    c_file = out_dir / f"target_{norm_c}.tsv"
                    open_handles[norm_c] = open(c_file, "w", encoding="utf-8")
                    open_handles[norm_c].write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                    countries_seen.add(norm_c)
                
                clean_n = str(name or "").replace("\t", " ").replace("\n", " ")
                clean_a = str(addr or "").replace("\t", " ").replace("\n", " ")
                open_handles[norm_c].write(f"{eid}\t{clean_n}\t{clean_a}\t{norm_c}\n")
                count += 1
        print(f"    Partitioned {count:,} records from {label}.")

    for h in open_handles.values():
        h.close()

    print(f"  Partitioning complete. Discovered countries: {sorted(list(countries_seen))}")
    return countries_seen

def run_full_training(
    val_size: int = 50000,
    chunk_size: int = 50000,
    max_bucket_size: int = 500,
    max_candidates: int = 65,
    max_negatives_per_entity: int = 2
):
    start_total_time = time.time()
    print("=" * 70)
    print("AMAZON ML CHALLENGE — FULL DATASET TRAINING PIPELINE (2.2M S1)")
    print(f"Validation Holdout: {val_size:,} S1 entities | Chunk Size: {chunk_size:,}")
    print(f"Max Bucket Size: {max_bucket_size} | Max Candidates: {max_candidates}")
    print("=" * 70)

    # 1. Inspect ground truth count and split S1 IDs into train and validation
    print("\n[STEP 1/6] Indexing S1 entities and Ground Truth...")
    gt_matches: Dict[str, Set[str]] = defaultdict(set)
    total_gt_rows = 0
    for chunk in stream_tsv_chunks(config.TRAIN_GROUND_TRUTH, chunk_size=100000):
        for s1_id, m_str in chunk.itertuples(index=False, name=None):
            total_gt_rows += 1
            if pd.notna(m_str) and str(m_str).strip():
                for mid in str(m_str).split(","):
                    mid = mid.strip()
                    if mid:
                        gt_matches[s1_id].add(mid)

    print(f"  Total Ground Truth S1 entities indexed: {total_gt_rows:,}")
    print(f"  Total non-singleton S1 entities in GT: {len(gt_matches):,}")
    print(f"  Current RAM: {get_peak_ram_mb():.1f} MB")

    # Partition target files by country
    print("\n[STEP 2/6] Partitioning Target Training Records (S2 & S3) by Country...")
    part_dir = config.ARTIFACT_DIR / "train_partitions"
    countries = partition_target_files_by_country(config.TRAIN_SOURCE2, config.TRAIN_SOURCE3, part_dir)

    # 3. Stream S1 and extract features country by country
    print("\n[STEP 3/6] Mining Positive Matches and Hard Negatives Country-by-Country...")
    all_X_chunks: List[np.ndarray] = []
    all_y_chunks: List[np.ndarray] = []
    total_positives = 0
    total_negatives = 0

    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_gt: Dict[str, Set[str]] = {}

    for country in sorted(list(countries)):
        target_file = part_dir / f"target_{country}.tsv"
        if not target_file.is_file():
            continue

        print(f"\n  --- Processing Country: {country} ---")
        t_c_start = time.time()
        c_index = BlockingIndex(max_bucket_size=max_bucket_size)

        for chunk in stream_tsv_chunks(target_file, chunk_size=100000):
            for eid, name, addr, c_code in chunk.itertuples(index=False, name=None):
                c_index.add_entity(eid, str(name or ""), str(addr or ""), str(c_code or ""))

        print(f"    Indexed {len(c_index.entity_records):,} target records for {country} in {time.time() - t_c_start:.1f}s (RAM: {get_peak_ram_mb():.1f} MB)")

        # Stream S1 entities belonging to this country
        s1_country_count = 0
        s1_buffer: List[Tuple[str, Tuple[str, str, str, str]]] = []

        usecols_s1 = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]
        s1_idx = 0
        for chunk in stream_tsv_chunks(config.TRAIN_SOURCE1, chunk_size=chunk_size, usecols=usecols_s1):
            for eid, name, addr, c_code in chunk.itertuples(index=False, name=None):
                norm_c = (str(c_code or "")).strip().upper() or "UNKNOWN"
                s1_idx += 1

                # Reserve the last val_size entities for validation
                is_val = (s1_idx > (total_gt_rows - val_size))
                if norm_c == country:
                    s1_country_count += 1
                    rec = (str(name or ""), str(name or ""), str(addr or ""), norm_c)
                    if is_val:
                        val_s1_records[eid] = rec
                        val_gt[eid] = gt_matches.get(eid, set())
                    else:
                        s1_buffer.append((eid, rec))

            if len(s1_buffer) >= chunk_size:
                # Process train chunk
                chunk_X = []
                chunk_y = []
                for s1_id, s1_rec in s1_buffer:
                    true_mids = gt_matches.get(s1_id, set())
                    cand_hits = c_index.get_candidates_with_hits(
                        business_name=s1_rec[0],
                        business_address=s1_rec[2],
                        country=s1_rec[3],
                        max_candidates=max_candidates
                    )

                    # Positives
                    for t_id in true_mids:
                        if t_id in c_index.entity_records:
                            crec = c_index.entity_records[t_id]
                            hits = next((h for cid, h in cand_hits if cid == t_id), 1)
                            feats = extract_pairwise_features(s1_rec, t_id, crec, blocking_hits=hits)
                            chunk_X.append(feats)
                            chunk_y.append(1)
                            total_positives += 1

                    # Hard Negatives
                    neg_added = 0
                    for cid, hits in cand_hits:
                        if cid not in true_mids and cid in c_index.entity_records:
                            crec = c_index.entity_records[cid]
                            feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                            chunk_X.append(feats)
                            chunk_y.append(0)
                            total_negatives += 1
                            neg_added += 1
                            if neg_added >= max_negatives_per_entity:
                                break

                if chunk_X:
                    all_X_chunks.append(np.array(chunk_X, dtype=np.float32))
                    all_y_chunks.append(np.array(chunk_y, dtype=np.int32))
                del chunk_X, chunk_y, s1_buffer
                s1_buffer = []
                gc.collect()

        if s1_buffer:
            chunk_X = []
            chunk_y = []
            for s1_id, s1_rec in s1_buffer:
                true_mids = gt_matches.get(s1_id, set())
                cand_hits = c_index.get_candidates_with_hits(
                    business_name=s1_rec[0],
                    business_address=s1_rec[2],
                    country=s1_rec[3],
                    max_candidates=max_candidates
                )
                for t_id in true_mids:
                    if t_id in c_index.entity_records:
                        crec = c_index.entity_records[t_id]
                        hits = next((h for cid, h in cand_hits if cid == t_id), 1)
                        feats = extract_pairwise_features(s1_rec, t_id, crec, blocking_hits=hits)
                        chunk_X.append(feats)
                        chunk_y.append(1)
                        total_positives += 1

                neg_added = 0
                for cid, hits in cand_hits:
                    if cid not in true_mids and cid in c_index.entity_records:
                        crec = c_index.entity_records[cid]
                        feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                        chunk_X.append(feats)
                        chunk_y.append(0)
                        total_negatives += 1
                        neg_added += 1
                        if neg_added >= max_negatives_per_entity:
                            break

            if chunk_X:
                all_X_chunks.append(np.array(chunk_X, dtype=np.float32))
                all_y_chunks.append(np.array(chunk_y, dtype=np.int32))
            del chunk_X, chunk_y, s1_buffer
            gc.collect()

        print(f"    Finished {country}: processed {s1_country_count:,} S1 entities. Pairs so far: Pos={total_positives:,}, Neg={total_negatives:,}")
        del c_index
        gc.collect()

    print(f"\n  Cumulative training pairs mined: Positives={total_positives:,}, Hard Negatives={total_negatives:,}, Total={total_positives + total_negatives:,}")
    X_train = np.vstack(all_X_chunks)
    y_train = np.concatenate(all_y_chunks)
    del all_X_chunks, all_y_chunks
    gc.collect()
    print(f"  Feature Matrix Shape: {X_train.shape} | float32 | Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 4. Train Single Global XGBoost Model
    print("\n[STEP 4/6] Training Single Global XGBoost Hist Model...")
    model = EntityMatchingModel(
        max_depth=6,
        learning_rate=0.08,
        n_estimators=150,
        n_jobs=min(8, os.cpu_count() or 4)
    )
    t_m_start = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t_m_start
    print(f"  Global model fitted in {t_train:.1f}s.")
    model.save(config.MODEL_PATH)
    print(f"  Model saved to {config.MODEL_PATH}.")

    # Free training matrix before validation
    del X_train, y_train
    gc.collect()

    # 5. Validation Evaluation & Threshold Search
    print(f"\n[STEP 5/6] Evaluating on {len(val_s1_records):,} Strictly Isolated Validation Entities...")
    val_cand_scores: Dict[str, List[Tuple[str, float]]] = {}
    val_cand_counts = []
    total_val_cands = 0
    found_true_pairs = 0
    total_true_pairs = sum(len(m) for m in val_gt.values())

    # Evaluate validation country by country to keep memory bounded
    val_by_country: Dict[str, List[str]] = defaultdict(list)
    for eid, rec in val_s1_records.items():
        val_by_country[rec[3]].append(eid)

    for country, eids in val_by_country.items():
        target_file = part_dir / f"target_{country}.tsv"
        if not target_file.is_file():
            for eid in eids:
                val_cand_scores[eid] = []
            continue

        c_index = BlockingIndex(max_bucket_size=max_bucket_size)
        for chunk in stream_tsv_chunks(target_file, chunk_size=100000):
            for tid, name, addr, c_code in chunk.itertuples(index=False, name=None):
                c_index.add_entity(tid, str(name or ""), str(addr or ""), str(c_code or ""))

        for eid in eids:
            s1_rec = val_s1_records[eid]
            cand_hits = c_index.get_candidates_with_hits(
                business_name=s1_rec[0],
                business_address=s1_rec[2],
                country=s1_rec[3],
                max_candidates=max_candidates
            )
            val_cand_counts.append(len(cand_hits))
            total_val_cands += len(cand_hits)

            true_set = val_gt.get(eid, set())
            cand_set = {cid for cid, _ in cand_hits}
            found_true_pairs += len(true_set & cand_set)

            if not cand_hits:
                val_cand_scores[eid] = []
                continue

            feat_batch = []
            cand_ids = []
            for cid, hits in cand_hits:
                if cid in c_index.entity_records:
                    crec = c_index.entity_records[cid]
                    feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                    feat_batch.append(feats)
                    cand_ids.append(cid)

            if not feat_batch:
                val_cand_scores[eid] = []
                continue

            X_b = np.array(feat_batch, dtype=np.float32)
            probs = model.predict_proba(X_b)
            val_cand_scores[eid] = list(zip(cand_ids, probs))

        del c_index
        gc.collect()

    cand_recall = found_true_pairs / max(1, total_true_pairs)
    avg_cands = float(np.mean(val_cand_counts)) if val_cand_counts else 0.0
    med_cands = float(np.median(val_cand_counts)) if val_cand_counts else 0.0
    p95_cands = float(np.percentile(val_cand_counts, 95)) if val_cand_counts else 0.0
    max_cands = int(np.max(val_cand_counts)) if val_cand_counts else 0
    red_ratio = 1.0 - (total_val_cands / max(1, len(val_s1_records) * (total_gt_rows * 5)))

    print(f"  Candidate Recall                 : {cand_recall:.4f} ({found_true_pairs:,}/{total_true_pairs:,})")
    print(f"  Avg / Median / P95 / Max Cands   : {avg_cands:.1f} / {med_cands:.1f} / {p95_cands:.1f} / {max_cands}")
    print(f"  Candidate Reduction Ratio        : {red_ratio:.6f}")

    # Threshold grid search
    best_thresh = 0.50
    best_val_f05 = -1.0
    thresh_grid = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    for t in thresh_grid:
        preds = {}
        for eid, cands in val_cand_scores.items():
            matches = [cid for cid, prob in cands if prob >= t]
            preds[eid] = set(matches)
        f05 = compute_macro_f05(val_gt, preds)
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_thresh = t

    print(f"  Optimal Decision Threshold       : {best_thresh:.2f}")
    print(f"  Best Validation Macro F_0.5      : {best_val_f05:.4f}")

    # Precision, Recall, Accuracy at best threshold
    tp, fp = 0, 0
    total_preds = 0
    for eid, cands in val_cand_scores.items():
        pred_set = {cid for cid, prob in cands if prob >= best_thresh}
        true_set = val_gt.get(eid, set())
        for cid in pred_set:
            if cid in true_set:
                tp += 1
            else:
                fp += 1
        total_preds += len(pred_set)

    fn = total_true_pairs - tp
    tn = max(0, total_val_cands - (tp + fp))
    pair_precision = tp / max(1, total_preds)
    pair_recall = tp / max(1, total_true_pairs)
    pairwise_acc = (tp + tn) / max(1, total_val_cands)

    print(f"  Pair Precision                   : {pair_precision:.4f} ({tp:,}/{total_preds:,})")
    print(f"  Pair Recall                      : {pair_recall:.4f} ({tp:,}/{total_true_pairs:,})")
    print(f"  Diagnostic Pairwise Accuracy     : {pairwise_acc:.4%} (TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,})")

    # Save metrics
    metrics = {
        "candidate_recall": float(cand_recall),
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(best_val_f05),
        "pair_precision": float(pair_precision),
        "pair_recall": float(pair_recall),
        "pairwise_accuracy": float(pairwise_acc),
        "avg_candidates_per_entity": float(avg_cands),
        "median_candidates_per_entity": float(med_cands),
        "p95_candidates_per_entity": float(p95_cands),
        "max_candidates_per_entity": int(max_cands),
        "candidate_reduction_ratio": float(red_ratio),
        "n_train_s1": total_gt_rows - val_size,
        "n_val_s1": val_size,
        "n_train_positives": int(total_positives),
        "n_train_negatives": int(total_negatives),
        "n_train_pairs_total": int(total_positives + total_negatives),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "peak_ram_mb": float(get_peak_ram_mb()),
        "training_time_s": float(t_train),
        "total_runtime_s": float(time.time() - start_total_time)
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved validation metrics to {config.METRICS_PATH}.")

    # 6. Cleanup temporary train partitions
    print("\n[STEP 6/6] Cleaning Up Temporary Training Partitions...")
    shutil.rmtree(part_dir, ignore_errors=True)
    print(f"  Cleaned {part_dir}.")
    print(f"\nFULL TRAINING COMPLETED in {time.time() - start_total_time:.1f}s | Peak RAM: {get_peak_ram_mb():.1f} MB")
    return metrics

if __name__ == "__main__":
    run_full_training()
