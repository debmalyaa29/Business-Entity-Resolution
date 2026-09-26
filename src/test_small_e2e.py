"""
Amazon ML Challenge 2026 — Small-Scale End-to-End Safety Test.
Validates the entire pipeline on 10,000 Source-1 entities before full 2.2M training:
1. Verifies conda amazon-ml and Python 3.10.x.
2. Chunked processing with memory cleanup.
3. Memory-safe country-partitioned S2/S3 lookup.
4. Candidate generation with PRE4 and distinctive address blocking.
5. Cumulative global feature accumulation and single XGBoost model training.
6. Strict validation isolation and threshold optimization for Macro F0.5.
7. Submission generation and official validator check.
"""

import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src import config
from src.blocking import CountryPartitionedBlockingIndex, generate_blocking_keys
from src.data_loader import stream_tsv_chunks
from src.evaluate import compute_macro_f05, compute_entity_f05
from src.features import extract_pairwise_features
from src.model import EntityMatchingModel

def get_peak_ram_mb() -> float:
    """Return peak resident memory in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux, ru_maxrss is in kilobytes; on macOS, bytes
    if sys.platform.startswith("linux"):
        return usage / 1024.0
    return usage / (1024.0 * 1024.0)

def run_small_e2e_test(
    total_s1: int = 10000,
    n_train_s1: int = 8000,
    chunk_size: int = 4000,
    n_test_s1: int = 5000,
    max_bucket_size: int = 500,
    max_candidates: int = 65,
    max_s2_s3_records: int = 300000
):
    start_time = time.time()
    print("=" * 70)
    print("AMAZON ML CHALLENGE — SMALL-SCALE END-TO-END SAFETY TEST")
    print(f"S1 Sample: {total_s1:,} entities ({n_train_s1:,} train, {total_s1 - n_train_s1:,} val)")
    print(f"Chunk size: {chunk_size:,} | Test Sample: {n_test_s1:,}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. PRE-FLIGHT VERIFICATION
    # -------------------------------------------------------------
    print("\n[STEP 1/7] Environment Pre-flight Verification...")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    python_ver = sys.version.split()[0]
    exec_path = sys.executable
    print(f"  Conda Env   : {conda_env}")
    print(f"  Python      : {python_ver} ({exec_path})")
    assert python_ver.startswith("3.10"), f"Expected Python 3.10.x, got {python_ver}"
    print("  Pre-flight check PASSED.")

    # -------------------------------------------------------------
    # 2. LOAD GROUND TRUTH & S1 DATA
    # -------------------------------------------------------------
    print("\n[STEP 2/7] Loading Ground Truth & S1 Sample...")
    # Load ground truth for the first total_s1 entities
    s1_sample_ids: List[str] = []
    s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    usecols_s1 = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]

    for chunk in stream_tsv_chunks(config.TRAIN_SOURCE1, chunk_size=chunk_size, usecols=usecols_s1):
        for eid, name, addr, country in chunk.itertuples(index=False, name=None):
            s1_sample_ids.append(eid)
            s1_records[eid] = (str(name or ""), str(name or ""), str(addr or ""), str(country or ""))
            if len(s1_sample_ids) >= total_s1:
                break
        if len(s1_sample_ids) >= total_s1:
            break

    s1_set = set(s1_sample_ids)
    print(f"  Loaded {len(s1_sample_ids):,} Source 1 entities.")

    gt_matches: Dict[str, Set[str]] = defaultdict(set)
    for chunk in stream_tsv_chunks(config.TRAIN_GROUND_TRUTH, chunk_size=50000):
        for s1_id, m_str in chunk.itertuples(index=False, name=None):
            if s1_id in s1_set and pd.notna(m_str) and str(m_str).strip():
                for mid in str(m_str).split(","):
                    mid = mid.strip()
                    if mid:
                        gt_matches[s1_id].add(mid)

    train_ids = s1_sample_ids[:n_train_s1]
    val_ids = s1_sample_ids[n_train_s1:total_s1]
    train_gt = {eid: gt_matches[eid] for eid in train_ids if eid in gt_matches}
    val_gt = {eid: gt_matches[eid] for eid in val_ids}
    val_singletons = sum(1 for eid in val_ids if len(val_gt[eid]) == 0)
    print(f"  Train S1: {len(train_ids):,} ({sum(len(v) for v in train_gt.values()):,} true matches)")
    print(f"  Val S1  : {len(val_ids):,} ({sum(len(v) for v in val_gt.values()):,} true matches, {val_singletons:,} singletons)")

    # -------------------------------------------------------------
    # 3. BUILD COUNTRY-PARTITIONED S2/S3 INDEX
    # -------------------------------------------------------------
    print(f"\n[STEP 3/7] Building Country-Partitioned S2/S3 Target Index (up to {max_s2_s3_records:,} records)...")
    target_index = CountryPartitionedBlockingIndex(max_bucket_size=max_bucket_size)
    usecols_target = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]

    total_indexed = 0
    for label, path in [("Train S2", config.TRAIN_SOURCE2), ("Train S3", config.TRAIN_SOURCE3)]:
        for chunk in stream_tsv_chunks(path, chunk_size=config.CHUNK_SIZE, usecols=usecols_target):
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                target_index.add_entity(eid, str(name or ""), str(addr or ""), str(country or ""))
                total_indexed += 1
                if total_indexed >= max_s2_s3_records:
                    break
            if total_indexed >= max_s2_s3_records:
                break
        if total_indexed >= max_s2_s3_records:
            break

    print(f"  Indexed {target_index.total_entities:,} target entities across {len(target_index.partitions)} country partitions: {list(target_index.partitions.keys())}")
    print(f"  Memory after S2/S3 indexing: {get_peak_ram_mb():.1f} MB")

    # -------------------------------------------------------------
    # 4. CHUNKED FEATURE EXTRACTION & HARD NEGATIVE MINING
    # -------------------------------------------------------------
    print(f"\n[STEP 4/7] Chunked Feature Generation (chunk size = {chunk_size:,})...")
    X_chunks = []
    y_chunks = []
    total_pos = 0
    total_neg = 0

    # Process train entities in chunks to verify chunking and memory cleanup
    n_train_chunks = (len(train_ids) + chunk_size - 1) // chunk_size
    for c_idx in range(n_train_chunks):
        c_train_ids = train_ids[c_idx * chunk_size : (c_idx + 1) * chunk_size]
        chunk_X = []
        chunk_y = []

        for s1_id in c_train_ids:
            s1_rec = s1_records[s1_id]
            true_mids = train_gt.get(s1_id, set())

            # Candidate generation via blocking
            cand_hits = target_index.get_candidates_with_hits(
                business_name=s1_rec[0],
                business_address=s1_rec[2],
                country=s1_rec[3],
                max_candidates=max_candidates
            )
            cand_set = {cid for cid, _ in cand_hits}

            # 1. Ground truth positives (if indexed in target)
            for t_id in true_mids:
                crec = target_index.get_entity_record(t_id, s1_rec[3])
                if crec is not None:
                    # Hits: check if it appeared in blocking collisions
                    hits = next((h for cid, h in cand_hits if cid == t_id), 1)
                    feats = extract_pairwise_features(s1_rec, t_id, crec, blocking_hits=hits)
                    chunk_X.append(feats)
                    chunk_y.append(1)
                    total_pos += 1

            # 2. Hard negatives from realistic blocking collisions
            neg_count = 0
            max_neg_per_s1 = 2
            for cid, hits in cand_hits:
                if cid not in true_mids:
                    crec = target_index.get_entity_record(cid, s1_rec[3])
                    if crec is not None:
                        feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                        chunk_X.append(feats)
                        chunk_y.append(0)
                        total_neg += 1
                        neg_count += 1
                        if neg_count >= max_neg_per_s1:
                            break

        if chunk_X:
            X_chunks.append(np.array(chunk_X, dtype=np.float32))
            y_chunks.append(np.array(chunk_y, dtype=np.int32))
        
        # Explicit memory cleanup after chunk
        del chunk_X, chunk_y
        gc.collect()
        print(f"  Processed Train Chunk {c_idx+1}/{n_train_chunks}: cumulative pairs = {total_pos + total_neg:,} (RAM: {get_peak_ram_mb():.1f} MB)")

    # Global concatenation into single training matrix
    X_train = np.vstack(X_chunks)
    y_train = np.concatenate(y_chunks)
    del X_chunks, y_chunks
    gc.collect()

    print(f"  Global Feature Matrix: shape = {X_train.shape} | float32 | RAM = {get_peak_ram_mb():.1f} MB")
    print(f"  Training pairs: Positives = {total_pos:,}, Hard Negatives = {total_neg:,}, Total = {len(y_train):,}")

    # -------------------------------------------------------------
    # 5. TRAIN ONE GLOBAL XGBOOST MODEL
    # -------------------------------------------------------------
    print("\n[STEP 5/7] Training Global XGBoost Matching Model...")
    model = EntityMatchingModel(
        max_depth=6,
        learning_rate=0.08,
        n_estimators=100,
        n_jobs=min(8, os.cpu_count() or 4)
    )
    t_train_start = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t_train_start
    print(f"  Model trained in {t_train:.2f} seconds.")

    # -------------------------------------------------------------
    # 6. VALIDATION EVALUATION & THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------
    print("\n[STEP 6/7] Evaluating Validation Entities (Strict Isolation)...")
    val_cand_scores: Dict[str, List[Tuple[str, float]]] = {}
    found_true_pairs = 0
    total_true_pairs = sum(len(m) for m in val_gt.values())
    total_val_cands = 0
    cand_per_entity_list = []

    for s1_id in val_ids:
        s1_rec = s1_records[s1_id]
        cand_hits = target_index.get_candidates_with_hits(
            business_name=s1_rec[0],
            business_address=s1_rec[2],
            country=s1_rec[3],
            max_candidates=max_candidates
        )
        cand_per_entity_list.append(len(cand_hits))
        total_val_cands += len(cand_hits)

        # Candidate recall check
        true_set = val_gt[s1_id]
        cand_set = {cid for cid, _ in cand_hits}
        found_true_pairs += len(true_set & cand_set)

        if not cand_hits:
            val_cand_scores[s1_id] = []
            continue

        feat_batch = []
        cand_ids = []
        for cid, hits in cand_hits:
            crec = target_index.get_entity_record(cid, s1_rec[3])
            if crec is not None:
                feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                feat_batch.append(feats)
                cand_ids.append(cid)

        if not feat_batch:
            val_cand_scores[s1_id] = []
            continue

        X_val_batch = np.array(feat_batch, dtype=np.float32)
        probs = model.predict_proba(X_val_batch)
        val_cand_scores[s1_id] = list(zip(cand_ids, probs))

    cand_recall = found_true_pairs / max(1, total_true_pairs)
    avg_cands = np.mean(cand_per_entity_list)
    med_cands = np.median(cand_per_entity_list)
    p95_cands = np.percentile(cand_per_entity_list, 95)
    max_cands = np.max(cand_per_entity_list)
    reduction_ratio = 1.0 - (total_val_cands / max(1, len(val_ids) * target_index.total_entities))

    print(f"  Candidate Recall                 : {cand_recall:.4f} ({found_true_pairs}/{total_true_pairs})")
    print(f"  Avg / Median / P95 / Max Cands   : {avg_cands:.1f} / {med_cands:.1f} / {p95_cands:.1f} / {max_cands}")
    print(f"  Candidate Reduction Ratio        : {reduction_ratio:.6f}")

    # Optimize threshold strictly on validation Macro F0.5
    best_thresh = 0.50
    best_val_f05 = -1.0
    thresh_grid = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    for t in thresh_grid:
        preds = {}
        for eid, cands in val_cand_scores.items():
            matches = [cid for cid, prob in cands if prob >= t]
            preds[eid] = set(matches)
        macro_f05 = compute_macro_f05(val_gt, preds)
        if macro_f05 > best_val_f05:
            best_val_f05 = macro_f05
            best_thresh = t

    print(f"  Optimal Decision Threshold       : {best_thresh:.2f}")
    print(f"  Best Validation Macro F_0.5      : {best_val_f05:.4f}")

    # Compute pair precision, recall, and diagnostic accuracy at best threshold
    tp, fp = 0, 0
    total_preds = 0
    for s1_id, cands in val_cand_scores.items():
        pred_set = {cid for cid, prob in cands if prob >= best_thresh}
        true_set = val_gt.get(s1_id, set())
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

    print(f"  Pair Precision                   : {pair_precision:.4f} ({tp}/{total_preds})")
    print(f"  Pair Recall                      : {pair_recall:.4f} ({tp}/{total_true_pairs})")
    print(f"  Diagnostic Pairwise Accuracy     : {pairwise_acc:.4%} (TP={tp}, FP={fp}, FN={fn}, TN={tn:,})")

    # -------------------------------------------------------------
    # 7. TEST INFERENCE & OFFICIAL SUBMISSION VALIDATOR
    # -------------------------------------------------------------
    print(f"\n[STEP 7/7] Test Output Generation & Official Submission Validation...")
    test_small_dir = config.DATA_DIR / "test_small"
    test_small_dir.mkdir(parents=True, exist_ok=True)
    test_small_s1_path = test_small_dir / "test_source1.tsv"

    # Extract first n_test_s1 from test_source1.tsv for validator testing
    test_s1_rows = []
    with open(config.TEST_SOURCE1, "r", encoding="utf-8") as f_in:
        header = f_in.readline()
        for i, line in enumerate(f_in):
            if i >= n_test_s1:
                break
            test_s1_rows.append(line)

    with open(test_small_s1_path, "w", encoding="utf-8") as f_out:
        f_out.write(header)
        f_out.writelines(test_s1_rows)

    # Generate output files
    matching_out_path = config.OUTPUT_DIR / "matching_results.tsv"
    candidate_out_path = config.OUTPUT_DIR / "candidate_pairs.tsv"

    with open(matching_out_path, "w", encoding="utf-8") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8") as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for line in test_s1_rows:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            cand_hits = target_index.get_candidates_with_hits(
                business_name=name,
                business_address=addr,
                country=country,
                max_candidates=max_candidates
            )
            cands = [cid for cid, _ in cand_hits]
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

            if not cands:
                f_match.write(f"{s1_id}\t\n")
                continue

            feat_batch = []
            cand_ids = []
            for cid, hits in cand_hits:
                crec = target_index.get_entity_record(cid, country)
                if crec is not None:
                    feats = extract_pairwise_features((name, name, addr, country), cid, crec, blocking_hits=hits)
                    feat_batch.append(feats)
                    cand_ids.append(cid)

            if not feat_batch:
                f_match.write(f"{s1_id}\t\n")
                continue

            X_b = np.array(feat_batch, dtype=np.float32)
            probs = model.predict_proba(X_b)
            matched = [cand_ids[i] for i, p in enumerate(probs) if p >= best_thresh]
            # Deduplicate preserving order
            unique_matched = list(dict.fromkeys(matched))
            f_match.write(f"{s1_id}\t{','.join(unique_matched)}\n")

    print(f"  Generated matching_results.tsv ({n_test_s1:,} rows)")
    print(f"  Generated candidate_pairs.tsv  ({n_test_s1:,} rows)")

    # Run official validator
    validator_script = config.BASE_DIR / "utils" / "validate_submission.py"
    val_cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_out_path),
        "--candidate", str(candidate_out_path),
        "--test-dir", str(test_small_dir)
    ]
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print("\n--- OFFICIAL VALIDATOR OUTPUT ---")
    print(val_proc.stdout.strip())
    if val_proc.stderr.strip():
        print("Validator STDERR:", val_proc.stderr.strip())
    print("---------------------------------")
    assert val_proc.returncode == 0, f"Validator returned non-zero code {val_proc.returncode}"
    print(f"  Official Validator Result: PASS (exit code {val_proc.returncode})")

    peak_ram = get_peak_ram_mb()
    total_time = time.time() - start_time
    print(f"\nSmall-Scale End-to-End Test COMPLETED in {total_time:.1f}s | Peak RAM: {peak_ram:.1f} MB")

    return {
        "candidate_recall": cand_recall,
        "avg_candidates": avg_cands,
        "med_candidates": med_cands,
        "p95_candidates": p95_cands,
        "max_candidates": max_cands,
        "reduction_ratio": reduction_ratio,
        "optimal_threshold": best_thresh,
        "validation_macro_f05": best_val_f05,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pairwise_accuracy": pairwise_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "peak_ram_mb": peak_ram,
        "runtime_s": total_time,
        "validator_passed": (val_proc.returncode == 0)
    }

if __name__ == "__main__":
    run_small_e2e_test()
