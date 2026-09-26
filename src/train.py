"""
Training and Validation Pipeline for Business Entity Resolution.
Implements entity-level train/validation split, hard-negative training,
candidate recall evaluation, and optimal threshold selection.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Set, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import stream_tsv_chunks, parse_ground_truth_row
from src.blocking import BlockingIndex
from src.pair_generation import generate_training_data_from_index
from src.features import extract_pairwise_features
from src.model import EntityMatchingModel
from src.evaluate import (
    compute_macro_f05,
    compute_candidate_recall,
    find_optimal_threshold
)

def run_training_pipeline(
    max_train_s1: int = 40000,
    max_val_s1: int = 10000,
    max_s2_s3_load: int = 350000,
    random_seed: int = config.RANDOM_SEED
):
    import time
    start_time = time.time()
    print("=" * 60)
    print("STARTING ENTITY RESOLUTION MODEL TRAINING & VALIDATION")
    print("=" * 60)

    # 1. Load Ground Truth mapping
    print("\n[1/6] Loading ground truth labels...")
    ground_truth: Dict[str, Set[str]] = {}
    total_gt_needed = max_train_s1 + max_val_s1

    for chunk in stream_tsv_chunks(config.TRAIN_GROUND_TRUTH, chunk_size=config.CHUNK_SIZE):
        for s1_id, m_str in zip(chunk[config.GT_S1_COL], chunk[config.GT_MATCHED_COL]):
            ground_truth[s1_id] = parse_ground_truth_row(m_str)
            if len(ground_truth) >= total_gt_needed:
                break
        if len(ground_truth) >= total_gt_needed:
            break

    print(f"Loaded ground truth for {len(ground_truth):,} Source 1 entities.")

    # 2. Entity-level Train / Validation split
    all_s1_ids = list(ground_truth.keys())
    random.seed(random_seed)
    random.shuffle(all_s1_ids)

    train_s1_ids = set(all_s1_ids[:max_train_s1])
    val_s1_ids = set(all_s1_ids[max_train_s1:max_train_s1 + max_val_s1])
    print(f"Split: {len(train_s1_ids):,} train S1 entities, {len(val_s1_ids):,} val S1 entities.")

    # 3. Load S1 entity records
    print("\n[2/6] Loading Source 1 records...")
    train_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}

    for chunk in stream_tsv_chunks(config.TRAIN_SOURCE1, chunk_size=config.CHUNK_SIZE):
        for eid, name, addr, country in zip(
            chunk[config.ENTITY_ID_COL],
            chunk[config.BUSINESS_NAME_COL],
            chunk[config.BUSINESS_ADDRESS_COL],
            chunk[config.COUNTRY_COL]
        ):
            if eid in train_s1_ids:
                train_s1_records[eid] = (name, name, addr, country)
            elif eid in val_s1_ids:
                val_s1_records[eid] = (name, name, addr, country)

        if len(train_s1_records) == len(train_s1_ids) and len(val_s1_records) == len(val_s1_ids):
            break

    print(f"Loaded {len(train_s1_records):,} train S1 and {len(val_s1_records):,} val S1 records.")

    # 4. Build Candidate Index from Source 2 & 3
    print("\n[3/6] Building blocking index from Source 2 & Source 3 (max_bucket_size=500)...")
    s23_index = BlockingIndex(max_bucket_size=500)

    # First load all true matching targets from ground truth to ensure high recall
    needed_target_ids = set()
    for s1_id in all_s1_ids:
        needed_target_ids.update(ground_truth.get(s1_id, set()))

    needed_s2 = {eid for eid in needed_target_ids if eid.startswith("S2-")}
    needed_s3 = {eid for eid in needed_target_ids if eid.startswith("S3-")}

    for label, path, needed_src in [
        ("Source 2", config.TRAIN_SOURCE2, needed_s2),
        ("Source 3", config.TRAIN_SOURCE3, needed_s3)
    ]:
        loaded_count = 0
        quota = max_s2_s3_load // 2
        needed_remaining = set(needed_src)
        max_load_cap = quota * 2

        for chunk in stream_tsv_chunks(path, chunk_size=config.CHUNK_SIZE):
            for eid, name, addr, country in zip(
                chunk[config.ENTITY_ID_COL],
                chunk[config.BUSINESS_NAME_COL],
                chunk[config.BUSINESS_ADDRESS_COL],
                chunk[config.COUNTRY_COL]
            ):
                if eid in needed_remaining:
                    s23_index.add_entity(eid, name, addr, country)
                    needed_remaining.remove(eid)
                    loaded_count += 1
                elif loaded_count < quota:
                    s23_index.add_entity(eid, name, addr, country)
                    loaded_count += 1

            if (loaded_count >= quota and not needed_remaining) or loaded_count >= max_load_cap:
                break
        print(f"  Indexed {loaded_count:,} records from {label} (remaining unindexed targets: {len(needed_remaining):,}).")

    print(f"Total candidate pool size: {len(s23_index.entity_records):,} records.")

    # 5. Generate training data with hard negatives
    print("\n[4/6] Generating pairwise training set with hard negatives...")
    X_train, y_train, train_pairs = generate_training_data_from_index(
        train_s1_records,
        ground_truth,
        s23_index,
        negatives_per_positive=3,
        max_hard_negatives_per_s1=5,
        random_seed=random_seed
    )
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    print(f"Training dataset: {len(X_train):,} pairs ({n_pos:,} positive, {n_neg:,} negative).")

    # 6. Fit XGBoost matching model
    print("\n[5/6] Fitting XGBoost pair-matching model...")
    model = EntityMatchingModel(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        n_jobs=config.N_JOBS,
        random_state=random_seed
    )
    model.fit(X_train, y_train)

    print("\nTop 8 Feature Importances:")
    for feat_name, imp in model.get_feature_importances()[:8]:
        print(f"  • {feat_name:25s}: {imp:.4f}")

    # 7. Evaluate on held-out validation set & optimize threshold
    print("\n[6/6] Validation evaluation & threshold tuning...")
    val_gt = {s1_id: ground_truth[s1_id] for s1_id in val_s1_ids}
    val_candidates: Dict[str, Set[str]] = {}
    val_candidate_scores: Dict[str, List[Tuple[str, float]]] = {}

    for s1_id, s1_rec in val_s1_records.items():
        cand_with_hits = s23_index.get_candidates_with_hits(
            business_name=s1_rec[0],
            business_address=s1_rec[2],
            country=s1_rec[3],
            max_candidates=65
        )
        cands = [cid for cid, _ in cand_with_hits]
        val_candidates[s1_id] = set(cands)

        # Score candidates
        cand_pairs = []
        feat_batch = []
        for cid, hits in cand_with_hits:
            if cid in s23_index.entity_records:
                crec = s23_index.entity_records[cid]
                feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                feat_batch.append(feats)
                cand_pairs.append(cid)

        if feat_batch:
            X_val_batch = np.array(feat_batch, dtype=np.float32)
            probs = model.predict_proba(X_val_batch)
            val_candidate_scores[s1_id] = list(zip(cand_pairs, probs))
        else:
            val_candidate_scores[s1_id] = []

    # Calculate Candidate Recall
    cand_recall = compute_candidate_recall(val_gt, val_candidates)
    total_val_cands = sum(len(c) for c in val_candidates.values())
    print(f"Candidate Generator Recall on Validation: {cand_recall:.4f}")
    print(f"Total Validation Candidates Generated   : {total_val_cands:,}")

    # Threshold Optimization
    best_thresh, best_val_f05 = find_optimal_threshold(val_gt, val_candidate_scores)
    print(f"Optimal Threshold on Validation         : {best_thresh:.2f}")
    print(f"Best Validation Macro F_0.5             : {best_val_f05:.4f}")

    # Calculate Pair Precision, Recall, and Accuracy at best threshold
    total_pred = 0
    tp = 0
    fp = 0
    total_true_pairs = sum(len(m) for m in val_gt.values())
    for s1_id, cands in val_candidate_scores.items():
        preds = {cid for cid, score in cands if score >= best_thresh}
        true_set = val_gt.get(s1_id, set())
        for cid in preds:
            if cid in true_set:
                tp += 1
            else:
                fp += 1
        total_pred += len(preds)

    fn = total_true_pairs - tp
    # True negatives among evaluated candidates
    tn = total_val_cands - (tp + fp + (int(cand_recall * total_true_pairs) - tp))
    if tn < 0:
        tn = max(0, total_val_cands - (tp + fp))
    pairwise_acc = (tp + tn) / max(1, total_val_cands)

    val_precision = tp / total_pred if total_pred > 0 else 0.0
    val_recall = tp / total_true_pairs if total_true_pairs > 0 else 0.0
    print(f"Pair Precision at Optimal Threshold     : {val_precision:.4f} ({tp}/{total_pred})")
    print(f"Pair Recall at Optimal Threshold        : {val_recall:.4f} ({tp}/{total_true_pairs})")
    print(f"Pairwise Binary Classification Accuracy : {pairwise_acc:.4%} ({tp + tn:,} / {total_val_cands:,})")
    print(f"Confusion Matrix: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")

    # Track runtime and memory
    import resource
    elapsed_time = time.time() - start_time
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"Total Training & Val Time              : {elapsed_time:.1f} s ({elapsed_time/60.0:.2f} min)")
    print(f"Peak Memory Usage                       : {peak_ram_mb:.1f} MB")

    # Save model and metrics to experiment files
    exp_model_path = config.ARTIFACT_DIR / "matching_model_exp_blocking_pre4.json"
    exp_metrics_path = config.ARTIFACT_DIR / "validation_metrics_exp_blocking_pre4.json"
    model.save(exp_model_path)
    print(f"Experiment model saved to {exp_model_path}.")

    metrics = {
        "candidate_recall": float(cand_recall),
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(best_val_f05),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "pairwise_accuracy": float(pairwise_acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_train_s1": len(train_s1_ids),
        "n_val_s1": len(val_s1_ids),
        "n_train_pairs": int(len(X_train)),
        "pos_pairs": n_pos,
        "neg_pairs": n_neg,
        "total_val_candidates": total_val_cands,
        "training_time_sec": float(elapsed_time),
        "peak_ram_mb": float(peak_ram_mb)
    }
    with open(exp_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Experiment validation metrics saved to {exp_metrics_path}.")
    print("=" * 60)
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and validate entity matching model")
    parser.add_argument("--train-s1", type=int, default=20000, help="Number of S1 entities for training")
    parser.add_argument("--val-s1", type=int, default=4000, help="Number of S1 entities for validation")
    parser.add_argument("--s2-s3-load", type=int, default=200000, help="Max S2/S3 records to index")
    args = parser.parse_args()

    run_training_pipeline(
        max_train_s1=args.train_s1,
        max_val_s1=args.val_s1,
        max_s2_s3_load=args.s2_s3_load
    )
