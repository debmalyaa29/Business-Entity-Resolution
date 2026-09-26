"""
Inference and Submission Generation Module for Business Entity Resolution.
Generates matching_results.tsv and candidate_pairs.tsv conforming to the challenge validator.
Processes candidates and predictions in streaming batches to strictly respect RAM constraints.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

from src import config
from src.data_loader import stream_tsv_chunks
from src.blocking import BlockingIndex
from src.features import extract_pairwise_features
from src.model import EntityMatchingModel

def run_test_inference(
    threshold: float = 0.50,
    max_candidates_per_entity: int = 30,
    batch_size: int = 5000
):
    print("=" * 60)
    print("STARTING TEST SET INFERENCE AND SUBMISSION CREATION")
    print("=" * 60)

    # 1. Load trained model
    print(f"\n[1/4] Loading trained model from {config.MODEL_PATH}...")
    model = EntityMatchingModel()
    model.load(config.MODEL_PATH)

    # Check for calibrated threshold in validation_metrics.json
    if config.METRICS_PATH.is_file():
        try:
            with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                if "optimal_threshold" in metrics:
                    threshold = float(metrics["optimal_threshold"])
                    print(f"Loaded optimal threshold from validation metrics: {threshold:.2f}")
        except Exception:
            pass

    print(f"Applying decision threshold: {threshold:.2f}")

    # 2. Build Candidate Index from Test Source 2 and Source 3
    print("\n[2/4] Indexing Test Source 2 and Source 3...")
    test_index = BlockingIndex(max_bucket_size=120)

    usecols = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]
    for label, path in [("Test Source 2", config.TEST_SOURCE2), ("Test Source 3", config.TEST_SOURCE3)]:
        count = 0
        for chunk in stream_tsv_chunks(path, chunk_size=config.CHUNK_SIZE, usecols=usecols):
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                test_index.add_entity(eid, name, addr, country)
                count += 1
        print(f"  Indexed {count:,} records from {label}.")

    print(f"Total candidate pool indexed: {len(test_index.entity_records):,} records.")

    # 3. Stream Test Source 1 and generate predictions
    print(f"\n[3/4] Streaming Test Source 1 and generating candidate pairs & matches...")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matching_file = open(config.MATCHING_RESULTS_PATH, "w", encoding="utf-8")
    candidate_file = open(config.CANDIDATE_PAIRS_PATH, "w", encoding="utf-8")

    # Write headers
    matching_file.write("source1_entity_id\tmatched_entity_ids\n")
    candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    total_singletons = 0
    total_matches_predicted = 0
    total_candidates_written = 0

    s1_batch_records: List[Tuple[str, Tuple[str, str, str, str]]] = []

    def process_s1_batch(batch):
        nonlocal total_singletons, total_matches_predicted, total_candidates_written

        for s1_id, s1_rec in batch:
            cand_with_hits = test_index.get_candidates_with_hits(
                business_name=s1_rec[0],
                business_address=s1_rec[2],
                country=s1_rec[3],
                max_candidates=max_candidates_per_entity
            )
            cands = [cid for cid, _ in cand_with_hits]

            # Write candidates (every final match must be in candidates)
            cand_str = ",".join(cands)
            candidate_file.write(f"{s1_id}\t{cand_str}\n")
            total_candidates_written += len(cands)

            if not cands:
                # Singleton
                matching_file.write(f"{s1_id}\t\n")
                total_singletons += 1
                continue

            # Feature extraction for candidates
            feat_batch = []
            cand_ids = []
            for cid, hits in cand_with_hits:
                if cid in test_index.entity_records:
                    crec = test_index.entity_records[cid]
                    feats = extract_pairwise_features(s1_rec, cid, crec, blocking_hits=hits)
                    feat_batch.append(feats)
                    cand_ids.append(cid)

            if not feat_batch:
                matching_file.write(f"{s1_id}\t\n")
                total_singletons += 1
                continue

            X_batch = np.array(feat_batch, dtype=np.float32)
            probs = model.predict_proba(X_batch)

            matched_ids = [
                cand_ids[i] for i, prob in enumerate(probs)
                if prob >= threshold
            ]

            if matched_ids:
                # Deduplicate preserving order
                seen = set()
                unique_matched = []
                for mid in matched_ids:
                    if mid not in seen:
                        seen.add(mid)
                        unique_matched.append(mid)
                match_str = ",".join(unique_matched)
                matching_file.write(f"{s1_id}\t{match_str}\n")
                total_matches_predicted += len(unique_matched)
            else:
                matching_file.write(f"{s1_id}\t\n")
                total_singletons += 1

    for chunk in stream_tsv_chunks(config.TEST_SOURCE1, chunk_size=config.CHUNK_SIZE, usecols=usecols):
        for s1_id, name, addr, country in chunk.itertuples(index=False, name=None):
            s1_batch_records.append((s1_id, (name, name, addr, country)))
            total_s1_processed += 1

            if len(s1_batch_records) >= batch_size:
                process_s1_batch(s1_batch_records)
                s1_batch_records = []
                if total_s1_processed % 50000 == 0:
                    print(f"  Processed {total_s1_processed:,} Source 1 entities...")

    if s1_batch_records:
        process_s1_batch(s1_batch_records)

    matching_file.close()
    candidate_file.close()

    print("\n[4/4] Inference Complete!")
    print(f"  Total Source 1 entities processed : {total_s1_processed:,}")
    print(f"  Total Singletons predicted (empty) : {total_singletons:,} ({total_singletons/max(1, total_s1_processed):.2%})")
    print(f"  Total Candidate pairs generated    : {total_candidates_written:,}")
    print(f"  Total Matches predicted            : {total_matches_predicted:,}")
    print(f"  matching_results.tsv written to    : {config.MATCHING_RESULTS_PATH}")
    print(f"  candidate_pairs.tsv written to     : {config.CANDIDATE_PAIRS_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    run_test_inference()
