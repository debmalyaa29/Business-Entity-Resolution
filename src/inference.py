"""
Inference and Submission Generation Module for Business Entity Resolution.
Generates matching_results.tsv and candidate_pairs.tsv conforming to the challenge validator.
Processes candidates and predictions in country-partitioned batches to strictly respect RAM constraints (< 1 GB RAM).
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
from src.blocking import BlockingIndex
from src.data_loader import stream_tsv_chunks
from src.features import extract_pairwise_features
from src.model import EntityMatchingModel

def get_peak_ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return usage / 1024.0
    return usage / (1024.0 * 1024.0)

def run_test_inference(
    threshold: float = 0.50,
    max_candidates_per_entity: int = 65,
    chunk_size: int = 50000
):
    start_time = time.time()
    print("=" * 60)
    print("STARTING TEST SET INFERENCE AND SUBMISSION CREATION")
    print("=" * 60)

    # 1. Load trained model
    print(f"\n[1/5] Loading trained model from {config.MODEL_PATH}...")
    model = EntityMatchingModel()
    model.load(config.MODEL_PATH)

    # Calibrated threshold from validation_metrics.json
    if config.METRICS_PATH.is_file():
        try:
            with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
                if "optimal_threshold" in metrics:
                    threshold = float(metrics["optimal_threshold"])
                    print(f"  Loaded optimal threshold from validation metrics: {threshold:.2f}")
        except Exception:
            pass

    print(f"  Applying decision threshold: {threshold:.2f}")

    # 2. Partition Test Target Records (S2 and S3) by Country on Disk
    print("\n[2/5] Partitioning Test Target Records (S2 & S3) by Country on Disk...")
    target_part_dir = config.ARTIFACT_DIR / "test_target_partitions"
    target_part_dir.mkdir(parents=True, exist_ok=True)
    target_handles: Dict[str, any] = {}
    countries_seen: Set[str] = set()

    usecols = [config.ENTITY_ID_COL, config.BUSINESS_NAME_COL, config.BUSINESS_ADDRESS_COL, config.COUNTRY_COL]
    total_target_records = 0

    for label, path in [("Test Source 2", config.TEST_SOURCE2), ("Test Source 3", config.TEST_SOURCE3)]:
        count = 0
        for chunk in stream_tsv_chunks(path, chunk_size=100000, usecols=usecols):
            for eid, name, addr, country in chunk.itertuples(index=False, name=None):
                norm_c = (str(country or "")).strip().upper() or "UNKNOWN"
                if norm_c not in target_handles:
                    f_path = target_part_dir / f"target_{norm_c}.tsv"
                    target_handles[norm_c] = open(f_path, "w", encoding="utf-8")
                    target_handles[norm_c].write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                    countries_seen.add(norm_c)

                clean_n = str(name or "").replace("\t", " ").replace("\n", " ")
                clean_a = str(addr or "").replace("\t", " ").replace("\n", " ")
                target_handles[norm_c].write(f"{eid}\t{clean_n}\t{clean_a}\t{norm_c}\n")
                count += 1
                total_target_records += 1
        print(f"    Indexed {count:,} records from {label}.")

    for h in target_handles.values():
        h.close()
    print(f"  Total target records partitioned: {total_target_records:,} across countries: {sorted(list(countries_seen))}")
    print(f"  Current Peak RAM: {get_peak_ram_mb():.1f} MB")

    # 3. Partition Test Source 1 by Country on Disk
    print("\n[3/5] Partitioning Test Source 1 Entities by Country on Disk...")
    s1_part_dir = config.ARTIFACT_DIR / "test_s1_partitions"
    s1_part_dir.mkdir(parents=True, exist_ok=True)
    s1_handles: Dict[str, any] = {}
    total_s1_count = 0

    for chunk in stream_tsv_chunks(config.TEST_SOURCE1, chunk_size=100000, usecols=usecols):
        for eid, name, addr, country in chunk.itertuples(index=False, name=None):
            norm_c = (str(country or "")).strip().upper() or "UNKNOWN"
            if norm_c not in s1_handles:
                f_path = s1_part_dir / f"s1_{norm_c}.tsv"
                s1_handles[norm_c] = open(f_path, "w", encoding="utf-8")
                s1_handles[norm_c].write("line_num\tentity_id\tbusiness_name\tbusiness_address\tcountry\n")

            clean_n = str(name or "").replace("\t", " ").replace("\n", " ")
            clean_a = str(addr or "").replace("\t", " ").replace("\n", " ")
            s1_handles[norm_c].write(f"{total_s1_count}\t{eid}\t{clean_n}\t{clean_a}\t{norm_c}\n")
            total_s1_count += 1

    for h in s1_handles.values():
        h.close()
    print(f"  Total Test Source 1 entities partitioned: {total_s1_count:,}")

    # 4. Process Each Country Partition Independently
    print("\n[4/5] Executing Candidate Generation & Inference Country-by-Country...")
    temp_pred_dir = config.ARTIFACT_DIR / "temp_predictions"
    temp_pred_dir.mkdir(parents=True, exist_ok=True)

    total_singletons = 0
    total_matches_predicted = 0
    total_candidates_generated = 0

    all_countries = sorted(list(set(list(countries_seen) + list(s1_handles.keys()))))

    for country in all_countries:
        s1_file = s1_part_dir / f"s1_{country}.tsv"
        target_file = target_part_dir / f"target_{country}.tsv"
        pred_file = temp_pred_dir / f"pred_{country}.tsv"

        if not s1_file.is_file():
            continue

        print(f"\n  --- Running Inference for Country: {country} ---")
        t_c_start = time.time()
        c_index = BlockingIndex(max_bucket_size=500)

        if target_file.is_file():
            for chunk in stream_tsv_chunks(target_file, chunk_size=100000):
                for tid, name, addr, c_code in chunk.itertuples(index=False, name=None):
                    c_index.add_entity(tid, str(name or ""), str(addr or ""), str(c_code or ""))
            print(f"    Indexed {len(c_index.entity_records):,} target records for {country} in {time.time() - t_c_start:.1f}s.")
        else:
            print(f"    Notice: No target records found for country {country} (all will be singletons).")

        out_pred = open(pred_file, "w", encoding="utf-8")
        out_pred.write("line_num\tentity_id\tmatched_ids\tcandidate_ids\n")

        country_s1_count = 0
        usecols_s1_part = ["line_num", "entity_id", "business_name", "business_address", "country"]

        for chunk in stream_tsv_chunks(s1_file, chunk_size=chunk_size, usecols=usecols_s1_part):
            for l_num, s1_id, name, addr, c_code in chunk.itertuples(index=False, name=None):
                country_s1_count += 1
                s1_rec = (str(name or ""), str(name or ""), str(addr or ""), str(c_code or ""))

                cand_hits = c_index.get_candidates_with_hits(
                    business_name=s1_rec[0],
                    business_address=s1_rec[2],
                    country=s1_rec[3],
                    max_candidates=max_candidates_per_entity
                )
                cands = [cid for cid, _ in cand_hits]
                cand_str = ",".join(cands)
                total_candidates_generated += len(cands)

                if not cands:
                    out_pred.write(f"{l_num}\t{s1_id}\t\t{cand_str}\n")
                    total_singletons += 1
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
                    out_pred.write(f"{l_num}\t{s1_id}\t\t{cand_str}\n")
                    total_singletons += 1
                    continue

                X_b = np.array(feat_batch, dtype=np.float32)
                probs = model.predict_proba(X_b)
                matched = [cand_ids[i] for i, p in enumerate(probs) if p >= threshold]
                unique_matched = list(dict.fromkeys(matched))
                match_str = ",".join(unique_matched)

                if unique_matched:
                    total_matches_predicted += len(unique_matched)
                else:
                    total_singletons += 1

                out_pred.write(f"{l_num}\t{s1_id}\t{match_str}\t{cand_str}\n")

            gc.collect()

        out_pred.close()
        print(f"    Completed {country}: processed {country_s1_count:,} entities (Peak RAM: {get_peak_ram_mb():.1f} MB)")
        del c_index
        gc.collect()

    # 5. Assemble Final Files Preserving Exact Test S1 Row Order
    print("\n[5/5] Assembling Final Submission TSVs in Exact Test Source 1 Order...")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read predictions by line_num
    results_by_line: Dict[int, Tuple[str, str, str]] = {}
    for country in all_countries:
        pred_file = temp_pred_dir / f"pred_{country}.tsv"
        if not pred_file.is_file():
            continue
        with open(pred_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    l_num = int(parts[0])
                    s1_id = parts[1]
                    m_str = parts[2]
                    c_str = parts[3]
                    results_by_line[l_num] = (s1_id, m_str, c_str)
                elif len(parts) == 3:
                    l_num = int(parts[0])
                    s1_id = parts[1]
                    m_str = parts[2]
                    results_by_line[l_num] = (s1_id, m_str, "")

    with open(config.MATCHING_RESULTS_PATH, "w", encoding="utf-8") as f_match, \
         open(config.CANDIDATE_PAIRS_PATH, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for l_num in range(total_s1_count):
            if l_num in results_by_line:
                s1_id, m_str, c_str = results_by_line[l_num]
                f_match.write(f"{s1_id}\t{m_str}\n")
                f_cand.write(f"{s1_id}\t{c_str}\n")
            else:
                # Fallback if somehow missing
                f_match.write(f"MISSING_{l_num}\t\n")
                f_cand.write(f"MISSING_{l_num}\t\n")

    # Cleanup temporary partitions
    shutil.rmtree(target_part_dir, ignore_errors=True)
    shutil.rmtree(s1_part_dir, ignore_errors=True)
    shutil.rmtree(temp_pred_dir, ignore_errors=True)

    total_time = time.time() - start_time
    print(f"\nInference Complete!")
    print(f"  Total S1 entities processed   : {total_s1_count:,}")
    print(f"  Total Singletons predicted    : {total_singletons:,} ({total_singletons / max(1, total_s1_count):.2%})")
    print(f"  Total Candidate pairs written : {total_candidates_generated:,}")
    print(f"  Total Matches predicted       : {total_matches_predicted:,}")
    print(f"  Output Files: {config.MATCHING_RESULTS_PATH} & {config.CANDIDATE_PAIRS_PATH}")
    print(f"  Total Inference Time: {total_time:.1f}s | Peak RAM: {get_peak_ram_mb():.1f} MB")

if __name__ == "__main__":
    run_test_inference()
