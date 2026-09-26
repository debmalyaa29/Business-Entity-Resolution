"""
Dataset profiling module for Amazon ML Challenge Business Entity Resolution.
Computes real dataset statistics using chunked streaming.
"""

import sys
from collections import Counter
from pathlib import Path
from src import config
from src.data_loader import stream_tsv_chunks, parse_ground_truth_row

def profile_dataset(sample_mode: bool = False, max_chunks: int = 4):
    """
    Profile the train and test datasets.
    If sample_mode is True, profiles max_chunks * chunk_size rows per file.
    """
    print("=" * 60)
    print("AMAZON ML CHALLENGE — DATASET PROFILING")
    print("=" * 60)

    files_to_profile = [
        ("Train Source 1", config.TRAIN_SOURCE1),
        ("Train Source 2", config.TRAIN_SOURCE2),
        ("Train Source 3", config.TRAIN_SOURCE3),
        ("Test Source 1", config.TEST_SOURCE1),
        ("Test Source 2", config.TEST_SOURCE2),
        ("Test Source 3", config.TEST_SOURCE3),
    ]

    for label, path in files_to_profile:
        if not path.is_file():
            print(f"[-] {label}: File not found at {path}")
            continue

        total_rows = 0
        country_counts = Counter()
        empty_name_count = 0
        empty_addr_count = 0
        empty_country_count = 0
        chunk_idx = 0

        for chunk in stream_tsv_chunks(path, chunk_size=config.CHUNK_SIZE):
            chunk_idx += 1
            n = len(chunk)
            total_rows += n

            if config.COUNTRY_COL in chunk.columns:
                country_counts.update(chunk[config.COUNTRY_COL].tolist())
            if config.BUSINESS_NAME_COL in chunk.columns:
                empty_name_count += (chunk[config.BUSINESS_NAME_COL] == "").sum()
            if config.BUSINESS_ADDRESS_COL in chunk.columns:
                empty_addr_count += (chunk[config.BUSINESS_ADDRESS_COL] == "").sum()
            if config.COUNTRY_COL in chunk.columns:
                empty_country_count += (chunk[config.COUNTRY_COL] == "").sum()

            if sample_mode and chunk_idx >= max_chunks:
                break

        mode_str = f" (Sample: {chunk_idx * config.CHUNK_SIZE} rows)" if sample_mode else " (Full)"
        print(f"\n[+] {label}{mode_str}")
        print(f"    Total Rows Processed : {total_rows:,}")
        print(f"    Empty Names          : {empty_name_count} ({empty_name_count / max(1, total_rows):.2%})")
        print(f"    Empty Addresses      : {empty_addr_count} ({empty_addr_count / max(1, total_rows):.2%})")
        print(f"    Empty Countries      : {empty_country_count} ({empty_country_count / max(1, total_rows):.2%})")
        print(f"    Country Distribution : {dict(country_counts.most_common(10))}")

    # Profile Ground Truth
    if config.TRAIN_GROUND_TRUTH.is_file():
        print(f"\n[+] Ground Truth Profile")
        gt_rows = 0
        match_distribution = Counter()
        s2_matches = 0
        s3_matches = 0
        chunk_idx = 0

        for chunk in stream_tsv_chunks(config.TRAIN_GROUND_TRUTH, chunk_size=config.CHUNK_SIZE):
            chunk_idx += 1
            for matched_str in chunk[config.GT_MATCHED_COL]:
                gt_rows += 1
                matched_set = parse_ground_truth_row(matched_str)
                count = len(matched_set)
                match_distribution[count] += 1
                for mid in matched_set:
                    if mid.startswith("S2-"):
                        s2_matches += 1
                    elif mid.startswith("S3-"):
                        s3_matches += 1

            if sample_mode and chunk_idx >= max_chunks:
                break

        print(f"    Ground Truth Rows   : {gt_rows:,}")
        print(f"    Singletons (0 match): {match_distribution[0]:,} ({match_distribution[0]/max(1, gt_rows):.2%})")
        print(f"    1 match             : {match_distribution[1]:,} ({match_distribution[1]/max(1, gt_rows):.2%})")
        print(f"    2 matches           : {match_distribution[2]:,} ({match_distribution[2]/max(1, gt_rows):.2%})")
        print(f"    3+ matches          : {sum(v for k, v in match_distribution.items() if k >= 3):,}")
        print(f"    Total S2 Links      : {s2_matches:,}")
        print(f"    Total S3 Links      : {s3_matches:,}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    is_sample = "--full" not in sys.argv
    profile_dataset(sample_mode=is_sample)
