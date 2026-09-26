"""
Configuration module for Amazon ML Challenge Business Entity Resolution.
Supports environment variables with sensible defaults.
"""

import os
from pathlib import Path

# Base project directory (student_resource)
BASE_DIR = Path(__file__).resolve().parent.parent

# Configurable paths via environment variables
DATA_DIR = Path(os.environ.get("AMAZON_ML_DATA_DIR", BASE_DIR / "dataset"))
OUTPUT_DIR = Path(os.environ.get("AMAZON_ML_OUTPUT_DIR", BASE_DIR / "output"))
ARTIFACT_DIR = Path(os.environ.get("AMAZON_ML_ARTIFACT_DIR", BASE_DIR / "artifacts"))

# Ensure output and artifact directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Submission output files
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Artifact file paths
MODEL_PATH = ARTIFACT_DIR / "matching_model.json"
METRICS_PATH = ARTIFACT_DIR / "validation_metrics.json"

# Pipeline execution parameters
CHUNK_SIZE = int(os.environ.get("AMAZON_ML_CHUNK_SIZE", 50000))
N_JOBS = int(os.environ.get("AMAZON_ML_N_JOBS", min(8, os.cpu_count() or 4)))
USE_GPU = os.environ.get("AMAZON_ML_USE_GPU", "false").lower() in ("true", "1", "yes")

# Random seed for reproducibility
RANDOM_SEED = 42

# Column specifications
ENTITY_ID_COL = "entity_id"
BUSINESS_NAME_COL = "business_name"
BUSINESS_ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GT_S1_COL = "source1_entity_id"
GT_MATCHED_COL = "matched_entity_ids"
CANDIDATE_MATCHED_COL = "candidate_entity_ids"
