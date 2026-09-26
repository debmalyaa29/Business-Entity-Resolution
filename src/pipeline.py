"""
End-to-End Pipeline Orchestrator for Business Entity Resolution.
Supports both fast small-scale dry runs and production-scale execution.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src import config
from src.train import run_training_pipeline
from src.inference import run_test_inference
from src.create_submission import package_submission

def run_validator() -> bool:
    """Run official submission validator."""
    print("\n" + "=" * 60)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 60)

    validator_script = config.BASE_DIR / "utils" / "validate_submission.py"
    if not validator_script.is_file():
        print(f"Validator script not found at {validator_script}")
        return False

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(config.MATCHING_RESULTS_PATH),
        "--candidate", str(config.CANDIDATE_PAIRS_PATH),
        "--test-dir", str(config.TEST_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode == 0

def execute_pipeline(dry_run: bool = False, skip_train: bool = False):
    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Mode: {'DRY RUN (Small Scale)' if dry_run else 'FULL PRODUCTION RUN'}")
    print("=" * 60)

    # Step 1: Training & Validation
    if not skip_train:
        if dry_run:
            print("\n>>> STAGE 1: Training on dry-run sample...")
            run_training_pipeline(
                max_train_s1=2000,
                max_val_s1=500,
                max_s2_s3_load=20000
            )
        else:
            print("\n>>> STAGE 1: Training on full production sample...")
            run_training_pipeline(
                max_train_s1=50000,
                max_val_s1=10000,
                max_s2_s3_load=400000
            )
    else:
        print("\n>>> STAGE 1: Skipped (using existing trained model).")

    # Step 2: Test Inference
    print("\n>>> STAGE 2: Test Inference...")
    run_test_inference()

    # Step 3: Run official validation
    print("\n>>> STAGE 3: Submission Verification...")
    passed = run_validator()
    if not passed:
        print("[!] Validation failed. Please check errors above.")
        return False

    # Step 4: Package final submission zip
    print("\n>>> STAGE 4: Packaging submission...")
    package_submission()

    print("\n>>> PIPELINE EXECUTION SUCCESSFUL!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run small-scale dry run")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing model")
    args = parser.parse_args()

    execute_pipeline(dry_run=args.dry_run, skip_train=args.skip_train)
