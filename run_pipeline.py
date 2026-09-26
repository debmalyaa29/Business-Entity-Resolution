#!/usr/bin/env python3
"""
Top-level entry point to execute the Amazon ML Challenge Business Entity Resolution pipeline.
Usage:
    python run_pipeline.py [--dry-run] [--skip-train]
"""

import sys
from pathlib import Path

# Add student_resource directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import execute_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Execute on a fast sample subset")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training and run inference only")
    args = parser.parse_args()

    success = execute_pipeline(dry_run=args.dry_run, skip_train=args.skip_train)
    sys.exit(0 if success else 1)
