"""
Memory-efficient chunked data loader for Business Entity Resolution.
"""

import csv
import pandas as pd
from typing import Iterator, Optional, Set, Dict, List
from pathlib import Path
from src import config

def stream_tsv_chunks(
    filepath: Path,
    chunk_size: int = config.CHUNK_SIZE,
    usecols: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    """
    Stream a TSV file in chunks to prevent memory exhaustion.
    Uses QUOTE_NONE and string dtype for reliability.
    """
    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        chunksize=chunk_size,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip"
    ):
        yield chunk

def load_tsv_sample(
    filepath: Path,
    nrows: int = 10000,
    usecols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load a small sample from a TSV file for profiling/testing."""
    return pd.read_csv(
        filepath,
        sep="\t",
        nrows=nrows,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip"
    )

def parse_ground_truth_row(matched_str: str) -> Set[str]:
    """Parse comma-separated matched entity IDs into a set."""
    if not matched_str or not matched_str.strip():
        return set()
    return {mid.strip() for mid in matched_str.split(",") if mid.strip()}
