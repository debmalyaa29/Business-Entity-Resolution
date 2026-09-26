"""
Pairwise feature extraction module for Business Entity Resolution.
Computes lightweight, high-signal similarity features between Source 1 and candidate records.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np

def char_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def overlap_coefficient(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Szymkiewicz-Simpson overlap coefficient."""
    if not set1 or not set2:
        return 0.0
    min_len = min(len(set1), len(set2))
    return len(set1 & set2) / min_len if min_len > 0 else 0.0

FEATURE_NAMES = [
    "exact_name_match",
    "exact_core_name_match",
    "exact_address_match",
    "name_char_jaccard_3",
    "name_token_jaccard",
    "name_token_overlap",
    "addr_char_jaccard_3",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_number_jaccard",
    "name_prefix_3_match",
    "name_prefix_5_match",
    "name_len_diff",
    "addr_len_diff",
    "country_match",
    "is_source3",
    "blocking_hits"
]

def extract_pairwise_features(
    s1_record: Tuple[str, str, str, str],  # (clean_name, core_name, clean_addr, country)
    cand_id: str,
    cand_record: Tuple[str, str, str, str], # (clean_name, core_name, clean_addr, country)
    blocking_hits: int = 1
) -> List[float]:
    """
    Extract a dense feature vector for a pair (Source 1, Candidate).
    """
    s1_name, s1_core, s1_addr, s1_country = s1_record
    if len(cand_record) == 4:
        c_name, c_core, c_addr, c_country = cand_record
    else:
        c_core, c_addr = cand_record
        c_name = c_core
        c_country = s1_country

    # 1. Exact matches
    exact_name = 1.0 if (s1_name and s1_name == c_name) else 0.0
    exact_core = 1.0 if (s1_core and s1_core == c_core) else 0.0
    exact_addr = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0

    # 2. Token sets
    s1_name_toks = set(s1_core.split())
    c_name_toks = set(c_core.split())
    name_tok_jaccard = jaccard_similarity(s1_name_toks, c_name_toks)
    name_tok_overlap = overlap_coefficient(s1_name_toks, c_name_toks)

    s1_addr_toks = set(s1_addr.split())
    c_addr_toks = set(c_addr.split())
    addr_tok_jaccard = jaccard_similarity(s1_addr_toks, c_addr_toks)
    addr_tok_overlap = overlap_coefficient(s1_addr_toks, c_addr_toks)

    # 3. Char n-gram similarities
    s1_name_ngrams = char_ngrams(s1_core, 3)
    c_name_ngrams = char_ngrams(c_core, 3)
    name_char_jaccard = jaccard_similarity(s1_name_ngrams, c_name_ngrams)

    s1_addr_ngrams = char_ngrams(s1_addr, 3)
    c_addr_ngrams = char_ngrams(c_addr, 3)
    addr_char_jaccard = jaccard_similarity(s1_addr_ngrams, c_addr_ngrams)

    # 4. Number/PIN code overlap
    s1_nums = {w for w in s1_addr_toks if w.isdigit()}
    c_nums = {w for w in c_addr_toks if w.isdigit()}
    addr_num_jaccard = jaccard_similarity(s1_nums, c_nums)

    # 5. Prefix matches
    prefix3_match = 1.0 if (s1_core[:3] and s1_core[:3] == c_core[:3]) else 0.0
    prefix5_match = 1.0 if (s1_core[:5] and s1_core[:5] == c_core[:5]) else 0.0

    # 6. Length differences (normalized)
    name_len_diff = abs(len(s1_core) - len(c_core)) / max(1, len(s1_core) + len(c_core))
    addr_len_diff = abs(len(s1_addr) - len(c_addr)) / max(1, len(s1_addr) + len(c_addr))

    # 7. Metadata features
    country_match = 1.0 if s1_country == c_country else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return [
        exact_name,
        exact_core,
        exact_addr,
        name_char_jaccard,
        name_tok_jaccard,
        name_tok_overlap,
        addr_char_jaccard,
        addr_tok_jaccard,
        addr_tok_overlap,
        addr_num_jaccard,
        prefix3_match,
        prefix5_match,
        name_len_diff,
        addr_len_diff,
        country_match,
        is_source3,
        float(blocking_hits)
    ]
