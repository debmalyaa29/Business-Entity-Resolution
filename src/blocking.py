"""
Blocking and Candidate Generation Module for Business Entity Resolution.
Constructs inverted indices over multi-pass blocking keys to retrieve high-recall candidate sets.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Iterator
from src.text_normalization import (
    clean_business_name,
    clean_address,
    extract_tokens,
    extract_distinctive_name_tokens,
    extract_distinctive_address_tokens,
    extract_numbers
)

def generate_blocking_keys(
    business_name: str,
    business_address: str,
    country: str
) -> Set[str]:
    """
    Generate multiple robust blocking keys for an entity.
    All keys are scoped by country to ensure cross-country separation.
    """
    keys = set()
    norm_country = (country or "").strip().upper()
    if not norm_country:
        norm_country = "UNKNOWN"

    clean_name, core_name = clean_business_name(business_name)
    clean_addr = clean_address(business_address)
    name_tokens = extract_distinctive_name_tokens(core_name)
    addr_tokens = extract_tokens(clean_addr, min_len=3)
    dist_addr_tokens = extract_distinctive_address_tokens(clean_addr)
    numbers = extract_numbers(clean_addr)

    # 1. Exact core name in country (high precision)
    if core_name and len(core_name) >= 3:
        keys.add(f"NAME_CORE:{norm_country}:{core_name}")

    # 2. Distinctive name tokens (for word order changes, variations)
    for tok in name_tokens:
        if len(tok) >= 4:
            keys.add(f"NAME_TOK:{norm_country}:{tok}")

    # 3. PRE4 / 4-character normalized core name prefix (captures spelling/suffix variants)
    if len(core_name) >= 4:
        keys.add(f"PRE4:{norm_country}:{core_name[:4]}")

    # 4. Name prefix (3 chars) + Address number/PIN (if available)
    prefix3 = core_name[:3] if len(core_name) >= 3 else core_name
    if prefix3:
        for num in numbers:
            keys.add(f"PRE_NUM:{norm_country}:{prefix3}:{num}")

    # 5. Distinctive address tokens + numbers/PINs
    for atok in dist_addr_tokens:
        for num in numbers:
            keys.add(f"ADDR_TOK_NUM:{norm_country}:{atok}:{num}")
        if len(atok) >= 5:
            keys.add(f"ADDR_TOK:{norm_country}:{atok}")

    # 6. Clean address exact match (for exact location matches)
    if clean_addr and len(clean_addr) >= 8:
        keys.add(f"ADDR_EXACT:{norm_country}:{clean_addr[:30]}")

    return keys

class BlockingIndex:
    """
    Inverted index storing entity IDs under blocking keys with frequency capping.
    """
    def __init__(self, max_bucket_size: int = 150):
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.max_bucket_size = max_bucket_size
        self.entity_records: Dict[str, Tuple[str, str, str, str]] = {}  # id -> (name, core_name, addr, country)

    def add_entity(self, entity_id: str, business_name: str, business_address: str, country: str):
        """Register an entity and index its blocking keys."""
        _, core_name = clean_business_name(business_name)
        clean_addr = clean_address(business_address)
        self.entity_records[entity_id] = (core_name, clean_addr)
        keys = generate_blocking_keys(business_name, business_address, country)

        for key in keys:
            bucket = self.index[key]
            # Cap bucket size to avoid popular terms exploding memory/comparisons
            if len(bucket) < self.max_bucket_size:
                bucket.append(entity_id)

    def get_candidates_with_hits(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Retrieve candidate entity IDs along with their blocking key collision count.
        """
        keys = generate_blocking_keys(business_name, business_address, country)
        candidate_hit_counts: Dict[str, int] = defaultdict(int)

        for key in keys:
            if key in self.index:
                for cand_id in self.index[key]:
                    candidate_hit_counts[cand_id] += 1

        if not candidate_hit_counts:
            return []

        # Sort candidates by number of matching blocking keys (highest first)
        ranked = sorted(candidate_hit_counts.items(), key=lambda x: x[1], reverse=True)
        return ranked[:max_candidates]

    def get_candidates(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 50
    ) -> List[str]:
        """
        Retrieve candidate entity IDs matching the query entity's blocking keys.
        Ranks by key collision count.
        """
        ranked = self.get_candidates_with_hits(
            business_name=business_name,
            business_address=business_address,
            country=country,
            max_candidates=max_candidates
        )
        return [cand_id for cand_id, _ in ranked]
