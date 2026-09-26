"""
Blocking and Candidate Generation Module for Business Entity Resolution.
Constructs inverted indices over multi-pass blocking keys to retrieve high-recall candidate sets.
"""

from collections import defaultdict
from pathlib import Path
import sqlite3
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

    # 2. Distinctive name tokens (for word order changes, variations - top 2 longest)
    sorted_name_toks = sorted(name_tokens, key=len, reverse=True)[:2]
    for tok in sorted_name_toks:
        if len(tok) >= 4:
            keys.add(f"NAME_TOK:{norm_country}:{tok}")

    # 3. PRE4 / 4-character normalized core name prefix (captures spelling/suffix variants)
    if len(core_name) >= 4:
        keys.add(f"PRE4:{norm_country}:{core_name[:4]}")

    # 4. Name prefix (3 chars) + Address number/PIN (if available - top 2 numbers)
    prefix3 = core_name[:3] if len(core_name) >= 3 else core_name
    sorted_nums = sorted(numbers, key=len, reverse=True)[:2]
    if prefix3:
        for num in sorted_nums:
            keys.add(f"PRE_NUM:{norm_country}:{prefix3}:{num}")

    # 5. Distinctive address tokens + numbers/PINs (top 2 distinctive tokens & top 2 numbers)
    sorted_addr_toks = sorted(dist_addr_tokens, key=len, reverse=True)[:2]
    for atok in sorted_addr_toks:
        for num in sorted_nums:
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

class CountryPartitionedBlockingIndex:
    """
    Partitioned inverted index storing candidates grouped by country.
    Ensures cross-country isolation and strictly limits peak memory by
    allowing per-country loading, querying, and eviction.
    """
    def __init__(self, max_bucket_size: int = 500):
        self.max_bucket_size = max_bucket_size
        self.partitions: Dict[str, BlockingIndex] = {}

    def get_partition(self, country: str) -> BlockingIndex:
        norm = (country or "").strip().upper() or "UNKNOWN"
        if norm not in self.partitions:
            self.partitions[norm] = BlockingIndex(max_bucket_size=self.max_bucket_size)
        return self.partitions[norm]

    def add_entity(self, entity_id: str, business_name: str, business_address: str, country: str):
        part = self.get_partition(country)
        part.add_entity(entity_id, business_name, business_address, country)

    def get_candidates_with_hits(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 65
    ) -> List[Tuple[str, int]]:
        norm = (country or "").strip().upper() or "UNKNOWN"
        if norm not in self.partitions:
            return []
        return self.partitions[norm].get_candidates_with_hits(
            business_name=business_name,
            business_address=business_address,
            country=country,
            max_candidates=max_candidates
        )

    def get_candidates(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 65
    ) -> List[str]:
        ranked = self.get_candidates_with_hits(
            business_name=business_name,
            business_address=business_address,
            country=country,
            max_candidates=max_candidates
        )
        return [cand_id for cand_id, _ in ranked]

    def get_entity_record(self, entity_id: str, country: str = ""):
        if country:
            norm = (country or "").strip().upper() or "UNKNOWN"
            if norm in self.partitions and entity_id in self.partitions[norm].entity_records:
                return self.partitions[norm].entity_records[entity_id]
        for part in self.partitions.values():
            if entity_id in part.entity_records:
                return part.entity_records[entity_id]
        return None

    def clear_country(self, country: str):
        norm = (country or "").strip().upper() or "UNKNOWN"
        if norm in self.partitions:
            del self.partitions[norm]

    @property
    def total_entities(self) -> int:
        return sum(len(p.entity_records) for p in self.partitions.values())

class DiskBackedBlockingIndex:
    """
    High-performance disk-backed blocking index powered by SQLite.
    Stores records and inverted index on disk, keeping RAM bounded (< 100 MB)
    even for tens of millions of records.
    """
    def __init__(self, db_path: Path, max_bucket_size: int = 500):
        self.db_path = Path(db_path)
        self.max_bucket_size = max_bucket_size
        self._init_db()

    def _init_db(self):
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                pass
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = -64000")  # 64 MB page cache
        self.conn.execute("CREATE TABLE records (id TEXT PRIMARY KEY, name TEXT, addr TEXT)")
        self.conn.execute("CREATE TABLE blk (key TEXT, id TEXT)")

    def add_entities_batch(self, batch: List[Tuple[str, str, str, str]]):
        recs = []
        blks = []
        for eid, name, addr, country in batch:
            recs.append((eid, str(name or ""), str(addr or "")))
            keys = generate_blocking_keys(name, addr, country)
            for k in keys:
                blks.append((k, eid))
        self.conn.executemany("INSERT OR IGNORE INTO records VALUES (?, ?, ?)", recs)
        self.conn.executemany("INSERT INTO blk VALUES (?, ?)", blks)

    def finalize_index(self):
        self.conn.execute("CREATE INDEX idx_blk_key ON blk(key)")
        self.conn.commit()

    def get_candidates_with_hits(
        self,
        business_name: str,
        business_address: str,
        country: str,
        max_candidates: int = 65
    ) -> List[Tuple[str, int]]:
        keys = list(generate_blocking_keys(business_name, business_address, country))
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        sql = f"""
            SELECT id, COUNT(*) as hits
            FROM blk
            WHERE key IN ({placeholders})
            GROUP BY id
            ORDER BY hits DESC
            LIMIT {max_candidates}
        """
        cur = self.conn.execute(sql, keys)
        return cur.fetchall()

    def get_records_batch(self, entity_ids: List[str]) -> Dict[str, Tuple[str, str]]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        sql = f"SELECT id, name, addr FROM records WHERE id IN ({placeholders})"
        cur = self.conn.execute(sql, entity_ids)
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def close(self):
        self.conn.close()

