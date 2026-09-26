"""
Training pair generation module with hard-negative mining.
Produces positive pairs from ground truth and hard negatives from blocking collisions.
"""

from typing import Dict, List, Set, Tuple
import random
import numpy as np
from src.blocking import BlockingIndex
from src.features import extract_pairwise_features

def generate_training_data_from_index(
    s1_entities: Dict[str, Tuple[str, str, str, str]], # id -> (name, core, addr, country)
    ground_truth: Dict[str, Set[str]],                 # s1_id -> set of true matched s2/s3 ids
    s23_index: BlockingIndex,
    negatives_per_positive: int = 3,
    max_hard_negatives_per_s1: int = 6,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Generate training feature matrix X, binary labels y, and pair list (s1_id, cand_id).
    Mines hard negatives directly from candidate generation buckets.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    X_list = []
    y_list = []
    pairs_list = []

    for s1_id, s1_rec in s1_entities.items():
        true_matches = ground_truth.get(s1_id, set())

        cand_with_hits = s23_index.get_candidates_with_hits(
            business_name=s1_rec[0],
            business_address=s1_rec[2],
            country=s1_rec[3],
            max_candidates=25
        )
        cand_hit_dict = dict(cand_with_hits)

        # 1. Add positive pairs
        for true_id in true_matches:
            if true_id in s23_index.entity_records:
                true_rec = s23_index.entity_records[true_id]
                hits = cand_hit_dict.get(true_id, 1)
                feats = extract_pairwise_features(s1_rec, true_id, true_rec, blocking_hits=hits)
                X_list.append(feats)
                y_list.append(1)
                pairs_list.append((s1_id, true_id))

        # 2. Add hard negatives from blocking candidates
        hard_negs = [(cid, hits) for cid, hits in cand_with_hits if cid not in true_matches and cid in s23_index.entity_records]
        num_negs = min(
            len(hard_negs),
            max(2, len(true_matches) * negatives_per_positive),
            max_hard_negatives_per_s1
        )

        if hard_negs:
            selected_negs = random.sample(hard_negs, num_negs)
            for neg_id, hits in selected_negs:
                neg_rec = s23_index.entity_records[neg_id]
                feats = extract_pairwise_features(s1_rec, neg_id, neg_rec, blocking_hits=hits)
                X_list.append(feats)
                y_list.append(0)
                pairs_list.append((s1_id, neg_id))

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, 17), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty((0,), dtype=np.int32)

    return X, y, pairs_list
