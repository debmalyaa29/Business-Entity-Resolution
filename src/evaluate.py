"""
Official Macro F_0.5 Evaluation and Threshold Optimization module.
Follows the exact Amazon ML Challenge 2026 evaluation specification.
"""

from typing import Dict, Set, List, Tuple
import numpy as np

def compute_entity_f05(true_matches: Set[str], predicted_matches: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    Includes explicit singleton credit and penalty.
    """
    # Case 1: True singleton (no matches in ground truth)
    if not true_matches:
        return 1.0 if not predicted_matches else 0.0

    # Case 2: True matches exist but model predicted nothing
    if not predicted_matches:
        return 0.0

    intersection = len(true_matches & predicted_matches)
    if intersection == 0:
        return 0.0

    precision = intersection / len(predicted_matches)
    recall = intersection / len(true_matches)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05

def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in ground truth.
    """
    if not ground_truth:
        return 0.0

    total_f05 = 0.0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_f05 += compute_entity_f05(true_set, pred_set)

    return total_f05 / len(ground_truth)

def compute_candidate_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> float:
    """
    Compute recall ceiling of the candidate generation stage.
    Recall = (Total true matches captured in candidates) / (Total true matches across all entities).
    """
    total_true = 0
    captured_true = 0

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        total_true += len(true_set)
        cand_set = candidates.get(s1_id, set())
        captured_true += len(true_set & cand_set)

    return captured_true / total_true if total_true > 0 else 1.0

def find_optimal_threshold(
    ground_truth: Dict[str, Set[str]],
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    threshold_range: List[float] = None
) -> Tuple[float, float]:
    """
    Grid search to find the probability threshold maximizing macro F_0.5 on validation.
    Returns (best_threshold, best_macro_f05).
    """
    if threshold_range is None:
        threshold_range = [round(t, 2) for t in np.arange(0.20, 0.90, 0.05)]

    best_thresh = 0.50
    best_score = -1.0

    for thresh in threshold_range:
        current_preds = {}
        for s1_id, cands in candidate_scores.items():
            pred_set = {cid for cid, score in cands if score >= thresh}
            current_preds[s1_id] = pred_set

        score = compute_macro_f05(ground_truth, current_preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score
