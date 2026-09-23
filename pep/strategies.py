"""
Selection Strategies for choosing which criterion to ask about next.

Each strategy takes:
- model: A likelihood model
- asked: Set of already-asked criterion indices
- observed_has: List of (criterion_idx, value) for criteria user has
- observed_not_has: List of criterion indices user doesn't have

Returns: Index of next criterion to ask about
"""

import numpy as np
from typing import Set, List, Tuple, Callable


def binary_entropy(p: float) -> float:
    """Compute binary entropy H(p)."""
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)


def select_random(
    model,
    asked: Set[int],
    observed_has: List[Tuple[int, float]],
    observed_not_has: List[int]
) -> int:
    """Random selection: uniformly random among unasked criteria."""
    unasked = [c for c in range(model.n_criteria) if c not in asked]
    return np.random.choice(unasked) if unasked else 0


def select_uncertainty(
    model,
    asked: Set[int],
    observed_has: List[Tuple[int, float]],
    observed_not_has: List[int]
) -> int:
    """Uncertainty sampling: select criterion with P(has) closest to 0.5."""
    pred_has, _ = model.predict(observed_has, observed_not_has)
    scores = -np.abs(pred_has - 0.5)  # Higher is better (closer to 0.5)
    for c in asked:
        scores[c] = -float('inf')
    return np.argmax(scores)


def select_fixed(
    model,
    asked: Set[int],
    observed_has: List[Tuple[int, float]],
    observed_not_has: List[int]
) -> int:
    """Fixed order: based on prior_has * prior_val + entropy (computed once)."""
    scores = model.prior_has * model.prior_val + np.array([binary_entropy(p) for p in model.prior_has])
    for c in asked:
        scores[c] = -float('inf')
    return np.argmax(scores)


def select_greedy_ev(
    model,
    asked: Set[int],
    observed_has: List[Tuple[int, float]],
    observed_not_has: List[int]
) -> int:
    """Greedy expected value: select criterion with highest P(has) * E[value]."""
    pred_has, pred_val = model.predict(observed_has, observed_not_has)
    scores = pred_has * pred_val
    for c in asked:
        scores[c] = -float('inf')
    return np.argmax(scores)


def select_binary_ig(
    model,
    asked: Set[int],
    observed_has: List[Tuple[int, float]],
    observed_not_has: List[int]
) -> int:
    """Binary information gain: select criterion that maximizes expected entropy reduction."""
    pred_has, pred_val = model.predict(observed_has, observed_not_has)
    unasked = [c for c in range(model.n_criteria) if c not in asked]

    if not unasked:
        return 0

    candidates = []
    for c in unasked:
        p_hit, p_miss = pred_has[c], 1 - pred_has[c]

        # Skip nearly-certain predictions
        if p_hit <= 0.01 or p_hit >= 0.99:
            candidates.append((c, -1, p_hit))
            continue

        remaining = [i for i in unasked if i != c]

        # Expected entropy if user HAS criterion
        obs_hit = observed_has + [(c, pred_val[c])]
        pred_hit, _ = model.predict(obs_hit, observed_not_has)
        H_hit = sum(binary_entropy(pred_hit[i]) for i in remaining)

        # Expected entropy if user DOESN'T HAVE criterion
        obs_miss = observed_not_has + [c]
        pred_miss, _ = model.predict(observed_has, obs_miss)
        H_miss = sum(binary_entropy(pred_miss[i]) for i in remaining)

        # Information gain
        H_before = sum(binary_entropy(pred_has[i]) for i in remaining)
        ig = H_before - (p_hit * H_hit + p_miss * H_miss)

        candidates.append((c, ig, p_hit))

    # Select best valid candidate (positive IG)
    valid = [(c, ig, p) for c, ig, p in candidates if ig >= 0]

    if valid:
        return max(valid, key=lambda x: x[1])[0]
    else:
        # Fallback: select most uncertain
        return min(candidates, key=lambda x: abs(x[2] - 0.5))[0]


# Registry of all strategies
STRATEGY_REGISTRY = {
    'Random': select_random,
    'Uncertainty': select_uncertainty,
    'Fixed': select_fixed,
    'Greedy EV': select_greedy_ev,
    'Binary IG': select_binary_ig,
}

# Grouping by type
RANDOM_STRATEGIES = ['Random']
FIXED_STRATEGIES = ['Fixed']
ADAPTIVE_STRATEGIES = ['Uncertainty', 'Greedy EV', 'Binary IG']
