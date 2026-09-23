"""
Additional filtering utilities for per-problem min_users filtering.
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List


def _filter_rubric(value: Any, removal_set: set[str]) -> Any:
    """Keep rubric entries synchronized with the filtered preference profile."""
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("evaluation_criteria"), list):
        value["evaluation_criteria"] = [
            item
            for item in value["evaluation_criteria"]
            if item.get("preference") not in removal_set
        ]
        return value
    return {key: item for key, item in value.items() if key not in removal_set}


def filter_with_min_users(
    records: List[Dict],
    per_problem_max: float = 0.10,
    min_users_per_problem: int = 3,
    verbose: bool = True
) -> List[Dict]:
    """
    Filter with per-problem max AND minimum users threshold.

    A criterion is kept in a problem only if:
    1. It appears in < per_problem_max fraction of users, AND
    2. It appears in >= min_users_per_problem users

    Args:
        records: List of record dictionaries
        per_problem_max: Remove if >= this fraction within problem (default 0.10 = 10%)
        min_users_per_problem: Remove if fewer than this many users have it (default 3)
        verbose: Print statistics

    Returns:
        Filtered records list
    """
    # Group by problem
    problems = defaultdict(list)
    for rec in records:
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            if len(pid) >= 2:
                prob_key = f'problem_{pid[1]}'
                problems[prob_key].append(rec)

    # Compute per-problem removals
    per_problem_removals = {}
    removed_too_common = 0
    removed_too_rare = 0

    for prob_id, recs in problems.items():
        n_users = len(recs)
        criteria_counts = Counter()
        for rec in recs:
            for pref_name in rec.get('persona_preferences', {}).keys():
                criteria_counts[pref_name] += 1

        too_common = set()
        too_rare = set()

        for c, count in criteria_counts.items():
            if count / n_users >= per_problem_max:
                too_common.add(c)
            elif count < min_users_per_problem:
                too_rare.add(c)

        per_problem_removals[prob_id] = too_common | too_rare
        removed_too_common += len(too_common)
        removed_too_rare += len(too_rare)

    if verbose:
        total_removals = sum(len(s) for s in per_problem_removals.values())
        unique_removed = set().union(*per_problem_removals.values()) if per_problem_removals else set()
        print(f"\nPer-problem filtering:")
        print(f"  Too common (>={per_problem_max*100:.0f}%): {removed_too_common} removals")
        print(f"  Too rare (<{min_users_per_problem} users): {removed_too_rare} removals")
        print(f"  Total removals: {total_removals}")
        print(f"  Unique criteria affected: {len(unique_removed)}")
        print(f"  Avg removed per problem: {total_removals/len(problems):.1f}")

    # Apply filtering
    cleaned = []
    for rec in records:
        rec = json.loads(json.dumps(rec))
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            prob_key = f'problem_{pid[1]}' if len(pid) >= 2 else None
            removal_set = per_problem_removals.get(prob_key, set())

            if 'persona_preferences' in rec:
                rec['persona_preferences'] = {
                    k: v for k, v in rec['persona_preferences'].items()
                    if k not in removal_set
                }
            if 'evaluation_rubric' in rec:
                rec['evaluation_rubric'] = _filter_rubric(
                    rec['evaluation_rubric'], removal_set
                )
            if 'evaluation_criteria' in rec:
                criteria = rec['evaluation_criteria']
                if isinstance(criteria, list):
                    rec['evaluation_criteria'] = [
                        item for item in criteria
                        if item.get('preference') not in removal_set
                    ]
                elif isinstance(criteria, dict):
                    rec['evaluation_criteria'] = {
                        k: v for k, v in criteria.items()
                        if k not in removal_set
                    }
        cleaned.append(rec)

    # Compute stats
    personalized = [r for r in cleaned if r.get('type') == 'personalized_problem']
    avg_criteria = sum(len(r.get('persona_preferences', {})) for r in personalized) / len(personalized) if personalized else 0

    if verbose:
        print(f"  Avg criteria per user after filtering: {avg_criteria:.2f}")

    return cleaned
