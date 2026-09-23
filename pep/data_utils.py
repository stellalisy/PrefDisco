"""
Data preprocessing and loading utilities.

This module handles:
1. Loading raw JSONL data
2. Comprehensive criteria filtering:
   - Cross-problem too common (global attributes)
   - Cross-problem too rare (rare attributes)
   - Within-problem too common (local universal attributes)
   - Explicit exclusion list
3. Preparing per-problem data structures for evaluation
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class FilteringStats:
    """Statistics from filtering process."""
    too_common_cross: Set[str]
    too_rare_cross: Set[str]
    too_common_within: Set[str]
    explicit_excluded: Set[str]
    all_removed: Set[str]
    avg_criteria_before: float
    avg_criteria_after: float

    def __str__(self):
        return f"""Filtering Statistics:
  Too common (cross-problem): {len(self.too_common_cross)}
  Too rare (cross-problem): {len(self.too_rare_cross)}
  Too common (within-problem): {len(self.too_common_within)}
  Explicitly excluded: {len(self.explicit_excluded)}
  Total removed: {len(self.all_removed)}
  Avg criteria per user: {self.avg_criteria_before:.1f} -> {self.avg_criteria_after:.1f}"""


def load_records(data_path: str) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {data_path}") from exc
    return records


def analyze_criteria_distribution(records: List[Dict]) -> Dict:
    """
    Analyze criteria distribution across problems and users.

    Returns dict with:
    - criteria_problem_count: {criterion: num_problems_it_appears_in}
    - criteria_user_count: {criterion: total_users_with_it}
    - problem_criteria_freq: {problem: {criterion: fraction_of_users}}
    - n_problems: total number of problems
    - all_criteria: set of all criteria names
    """
    problems = defaultdict(list)
    for rec in records:
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            if len(pid) >= 2:
                problems[f"problem_{pid[1]}"].append(rec)

    n_problems = len(problems)
    criteria_problem_count = Counter()
    criteria_user_count = Counter()
    problem_criteria_freq = {}
    all_criteria = set()

    for prob_id, recs in problems.items():
        n_users = len(recs)
        criteria_counts = Counter()

        for rec in recs:
            for pref_name in rec.get('persona_preferences', {}).keys():
                criteria_counts[pref_name] += 1
                criteria_user_count[pref_name] += 1
                all_criteria.add(pref_name)

        problem_criteria_freq[prob_id] = {
            c: count / n_users for c, count in criteria_counts.items()
        }

        # Count which criteria appear in this problem (with reasonable frequency)
        # Using T4 definition: 10-30% frequency within problem
        for c, count in criteria_counts.items():
            freq = count / n_users
            if 0.10 <= freq < 0.30:
                criteria_problem_count[c] += 1

    return {
        'criteria_problem_count': criteria_problem_count,
        'criteria_user_count': criteria_user_count,
        'problem_criteria_freq': problem_criteria_freq,
        'n_problems': n_problems,
        'all_criteria': all_criteria,
        'problems': problems
    }


def filter_criteria(
    records: List[Dict],
    cross_problem_max: float = 0.15,
    cross_problem_min: float = 0.03,
    within_problem_max: float = 0.70,
    per_problem_max: Optional[float] = None,
    explicit_exclude: Optional[Set[str]] = None,
    verbose: bool = True
) -> Tuple[List[Dict], FilteringStats]:
    """
    Comprehensive criteria filtering with automatic filters plus explicit exclusion.

    Filters:
    1. Cross-problem too common: Remove criteria appearing in >= cross_problem_max of problems
       (These are global attributes that don't differentiate users)
    2. Cross-problem too rare: Remove criteria appearing in <= cross_problem_min of problems
       (These are too rare to learn meaningful patterns)
    3. Within-problem too common (global removal): Remove criteria that >= within_problem_max
       of users have in ANY problem (These are locally universal and uninformative)
    4. Per-problem filtering (local removal): For each problem, remove criteria that >=
       per_problem_max of users have IN THAT PROBLEM ONLY (doesn't affect other problems)
    5. Explicit exclusion: Remove any criteria in explicit_exclude set

    Args:
        records: List of record dictionaries
        cross_problem_max: Remove if in MORE than this fraction of problems (default 0.15 = 15%)
        cross_problem_min: Remove if in FEWER than this fraction of problems (default 0.03 = 3%)
        within_problem_max: Remove if MORE than this fraction of users have it in ANY problem (default 0.70 = 70%)
        per_problem_max: If set, remove criteria from each problem where >= this fraction have it (e.g., 0.10 = 10%)
        explicit_exclude: Set of criteria names to always exclude
        verbose: Print filtering statistics

    Returns:
        Tuple of (filtered_records, FilteringStats)
    """
    if explicit_exclude is None:
        explicit_exclude = set()

    # Analyze distribution
    analysis = analyze_criteria_distribution(records)
    n_problems = analysis['n_problems']
    criteria_problem_count = analysis['criteria_problem_count']
    problem_criteria_freq = analysis['problem_criteria_freq']
    problems = analysis['problems']

    # Compute avg criteria before
    personalized_before = [r for r in records if r.get('type') == 'personalized_problem']
    avg_before = np.mean([len(r.get('persona_preferences', {})) for r in personalized_before])

    # === FILTER 1: Cross-problem TOO COMMON ===
    too_common_cross = set(
        c for c, count in criteria_problem_count.items()
        if count / n_problems >= cross_problem_max
    )

    # === FILTER 2: Cross-problem TOO RARE ===
    too_rare_cross = set(
        c for c, count in criteria_problem_count.items()
        if count / n_problems <= cross_problem_min
    )

    # === FILTER 3: Within-problem TOO COMMON (global removal) ===
    too_common_within = set()
    for prob_id, recs in problems.items():
        n_users = len(recs)
        criteria_counts = Counter()
        for rec in recs:
            for pref_name in rec.get('persona_preferences', {}).keys():
                criteria_counts[pref_name] += 1

        for c, count in criteria_counts.items():
            if count / n_users >= within_problem_max:
                too_common_within.add(c)

    # === FILTER 4: Per-problem filtering (compute per-problem removals) ===
    per_problem_removals = {}  # {problem_id: set of criteria to remove}
    if per_problem_max is not None:
        for prob_id, recs in problems.items():
            n_users = len(recs)
            criteria_counts = Counter()
            for rec in recs:
                for pref_name in rec.get('persona_preferences', {}).keys():
                    criteria_counts[pref_name] += 1

            per_problem_removals[prob_id] = set(
                c for c, count in criteria_counts.items()
                if count / n_users >= per_problem_max
            )

    # Combined GLOBAL removal (applies to all problems)
    all_removed = too_common_cross | too_rare_cross | too_common_within | explicit_exclude

    if verbose:
        print(f"\nFiltering criteria:")
        print(f"  1. Cross-problem too common (>={cross_problem_max*100:.0f}%): {len(too_common_cross)}")
        if too_common_cross:
            print(f"     Examples: {list(too_common_cross)[:3]}")
        print(f"  2. Cross-problem too rare (<={cross_problem_min*100:.0f}%): {len(too_rare_cross)}")
        if too_rare_cross:
            print(f"     Examples: {list(too_rare_cross)[:3]}")
        print(f"  3. Within-problem too common (>={within_problem_max*100:.0f}%): {len(too_common_within)}")
        if too_common_within:
            print(f"     Examples: {list(too_common_within)[:3]}")
        print(f"  4. Explicitly excluded: {len(explicit_exclude)}")
        print(f"  GLOBAL REMOVAL TOTAL: {len(all_removed)}")

        if per_problem_max is not None:
            total_per_problem = sum(len(s) for s in per_problem_removals.values())
            unique_per_problem = set().union(*per_problem_removals.values()) if per_problem_removals else set()
            print(f"  5. Per-problem filtering (>={per_problem_max*100:.0f}% within each problem):")
            print(f"     Total removals across problems: {total_per_problem}")
            print(f"     Unique criteria affected: {len(unique_per_problem)}")
            print(f"     Avg removed per problem: {total_per_problem/len(per_problem_removals):.1f}" if per_problem_removals else "")

    # Clean records
    cleaned = []
    for rec in records:
        rec = json.loads(json.dumps(rec))  # Deep copy

        if rec.get('type') == 'personalized_problem':
            # Determine problem ID for per-problem filtering
            pid = rec.get('problem_id', '').split('_')
            prob_key = f"problem_{pid[1]}" if len(pid) >= 2 else None

            # Get combined removal set for this record
            removal_set = all_removed.copy()
            if per_problem_max is not None and prob_key and prob_key in per_problem_removals:
                removal_set = removal_set | per_problem_removals[prob_key]

            # Clean persona_preferences
            if 'persona_preferences' in rec:
                cleaned_prefs = {}
                for name, details in rec['persona_preferences'].items():
                    if name not in removal_set:
                        if isinstance(details, dict):
                            details['weight'] = 1.0
                        cleaned_prefs[name] = details
                rec['persona_preferences'] = cleaned_prefs

            # Clean evaluation_rubric
            if 'evaluation_rubric' in rec:
                cleaned_rubric = {}
                for name, details in rec['evaluation_rubric'].items():
                    if name not in removal_set:
                        if isinstance(details, dict):
                            details['weight'] = 1.0
                        cleaned_rubric[name] = details
                rec['evaluation_rubric'] = cleaned_rubric

            # Clean evaluation_criteria (alternate field name)
            if 'evaluation_criteria' in rec:
                cleaned_criteria = {}
                for name, details in rec['evaluation_criteria'].items():
                    if name not in removal_set:
                        if isinstance(details, dict):
                            details['weight'] = 1.0
                        cleaned_criteria[name] = details
                rec['evaluation_criteria'] = cleaned_criteria

        cleaned.append(rec)

    # Compute avg criteria after
    personalized_after = [r for r in cleaned if r.get('type') == 'personalized_problem']
    avg_after = np.mean([len(r.get('persona_preferences', {})) for r in personalized_after])

    if verbose:
        print(f"  Avg criteria per user: {avg_before:.1f} -> {avg_after:.1f}")

    stats = FilteringStats(
        too_common_cross=too_common_cross,
        too_rare_cross=too_rare_cross,
        too_common_within=too_common_within,
        explicit_excluded=explicit_exclude,
        all_removed=all_removed,
        avg_criteria_before=avg_before,
        avg_criteria_after=avg_after
    )

    return cleaned, stats


def prune_criteria(records: List[Dict], excluded_criteria: Set[str]) -> List[Dict]:
    """
    Simple pruning: remove only explicitly excluded criteria.
    For comprehensive filtering, use filter_criteria() instead.

    Args:
        records: List of record dictionaries
        excluded_criteria: Set of criteria names to remove

    Returns:
        List of records with excluded criteria removed
    """
    pruned_records = []

    for record in records:
        new_record = json.loads(json.dumps(record))  # Deep copy

        # Prune persona_preferences
        if 'persona_preferences' in new_record:
            new_record['persona_preferences'] = {
                k: v for k, v in new_record['persona_preferences'].items()
                if k not in excluded_criteria
            }

        # Prune evaluation_criteria
        if 'evaluation_criteria' in new_record:
            new_record['evaluation_criteria'] = {
                k: v for k, v in new_record['evaluation_criteria'].items()
                if k not in excluded_criteria
            }

        # Prune evaluation_rubric
        if 'evaluation_rubric' in new_record:
            new_record['evaluation_rubric'] = {
                k: v for k, v in new_record['evaluation_rubric'].items()
                if k not in excluded_criteria
            }

        pruned_records.append(new_record)

    return pruned_records


def get_all_problems(records: List[Dict]) -> List[str]:
    """Extract all unique problem IDs from records."""
    problems = set()
    for r in records:
        if r.get('type') == 'personalized_problem':
            parts = r.get('problem_id', '').split('_')
            if len(parts) >= 2:
                problems.add(parts[1])
    return sorted(problems)


def prepare_problem_data(
    records: List[Dict],
    problem_num: str,
    min_users: int = 4
) -> Optional[Dict]:
    """
    Prepare data structures for a single problem.

    Args:
        records: List of all records
        problem_num: Problem number (string)
        min_users: Minimum users a criterion must appear in

    Returns:
        Dictionary with:
        - criteria: List of criterion names
        - n_criteria: Number of criteria
        - criterion_to_idx: Mapping from name to index
        - n_users: Number of users
        - R_has: Binary matrix (n_users x n_criteria) indicating if user has criterion
        - R_val: Value matrix (n_users x n_criteria) with preference values
        - user_prefs: List of dicts mapping criterion name to value for each user
        - valid_users: List of user indices with at least one preference
    """
    # Filter to this problem
    prob_recs = [r for r in records
                 if r.get('type') == 'personalized_problem'
                 and f"problem_{problem_num}_" in r.get('problem_id', '')]

    if not prob_recs:
        return None

    n_users = len(prob_recs)

    # Count criterion frequency
    criteria_counter = Counter()
    for rec in prob_recs:
        for c_name in rec.get('persona_preferences', {}).keys():
            criteria_counter[c_name] += 1

    # Filter to criteria appearing in enough users
    common_criteria = sorted([c for c, cnt in criteria_counter.items() if cnt >= min_users])
    if not common_criteria:
        return None

    criterion_to_idx = {c: i for i, c in enumerate(common_criteria)}
    n_criteria = len(common_criteria)

    # Build matrices
    R_has = np.zeros((n_users, n_criteria))
    R_val = np.zeros((n_users, n_criteria))
    user_prefs = []

    for u, rec in enumerate(prob_recs):
        prefs = {}
        for c_name, details in rec.get('persona_preferences', {}).items():
            if c_name in criterion_to_idx:
                idx = criterion_to_idx[c_name]
                val = details.get('value', 3) if isinstance(details, dict) else details
                if isinstance(val, (int, float)):
                    R_has[u, idx] = 1
                    R_val[u, idx] = val
                    prefs[c_name] = val
        user_prefs.append(prefs)

    valid_users = [i for i, p in enumerate(user_prefs) if len(p) > 0]

    return {
        'criteria': common_criteria,
        'n_criteria': n_criteria,
        'criterion_to_idx': criterion_to_idx,
        'n_users': n_users,
        'R_has': R_has,
        'R_val': R_val,
        'user_prefs': user_prefs,
        'valid_users': valid_users,
        'problem_num': problem_num
    }


def compute_global_statistics(
    records: List[Dict],
    train_problems: List[str],
    min_users: int = 4
) -> Dict:
    """
    Compute global statistics from training problems.

    These statistics are used by likelihood models:
    - priors: P(has criterion), E[value | has criterion]
    - similarity: Criterion-criterion cosine similarity
    - pairwise_cond: P(c2 | c1), P(c2 | not c1) for Naive Bayes
    - value_corr: Value correlations between criteria

    Args:
        records: All records
        train_problems: List of problem numbers to use for training
        min_users: Minimum users per criterion

    Returns:
        Dictionary with global statistics
    """
    # Accumulators
    criterion_stats = {}  # {name: {has_sum, val_sum, val_count, total}}
    cooccur = Counter()   # (c1, c2) -> count
    single = Counter()    # c -> count
    pair_stats = {}       # (c1, c2) -> {both, c1_only, c2_only, neither, total}
    value_cooccur = {}    # (c1, c2) -> [(v1, v2), ...]

    for prob_num in train_problems:
        data = prepare_problem_data(records, prob_num, min_users=min_users)
        if data is None:
            continue

        for u in data['valid_users']:
            # Get user's criteria and values
            user_criteria = []
            user_values = {}
            for c in range(data['n_criteria']):
                c_name = data['criteria'][c]
                if data['R_has'][u, c] == 1:
                    user_criteria.append(c_name)
                    user_values[c_name] = data['R_val'][u, c]

            # Update basic stats
            for c_name in data['criteria']:
                if c_name not in criterion_stats:
                    criterion_stats[c_name] = {'has_sum': 0, 'val_sum': 0, 'val_count': 0, 'total': 0}
                criterion_stats[c_name]['total'] += 1
                if c_name in user_criteria:
                    criterion_stats[c_name]['has_sum'] += 1
                    criterion_stats[c_name]['val_sum'] += user_values[c_name]
                    criterion_stats[c_name]['val_count'] += 1

            # Update single counts
            for c in user_criteria:
                single[c] += 1

            # Update co-occurrence
            for i, c1 in enumerate(user_criteria):
                for c2 in user_criteria[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    cooccur[key] += 1

                    # Value co-occurrence
                    if key not in value_cooccur:
                        value_cooccur[key] = []
                    value_cooccur[key].append((user_values[c1], user_values[c2]))

            # Update pairwise stats
            all_criteria_names = data['criteria']
            for i, c1 in enumerate(all_criteria_names):
                for c2 in all_criteria_names[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    if key not in pair_stats:
                        pair_stats[key] = {'both': 0, 'c1_only': 0, 'c2_only': 0, 'neither': 0, 'total': 0}

                    has_c1 = c1 in user_criteria
                    has_c2 = c2 in user_criteria

                    pair_stats[key]['total'] += 1
                    if has_c1 and has_c2:
                        pair_stats[key]['both'] += 1
                    elif has_c1:
                        pair_stats[key]['c1_only'] += 1
                    elif has_c2:
                        pair_stats[key]['c2_only'] += 1
                    else:
                        pair_stats[key]['neither'] += 1

    # Compute derived statistics
    global_priors = {}
    for c_name, stats in criterion_stats.items():
        if stats['total'] > 0:
            global_priors[c_name] = {
                'prior_has': stats['has_sum'] / stats['total'],
                'prior_val': stats['val_sum'] / stats['val_count'] if stats['val_count'] > 0 else 3.0,
                'count': stats['total']
            }

    # Item similarity (cosine)
    global_similarity = {}
    for (c1, c2), count in cooccur.items():
        if count >= 3:
            sim = count / np.sqrt(max(single[c1], 1) * max(single[c2], 1))
            global_similarity[(c1, c2)] = sim
            global_similarity[(c2, c1)] = sim

    # Pairwise conditionals
    pairwise_cond = {}
    for (c1, c2), stats in pair_stats.items():
        if stats['total'] >= 5:
            denom1 = stats['both'] + stats['c1_only']
            p_c2_given_c1 = stats['both'] / denom1 if denom1 > 0 else 0.5

            denom2 = stats['c2_only'] + stats['neither']
            p_c2_given_not_c1 = stats['c2_only'] / denom2 if denom2 > 0 else 0.5

            denom3 = stats['both'] + stats['c2_only']
            p_c1_given_c2 = stats['both'] / denom3 if denom3 > 0 else 0.5

            denom4 = stats['c1_only'] + stats['neither']
            p_c1_given_not_c2 = stats['c1_only'] / denom4 if denom4 > 0 else 0.5

            pairwise_cond[(c1, c2)] = {
                'p_c2_given_c1': p_c2_given_c1,
                'p_c2_given_not_c1': p_c2_given_not_c1,
                'p_c1_given_c2': p_c1_given_c2,
                'p_c1_given_not_c2': p_c1_given_not_c2
            }
            pairwise_cond[(c2, c1)] = {
                'p_c2_given_c1': p_c1_given_c2,
                'p_c2_given_not_c1': p_c1_given_not_c2,
                'p_c1_given_c2': p_c2_given_c1,
                'p_c1_given_not_c2': p_c2_given_not_c1
            }

    # Value correlations
    value_corr = {}
    for (c1, c2), pairs in value_cooccur.items():
        if len(pairs) >= 3:
            v1s = [p[0] for p in pairs]
            v2s = [p[1] for p in pairs]
            if np.std(v1s) > 0 and np.std(v2s) > 0:
                corr = np.corrcoef(v1s, v2s)[0, 1]
                if not np.isnan(corr):
                    value_corr[(c1, c2)] = corr
                    value_corr[(c2, c1)] = corr

    return {
        'priors': global_priors,
        'similarity': global_similarity,
        'pairwise_cond': pairwise_cond,
        'value_corr': value_corr,
        'single_counts': dict(single)
    }


def save_pruned_data(records: List[Dict], output_path: str):
    """Save pruned records to JSONL file."""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


# CLI for preprocessing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess and prune criteria from data')
    parser.add_argument('input_path', help='Path to input JSONL file')
    parser.add_argument('output_path', help='Path to output JSONL file')
    parser.add_argument('--exclude', nargs='+', default=[
        'Laboratory Value Interpretation',
        'Visual Problem Mapping'
    ], help='Criteria to exclude')
    parser.add_argument('--list-criteria', action='store_true',
                       help='List all criteria and their frequencies')

    args = parser.parse_args()

    print(f"Loading data from {args.input_path}...")
    records = load_records(args.input_path)
    print(f"Loaded {len(records)} records")

    if args.list_criteria:
        criteria_counter = Counter()
        for r in records:
            if r.get('type') == 'personalized_problem':
                for c in r.get('persona_preferences', {}).keys():
                    criteria_counter[c] += 1

        print("\nAll criteria (sorted by frequency):")
        print("-" * 60)
        for c, count in criteria_counter.most_common():
            print(f"  {count:4d}  {c}")
        print(f"\nTotal unique criteria: {len(criteria_counter)}")

    excluded = set(args.exclude)
    print(f"\nExcluding criteria: {excluded}")

    pruned = prune_criteria(records, excluded)

    print(f"Saving to {args.output_path}...")
    save_pruned_data(pruned, args.output_path)
    print("Done!")


def filter_criteria_with_min_users(
    records: List[Dict],
    per_problem_max: float = 0.10,
    min_users_per_problem: int = 3,
    cross_problem_min: float = 0.0,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Filter criteria with per-problem max frequency AND minimum user count.

    A criterion is kept in a problem only if:
    1. It appears in < per_problem_max fraction of users (not too common), AND
    2. It appears in >= min_users_per_problem users (not too rare)

    Optionally also removes globally rare criteria (cross_problem_min).

    Args:
        records: List of record dictionaries
        per_problem_max: Remove if >= this fraction within a problem (default 0.10 = 10%)
        min_users_per_problem: Keep only if >= this many users have it (default 3)
        cross_problem_min: Remove if in <= this fraction of problems globally (default 0.0 = disabled)
        verbose: Print filtering statistics

    Returns:
        Tuple of (filtered_records, stats_dict)
    """
    # Group by problem
    problems = defaultdict(list)
    for rec in records:
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            if len(pid) >= 2:
                prob_key = f"problem_{pid[1]}"
                problems[prob_key].append(rec)

    n_problems = len(problems)

    # Compute global criterion frequency (for cross_problem_min filter)
    global_criterion_problems = Counter()
    for prob_id, recs in problems.items():
        criteria_in_problem = set()
        for rec in recs:
            for pref_name in rec.get('persona_preferences', {}).keys():
                criteria_in_problem.add(pref_name)
        for c in criteria_in_problem:
            global_criterion_problems[c] += 1

    globally_rare = set()
    if cross_problem_min > 0:
        globally_rare = set(
            c for c, count in global_criterion_problems.items()
            if count / n_problems <= cross_problem_min
        )

    # Compute per-problem removals
    per_problem_removals = {}
    total_too_common = 0
    total_too_rare = 0

    for prob_id, recs in problems.items():
        n_users = len(recs)
        criteria_counts = Counter()
        for rec in recs:
            for pref_name in rec.get('persona_preferences', {}).keys():
                criteria_counts[pref_name] += 1

        too_common = set(
            c for c, count in criteria_counts.items()
            if count / n_users >= per_problem_max
        )
        too_rare = set(
            c for c, count in criteria_counts.items()
            if count < min_users_per_problem
        )

        total_too_common += len(too_common)
        total_too_rare += len(too_rare)

        per_problem_removals[prob_id] = too_common | too_rare | globally_rare

    # Compute stats before filtering
    personalized_before = [r for r in records if r.get('type') == 'personalized_problem']
    avg_before = np.mean([len(r.get('persona_preferences', {})) for r in personalized_before])

    # Apply filtering
    cleaned = []
    for rec in records:
        rec = json.loads(json.dumps(rec))  # Deep copy
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            prob_key = f"problem_{pid[1]}" if len(pid) >= 2 else None
            removal_set = per_problem_removals.get(prob_key, set())

            if 'persona_preferences' in rec:
                rec['persona_preferences'] = {
                    k: v for k, v in rec['persona_preferences'].items()
                    if k not in removal_set
                }
            if 'evaluation_rubric' in rec:
                rec['evaluation_rubric'] = {
                    k: v for k, v in rec['evaluation_rubric'].items()
                    if k not in removal_set
                }
            if 'evaluation_criteria' in rec:
                rec['evaluation_criteria'] = {
                    k: v for k, v in rec['evaluation_criteria'].items()
                    if k not in removal_set
                }
        cleaned.append(rec)

    # Compute stats after filtering
    personalized_after = [r for r in cleaned if r.get('type') == 'personalized_problem']
    avg_after = np.mean([len(r.get('persona_preferences', {})) for r in personalized_after])

    total_removals = sum(len(s) for s in per_problem_removals.values())
    unique_removed = set().union(*per_problem_removals.values()) if per_problem_removals else set()

    stats = {
        'n_problems': n_problems,
        'total_removals': total_removals,
        'unique_criteria_removed': len(unique_removed),
        'avg_removed_per_problem': total_removals / n_problems if n_problems > 0 else 0,
        'too_common_removals': total_too_common,
        'too_rare_removals': total_too_rare,
        'globally_rare_removed': len(globally_rare),
        'avg_criteria_before': avg_before,
        'avg_criteria_after': avg_after,
    }

    if verbose:
        print(f"\nFiltering with per_problem_max={per_problem_max}, min_users={min_users_per_problem}:")
        print(f"  Too common (>={per_problem_max*100:.0f}% within problem): {total_too_common} removals")
        print(f"  Too rare (<{min_users_per_problem} users within problem): {total_too_rare} removals")
        if cross_problem_min > 0:
            print(f"  Globally rare (<={cross_problem_min*100:.0f}% of problems): {len(globally_rare)} criteria")
        print(f"  Total removals across problems: {total_removals}")
        print(f"  Unique criteria affected: {len(unique_removed)}")
        print(f"  Avg removed per problem: {stats['avg_removed_per_problem']:.1f}")
        print(f"  Avg criteria per user: {avg_before:.1f} -> {avg_after:.1f}")

    return cleaned, stats
