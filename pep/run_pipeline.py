"""
Preference Elicitation Experiment Pipeline (v2)

Usage:
    python -m pep.run_pipeline <data_path> --per-problem-max 0.10 --min-users 4 --output <output_dir>

This script:
1. Loads the raw dataset
2. Plots PRE-filtering dataset statistics
3. Filters based on provided config parameters
4. Plots POST-filtering dataset statistics
5. Runs all model × strategy combinations
6. Generates results plots (including heatmaps for all metrics)
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

from .data_utils import (
    load_records, get_all_problems,
    prepare_problem_data, compute_global_statistics
)
from .filter_utils import filter_with_min_users
from .models import ALL_MODELS
from .strategies import STRATEGY_REGISTRY, RANDOM_STRATEGIES, FIXED_STRATEGIES, ADAPTIVE_STRATEGIES

plt.style.use('seaborn-v0_8-whitegrid')

# Filter out Uniform model
MODELS = [M for M in ALL_MODELS if M.name != 'Uniform']

# Default K values for Top-K metrics
TOP_K_VALUES = [3, 4, 5, 6]


@dataclass
class ExtendedMetrics:
    """Simplified metrics focused on Top-K performance."""
    # Top-K metrics (the main ones)
    topk_precision: Dict[int, float]  # Precision@K
    topk_recall: Dict[int, float]     # Recall@K
    topk_f1: Dict[int, float]         # F1@K
    mae_tp: Dict[int, float]          # MAE|TP@K
    rmse_tp: Dict[int, float]         # RMSE|TP@K
    n_tp: Dict[int, int]              # Number of true positives at K

    def to_dict(self) -> Dict:
        d = {}
        for k in self.topk_precision:
            d[f'precision_{k}'] = self.topk_precision[k]
            d[f'recall_{k}'] = self.topk_recall[k]
            d[f'f1_{k}'] = self.topk_f1[k]
            d[f'mae_tp_{k}'] = self.mae_tp[k]
            d[f'rmse_tp_{k}'] = self.rmse_tp[k]
            d[f'n_tp_{k}'] = self.n_tp[k]
        return d


def compute_extended_metrics(
    model,
    pred_has: np.ndarray,
    pred_val: np.ndarray,
    user_prefs: Dict[str, float],
    observed_has: List[Tuple[int, float]],
    top_k_values: List[int] = TOP_K_VALUES
) -> ExtendedMetrics:
    """
    Compute Top-K focused metrics.
    """
    # Get user's criteria indices and values
    user_criteria = {}  # idx -> value
    for c_name, true_val in user_prefs.items():
        if c_name in model.criterion_to_idx:
            c_idx = model.criterion_to_idx[c_name]
            user_criteria[c_idx] = true_val

    user_criteria_idx = set(user_criteria.keys())
    n_user_criteria = len(user_criteria_idx)

    # Rank criteria by P(has) descending
    ranked_indices = np.argsort(pred_has)[::-1]

    topk_precision = {}
    topk_recall = {}
    topk_f1 = {}
    mae_tp = {}
    rmse_tp = {}
    n_tp = {}

    for k in top_k_values:
        top_k_set = set(ranked_indices[:k])
        tp_set = top_k_set & user_criteria_idx
        n_tp[k] = len(tp_set)

        # Precision@K: what fraction of top-K are correct?
        topk_precision[k] = n_tp[k] / k if k > 0 else 0.0

        # Recall@K: what fraction of user's criteria are in top-K?
        topk_recall[k] = n_tp[k] / n_user_criteria if n_user_criteria > 0 else 0.0

        # F1@K: harmonic mean of Precision@K and Recall@K
        p, r = topk_precision[k], topk_recall[k]
        topk_f1[k] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # MAE|TP@K and RMSE|TP@K: value errors on true positives
        if tp_set:
            tp_errors = [abs(pred_val[idx] - user_criteria[idx]) for idx in tp_set]
            mae_tp[k] = np.mean(tp_errors)
            rmse_tp[k] = np.sqrt(np.mean([e**2 for e in tp_errors]))
        else:
            mae_tp[k] = np.nan
            rmse_tp[k] = np.nan

    return ExtendedMetrics(
        topk_precision=topk_precision,
        topk_recall=topk_recall,
        topk_f1=topk_f1,
        mae_tp=mae_tp,
        rmse_tp=rmse_tp,
        n_tp=n_tp
    )


def run_session_extended(
    model,
    user_prefs: Dict[str, float],
    select_fn,
    budget: int = 5,
    top_k_values: List[int] = TOP_K_VALUES
) -> List[ExtendedMetrics]:
    """Run a session and compute extended metrics at each step."""

    criteria = model.criteria
    observed_has = []
    observed_not_has = []
    asked = set()
    evolution = []

    # Initial metrics (prior)
    pred_has, pred_val = model.predict([], [])
    evolution.append(compute_extended_metrics(
        model, pred_has, pred_val, user_prefs, observed_has, top_k_values
    ))

    # Ask questions
    for q in range(budget):
        c = select_fn(model, asked, observed_has, observed_not_has)
        asked.add(c)
        c_name = criteria[c]

        if c_name in user_prefs:
            observed_has.append((c, user_prefs[c_name]))
        else:
            observed_not_has.append(c)

        pred_has, pred_val = model.predict(observed_has, observed_not_has)

        for (c_idx, val) in observed_has:
            pred_val[c_idx] = val

        evolution.append(compute_extended_metrics(
            model, pred_has, pred_val, user_prefs, observed_has, top_k_values
        ))

    return evolution


def aggregate_extended_metrics(evolutions: List[List[ExtendedMetrics]], budget: int,
                                top_k_values: List[int] = TOP_K_VALUES) -> Dict:
    """Aggregate Top-K metrics across sessions."""

    if not evolutions:
        return {}

    result = {
        'n_sessions': len(evolutions),
        'by_step': {}
    }

    for step in range(budget + 1):
        step_metrics = [e[step] for e in evolutions]
        step_result = {}

        for k in top_k_values:
            # Precision, Recall, F1 (always defined)
            step_result[f'precision_{k}_mean'] = np.mean([m.topk_precision[k] for m in step_metrics])
            step_result[f'recall_{k}_mean'] = np.mean([m.topk_recall[k] for m in step_metrics])
            step_result[f'f1_{k}_mean'] = np.mean([m.topk_f1[k] for m in step_metrics])

            # MAE|TP and RMSE|TP (filter out NaN when no TPs)
            mae_values = [m.mae_tp[k] for m in step_metrics if m.n_tp[k] > 0 and not np.isnan(m.mae_tp[k])]
            rmse_values = [m.rmse_tp[k] for m in step_metrics if m.n_tp[k] > 0 and not np.isnan(m.rmse_tp[k])]

            step_result[f'mae_tp_{k}_mean'] = np.mean(mae_values) if mae_values else 0.0
            step_result[f'rmse_tp_{k}_mean'] = np.mean(rmse_values) if rmse_values else 0.0
            step_result[f'n_tp_{k}_mean'] = np.mean([m.n_tp[k] for m in step_metrics])

        result['by_step'][step] = step_result

    # Final step summary
    final = result['by_step'][budget]
    result['summary'] = {}

    for k in top_k_values:
        result['summary'][f'precision_{k}'] = final[f'precision_{k}_mean'] * 100  # As percentage
        result['summary'][f'recall_{k}'] = final[f'recall_{k}_mean'] * 100
        result['summary'][f'f1_{k}'] = final[f'f1_{k}_mean'] * 100
        result['summary'][f'mae_tp_{k}'] = final[f'mae_tp_{k}_mean']
        result['summary'][f'rmse_tp_{k}'] = final[f'rmse_tp_{k}_mean']

    return result


def compute_dataset_stats(records):
    """Compute statistics about the dataset."""

    problems = defaultdict(list)
    for rec in records:
        if rec.get('type') == 'personalized_problem':
            pid = rec.get('problem_id', '').split('_')
            if len(pid) >= 2:
                problems[f'problem_{pid[1]}'].append(rec)

    criteria_per_user = []
    criteria_freq_within_problem = []
    unique_criteria_per_problem = []
    all_criteria_counts = Counter()
    users_with_zero_criteria = 0

    for prob_id, recs in problems.items():
        n_users = len(recs)
        prob_criteria = Counter()

        for rec in recs:
            prefs = rec.get('persona_preferences', {})
            n_prefs = len(prefs)
            criteria_per_user.append(n_prefs)
            if n_prefs == 0:
                users_with_zero_criteria += 1
            for c in prefs.keys():
                prob_criteria[c] += 1
                all_criteria_counts[c] += 1

        unique_criteria_per_problem.append(len(prob_criteria))

        for c, count in prob_criteria.items():
            criteria_freq_within_problem.append(count / n_users)

    return {
        'n_problems': len(problems),
        'n_users': len(criteria_per_user),
        'criteria_per_user': criteria_per_user,
        'criteria_freq_within_problem': criteria_freq_within_problem,
        'unique_criteria_per_problem': unique_criteria_per_problem,
        'all_criteria_counts': all_criteria_counts,
        'users_with_zero_criteria': users_with_zero_criteria
    }


def plot_dataset_statistics(stats, output_dir, title_prefix, per_problem_max=None, min_users=None):
    """Plot statistics about the dataset."""

    criteria_per_user = stats['criteria_per_user']
    criteria_freq_within_problem = stats['criteria_freq_within_problem']
    unique_criteria_per_problem = stats['unique_criteria_per_problem']
    all_criteria_counts = stats['all_criteria_counts']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    config_str = ""
    if per_problem_max is not None and min_users is not None:
        config_str = f"\n(per_problem_max={per_problem_max}, min_users={min_users})"

    fig.suptitle(f'{title_prefix} Dataset Statistics{config_str}',
                 fontsize=14, fontweight='bold')

    # 1. Distribution of criteria per user
    ax = axes[0, 0]
    max_criteria = max(criteria_per_user) if criteria_per_user else 1
    bins = range(0, max_criteria + 2)
    ax.hist(criteria_per_user, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(criteria_per_user), color='red', linestyle='--',
               label=f'Mean: {np.mean(criteria_per_user):.2f}')
    ax.set_xlabel('Number of Criteria per User')
    ax.set_ylabel('Frequency')
    ax.set_title('Criteria per User Distribution')
    ax.legend()

    # 2. Distribution of criteria frequency within problems
    ax = axes[0, 1]
    if criteria_freq_within_problem:
        ax.hist(criteria_freq_within_problem, bins=30, color='forestgreen',
                edgecolor='black', alpha=0.7)
        if per_problem_max is not None:
            ax.axvline(per_problem_max, color='red', linestyle='--',
                       label=f'Max threshold: {per_problem_max}')
            ax.legend()
    ax.set_xlabel('Fraction of Users with Criterion (within problem)')
    ax.set_ylabel('Frequency')
    ax.set_title('Within-Problem Criteria Frequency')

    # 3. Unique criteria per problem
    ax = axes[0, 2]
    if unique_criteria_per_problem:
        ax.hist(unique_criteria_per_problem, bins=20, color='coral',
                edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(unique_criteria_per_problem), color='red', linestyle='--',
                   label=f'Mean: {np.mean(unique_criteria_per_problem):.1f}')
        ax.legend()
    ax.set_xlabel('Unique Criteria per Problem')
    ax.set_ylabel('Frequency')
    ax.set_title('Unique Criteria per Problem')

    # 4. Top 15 most common criteria
    ax = axes[1, 0]
    if all_criteria_counts:
        top_criteria = all_criteria_counts.most_common(15)
        names = [c[0][:25] + '...' if len(c[0]) > 25 else c[0] for c in top_criteria]
        counts = [c[1] for c in top_criteria]
        y_pos = range(len(names))
        ax.barh(y_pos, counts, color='mediumpurple', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
    ax.set_xlabel('Total Occurrences')
    ax.set_title('Top 15 Most Common Criteria')

    # 5. CDF of criteria frequency
    ax = axes[1, 1]
    if criteria_freq_within_problem:
        sorted_freq = np.sort(criteria_freq_within_problem)
        cdf = np.arange(1, len(sorted_freq) + 1) / len(sorted_freq)
        ax.plot(sorted_freq, cdf, color='darkorange', linewidth=2)
        if per_problem_max is not None:
            ax.axvline(per_problem_max, color='red', linestyle='--', alpha=0.7,
                       label=f'Max threshold: {per_problem_max}')
            ax.legend()
    ax.set_xlabel('Within-Problem Frequency')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Criteria Frequency')
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics text
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = f"""
    DATASET SUMMARY
    ═══════════════════════════════

    Problems: {stats['n_problems']}
    Total Users: {stats['n_users']}
    Users with 0 criteria: {stats['users_with_zero_criteria']}

    Criteria per User:
      Mean: {np.mean(criteria_per_user):.2f}
      Median: {np.median(criteria_per_user):.1f}
      Std: {np.std(criteria_per_user):.2f}
      Min: {min(criteria_per_user)}
      Max: {max(criteria_per_user)}

    Unique Criteria per Problem:
      Mean: {np.mean(unique_criteria_per_problem):.1f}
      Min: {min(unique_criteria_per_problem) if unique_criteria_per_problem else 0}
      Max: {max(unique_criteria_per_problem) if unique_criteria_per_problem else 0}

    Total Unique Criteria: {len(all_criteria_counts)}
    """

    if per_problem_max is not None:
        stats_text += f"""
    Filter Settings:
      per_problem_max: {per_problem_max}
      min_users: {min_users}
    """

    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    filename = f'{output_dir}/{title_prefix.lower().replace(" ", "_")}_statistics.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

    return stats


def drop_zero_criteria_users(records):
    """Remove users with 0 criteria from records and sync evaluation fields."""
    cleaned = []
    dropped = 0
    for rec in records:
        if rec.get('type') == 'personalized_problem':
            prefs = rec.get('persona_preferences', {})
            if len(prefs) == 0:
                dropped += 1
                continue

            # Sync evaluation_rubric and evaluation_criteria with persona_preferences
            # Only keep criteria that are in persona_preferences
            rec = json.loads(json.dumps(rec))  # Deep copy
            pref_keys = set(prefs.keys())

            # Handle evaluation_rubric with nested evaluation_criteria list
            if 'evaluation_rubric' in rec:
                rubric = rec['evaluation_rubric']
                if isinstance(rubric, dict) and 'evaluation_criteria' in rubric:
                    # Filter the evaluation_criteria list to only include items whose 'preference' is in pref_keys
                    rubric['evaluation_criteria'] = [
                        item for item in rubric['evaluation_criteria']
                        if item.get('preference') in pref_keys
                    ]
                elif isinstance(rubric, dict):
                    # If it's a flat dict, filter by keys
                    rec['evaluation_rubric'] = {
                        k: v for k, v in rubric.items()
                        if k in pref_keys
                    }

            # Handle standalone evaluation_criteria (if it exists as a dict)
            if 'evaluation_criteria' in rec:
                ec = rec['evaluation_criteria']
                if isinstance(ec, dict):
                    rec['evaluation_criteria'] = {
                        k: v for k, v in ec.items()
                        if k in pref_keys
                    }
                elif isinstance(ec, list):
                    rec['evaluation_criteria'] = [
                        item for item in ec
                        if item.get('preference') in pref_keys
                    ]
        cleaned.append(rec)
    return cleaned, dropped


def run_experiment(records, budget=5, train_ratio=0.9,
                   min_users_per_criterion=4, random_seed=42):
    """Run the experiment with a single train-test split."""

    all_problems = get_all_problems(records)
    print(f"  Problems: {len(all_problems)}")

    all_results = {
        (M.name, s): []
        for M in MODELS
        for s in STRATEGY_REGISTRY
    }

    # Single train-test split
    np.random.seed(random_seed)
    shuffled = np.random.permutation(all_problems).tolist()
    n_train = int(len(shuffled) * train_ratio)
    train_problems = shuffled[:n_train]
    test_problems = shuffled[n_train:]

    print(f"  Split: {len(train_problems)} train, {len(test_problems)} test")

    global_stats = compute_global_statistics(
        records, train_problems,
        min_users=min_users_per_criterion
    )

    total_sessions = 0

    for prob_num in test_problems:
        data = prepare_problem_data(
            records, prob_num,
            min_users=min_users_per_criterion
        )
        if data is None or len(data['valid_users']) < 5:
            continue

        models = {M.name: M(data, global_stats) for M in MODELS}

        for idx in data['valid_users']:
            prefs = data['user_prefs'][idx]
            if not prefs:
                continue

            total_sessions += 1

            for model_name, model in models.items():
                for strat_name, select_fn in STRATEGY_REGISTRY.items():
                    np.random.seed(random_seed + idx)
                    evolution = run_session_extended(model, prefs, select_fn, budget)
                    all_results[(model_name, strat_name)].append(evolution)

    print(f"  Total sessions: {total_sessions}")

    aggregated = {}
    for key, evolutions in all_results.items():
        if evolutions:
            aggregated[key] = aggregate_extended_metrics(evolutions, budget)

    return aggregated


def plot_heatmaps_all_metrics(aggregated, output_dir, per_problem_max, min_users):
    """Plot heatmaps for all Top-K metrics."""

    models = sorted(set(k[0] for k in aggregated.keys()))
    strategies = list(STRATEGY_REGISTRY.keys())

    config_str = f"per_problem_max={per_problem_max}, min_users={min_users}"

    # Create heatmaps for each K
    for k in TOP_K_VALUES:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        metrics = [
            (f'precision_{k}', f'Precision@{k} (%)', 'RdYlGn', 0, 100),
            (f'recall_{k}', f'Recall@{k} (%)', 'RdYlGn', 0, 100),
            (f'f1_{k}', f'F1@{k} (%)', 'RdYlGn', 0, 100),
            (f'mae_tp_{k}', f'MAE|TP@{k} (↓)', 'RdYlGn_r', None, None),
            (f'rmse_tp_{k}', f'RMSE|TP@{k} (↓)', 'RdYlGn_r', None, None),
        ]

        for ax, (metric_key, title, cmap, vmin, vmax) in zip(axes, metrics):
            matrix = np.zeros((len(models), len(strategies)))

            for i, m in enumerate(models):
                for j, s in enumerate(strategies):
                    if (m, s) in aggregated and aggregated[(m, s)]:
                        val = aggregated[(m, s)]['summary'].get(metric_key, 0)
                        matrix[i, j] = val

            # Auto vmin/vmax for MAE/RMSE
            if vmin is None:
                vmin = 0
                vmax = max(0.5, matrix.max() * 1.1)

            im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(strategies)))
            ax.set_yticks(range(len(models)))
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(models, fontsize=10)

            for i in range(len(models)):
                for j in range(len(strategies)):
                    val = matrix[i, j]
                    if 'mae' in metric_key or 'rmse' in metric_key:
                        color = 'white' if val > (vmax * 0.6) else 'black'
                        txt = f'{val:.3f}'
                    else:
                        color = 'white' if val < 30 or val > 70 else 'black'
                        txt = f'{val:.1f}%'
                    ax.text(j, i, txt, ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')

            ax.set_title(title, fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide last subplot
        axes[-1].set_visible(False)

        plt.suptitle(f'Top-{k} Metrics Heatmaps ({config_str})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/heatmaps_top{k}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/heatmaps_top{k}.png")


def plot_convergence_per_model(aggregated, budget, output_dir, per_problem_max, min_users):
    """Plot convergence curves for each likelihood model."""

    config_str = f"per_problem_max={per_problem_max}, min_users={min_users}"

    models = sorted(set(k[0] for k in aggregated.keys()))
    strategies = list(STRATEGY_REGISTRY.keys())

    strategy_colors = {
        'Random': '#e74c3c',
        'Uncertainty': '#3498db',
        'Fixed': '#2ecc71',
        'Greedy EV': '#9b59b6',
        'Binary IG': '#f39c12'
    }

    n_models = len(models)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    # Plot for each K and metric
    for k in TOP_K_VALUES:
        metrics_to_plot = [
            (f'precision_{k}_mean', f'Precision@{k} (%)'),
            (f'recall_{k}_mean', f'Recall@{k} (%)'),
            (f'f1_{k}_mean', f'F1@{k} (%)'),
            (f'mae_tp_{k}_mean', f'MAE|TP@{k}'),
            (f'rmse_tp_{k}_mean', f'RMSE|TP@{k}'),
        ]

        for metric_key, metric_label in metrics_to_plot:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
            axes = axes.flatten()

            for idx, model in enumerate(models):
                ax = axes[idx]

                for strategy in strategies:
                    key = (model, strategy)
                    if key in aggregated and aggregated[key]:
                        by_step = aggregated[key]['by_step']
                        values = [by_step[step].get(metric_key, 0) for step in range(budget + 1)]
                        # Convert to percentage for precision/recall/f1
                        if 'precision' in metric_key or 'recall' in metric_key or 'f1' in metric_key:
                            values = [v * 100 for v in values]
                        ax.plot(range(budget + 1), values, 'o-',
                               label=strategy, color=strategy_colors.get(strategy, 'gray'),
                               linewidth=2, markersize=5)

                ax.set_xlabel('Questions Asked', fontsize=10)
                ax.set_ylabel(metric_label, fontsize=10)
                ax.set_title(f'{model}', fontsize=11, fontweight='bold')
                ax.set_xticks(range(budget + 1))
                ax.legend(loc='best', fontsize=7)
                ax.grid(True, alpha=0.3)

            for idx in range(len(models), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle(f'{metric_label} Convergence ({config_str})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Clean filename
            clean_metric = metric_key.replace('_mean', '')
            filename = f'{output_dir}/convergence_{clean_metric}_per_model.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")


def plot_strategy_comparison(aggregated, output_dir, per_problem_max, min_users):
    """Bar chart comparing best methods by strategy type for each K."""

    config_str = f"per_problem_max={per_problem_max}, min_users={min_users}"

    for k in TOP_K_VALUES:
        def get_best_result(strat_list):
            best_f1, best_result, best_key = -1, None, None
            for (m, s), result in aggregated.items():
                if s in strat_list and result and 'summary' in result:
                    f1 = result['summary'].get(f'f1_{k}', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_result = result
                        best_key = (m, s)
            return best_key, best_result

        results = {}
        for name, strat_list in [('Random', RANDOM_STRATEGIES),
                                  ('Fixed', FIXED_STRATEGIES),
                                  ('Adaptive', ADAPTIVE_STRATEGIES)]:
            key, result = get_best_result(strat_list)
            if result:
                results[name] = {'key': key, 'result': result}

        if not results:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        strategy_types = list(results.keys())

        for ax, (metric, title) in zip(axes, [
            (f'f1_{k}', f'F1@{k} (%)'),
            (f'recall_{k}', f'Recall@{k} (%)'),
            (f'mae_tp_{k}', f'MAE|TP@{k}')
        ]):
            vals = [results[s]['result']['summary'].get(metric, 0) for s in strategy_types]
            ax.bar(strategy_types, vals, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel(title)
            ax.set_title(title)

            for i, v in enumerate(vals):
                if 'mae' in metric or 'rmse' in metric:
                    label = f'{v:.3f}'
                else:
                    label = f'{v:.1f}%'
                ax.text(i, v + max(vals)*0.02, label, ha='center', fontsize=11, fontweight='bold')

            labels = [f"{s}\n({results[s]['key'][0][:12]})" for s in strategy_types]
            ax.set_xticks(range(len(strategy_types)))
            ax.set_xticklabels(labels, fontsize=10)

        plt.suptitle(f'Best Method by Strategy Type - Top-{k} ({config_str})', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/strategy_comparison_top{k}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/strategy_comparison_top{k}.png")


def generate_results_table(aggregated, budget, per_problem_max, min_users):
    """Generate text results with Top-K metrics."""

    lines = []
    lines.append("="*110)
    lines.append(f"EXPERIMENT RESULTS")
    lines.append(f"Config: per_problem_max={per_problem_max}, min_users_per_problem={min_users}")
    lines.append("="*110)

    models = sorted(set(k[0] for k in aggregated.keys()))
    strategies = list(STRATEGY_REGISTRY.keys())

    # For each K, show results
    for k in TOP_K_VALUES:
        lines.append(f"\n{'='*110}")
        lines.append(f"TOP-{k} METRICS")
        lines.append(f"{'='*110}")

        # Define metrics to find best for
        metric_configs = [
            (f'f1_{k}', 'F1', True),           # higher is better
            (f'precision_{k}', 'Precision', True),
            (f'recall_{k}', 'Recall', True),
            (f'mae_tp_{k}', 'MAE|TP', False),   # lower is better
            (f'rmse_tp_{k}', 'RMSE|TP', False),
        ]

        for metric_key, metric_name, higher_is_better in metric_configs:
            lines.append(f"\n--- Best models by {metric_name}@{k} {'(↑ higher is better)' if higher_is_better else '(↓ lower is better)'} ---")
            lines.append(f"{'Strategy':<12} {'Model':<18} {'Strat':<12} {'P@'+str(k):>7} {'R@'+str(k):>7} {'F1@'+str(k):>7} {'MAE|TP':>8} {'RMSE|TP':>9}")
            lines.append("-"*100)

            for strat_type, strat_list in [("Random", RANDOM_STRATEGIES),
                                            ("Fixed", FIXED_STRATEGIES),
                                            ("Adaptive", ADAPTIVE_STRATEGIES)]:
                best_key, best_val = None, None
                for (m, s), result in aggregated.items():
                    if s in strat_list and result and 'summary' in result:
                        val = result['summary'].get(metric_key, 0 if higher_is_better else float('inf'))
                        if best_val is None:
                            best_val = val
                            best_key = (m, s)
                        elif higher_is_better and val > best_val:
                            best_val = val
                            best_key = (m, s)
                        elif not higher_is_better and val < best_val:
                            best_val = val
                            best_key = (m, s)

                if best_key:
                    summary = aggregated[best_key]['summary']
                    model, strategy = best_key
                    lines.append(f"{strat_type:<12} {model:<18} {strategy:<12} "
                                f"{summary.get(f'precision_{k}', 0):>6.1f}% "
                                f"{summary.get(f'recall_{k}', 0):>6.1f}% "
                                f"{summary.get(f'f1_{k}', 0):>6.1f}% "
                                f"{summary.get(f'mae_tp_{k}', 0):>8.3f} "
                                f"{summary.get(f'rmse_tp_{k}', 0):>9.3f}")

        # Full matrices
        lines.append(f"\n{'='*110}")
        lines.append(f"FULL RESULTS MATRICES - TOP-{k}")
        lines.append(f"{'='*110}")

        header = f"{'Model':<20}" + "".join(f" {s:>12}" for s in strategies)

        for metric_key, metric_name, fmt, higher_is_better in [
            (f'precision_{k}', f'Precision@{k} (%)', '{:>11.1f}%', True),
            (f'recall_{k}', f'Recall@{k} (%)', '{:>11.1f}%', True),
            (f'f1_{k}', f'F1@{k} (%)', '{:>11.1f}%', True),
            (f'mae_tp_{k}', f'MAE|TP@{k}', '{:>12.3f}', False),
            (f'rmse_tp_{k}', f'RMSE|TP@{k}', '{:>12.3f}', False),
        ]:
            arrow = '↑' if higher_is_better else '↓'
            lines.append(f"\n{metric_name} ({arrow} {'higher' if higher_is_better else 'lower'} is better):")
            lines.append("-"*100)
            lines.append(header)
            lines.append("-"*100)

            for m in models:
                row = f"{m:<20}"
                for s in strategies:
                    if (m, s) in aggregated and aggregated[(m, s)] and 'summary' in aggregated[(m, s)]:
                        val = aggregated[(m, s)]['summary'].get(metric_key, 0)
                        row += fmt.format(val)
                    else:
                        row += f" {'-':>12}"
                lines.append(row)

    return "\n".join(lines)


def save_filtered_data(records, output_path, drop_zero=True):
    """Save filtered data, optionally dropping users with 0 criteria."""

    if drop_zero:
        records, n_dropped = drop_zero_criteria_users(records)
        print(f"  Dropped {n_dropped} users with 0 criteria")

    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"  Saved to {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the PEP elicitation policy')
    parser.add_argument('data_path', help='Path to raw JSONL dataset')
    parser.add_argument('--per-problem-max', type=float, default=0.10,
                        help='Remove criteria if >= this fraction within problem (default: 0.10)')
    parser.add_argument('--min-users', type=int, default=4,
                        help='Remove criteria if fewer than this many users have it (default: 4)')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Base output directory for results')
    parser.add_argument('--budget', type=int, default=5, help='Question budget (default: 5)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Fraction of problems used to fit population statistics (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the problem split and random policy (default: 42)')
    parser.add_argument('--save-filtered', type=str, default=None,
                        help='Path to save filtered dataset (optional)')
    parser.add_argument('--keep-zero-criteria', action='store_true',
                        help='Keep users with 0 criteria when saving (default: drop them)')

    args = parser.parse_args()

    # Create output folder named by config
    config_name = f"pmax{args.per_problem_max}_minU{args.min_users}_budget{args.budget}"
    output_dir = os.path.join(args.output, config_name)
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("PEP TRAINING AND EVALUATION")
    print("="*70)
    print(f"Data: {args.data_path}")
    print(f"Config: per_problem_max={args.per_problem_max}, min_users={args.min_users}, "
          f"budget={args.budget}, train_ratio={args.train_ratio}, seed={args.seed}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    records = load_records(args.data_path)
    print(f"  Loaded {len(records)} records")

    # Step 2: Plot PRE-filtering statistics
    print("\n[2/6] Plotting PRE-filtering dataset statistics...")
    pre_stats = compute_dataset_stats(records)
    plot_dataset_statistics(pre_stats, output_dir, "Pre-filtering")

    # Step 3: Filter data
    print("\n[3/6] Filtering data...")
    filtered = filter_with_min_users(
        records,
        per_problem_max=args.per_problem_max,
        min_users_per_problem=args.min_users,
        verbose=True
    )

    # Drop users with 0 criteria after filtering
    filtered, n_dropped = drop_zero_criteria_users(filtered)
    print(f"  Dropped {n_dropped} users with 0 criteria after filtering")

    # Save filtered dataset to output folder
    filtered_path = os.path.join(output_dir, 'filtered_data.jsonl')
    with open(filtered_path, 'w') as f:
        for rec in filtered:
            f.write(json.dumps(rec) + '\n')
    print(f"  Saved filtered dataset to {filtered_path}")

    # Step 4: Plot POST-filtering statistics
    print("\n[4/6] Plotting POST-filtering dataset statistics...")
    post_stats = compute_dataset_stats(filtered)
    plot_dataset_statistics(post_stats, output_dir, "Post-filtering",
                           args.per_problem_max, args.min_users)

    # Optional: Save filtered data
    if args.save_filtered:
        print(f"\n  Saving filtered dataset...")
        save_filtered_data(filtered, args.save_filtered,
                          drop_zero=not args.keep_zero_criteria)

    # Step 5: Run experiment
    print("\n[5/6] Running experiment...")
    aggregated = run_experiment(
        filtered,
        budget=args.budget,
        train_ratio=args.train_ratio,
        min_users_per_criterion=args.min_users,
        random_seed=args.seed,
    )

    # Step 6: Generate outputs
    print("\n[6/6] Generating results...")
    results_text = generate_results_table(aggregated, args.budget,
                                          args.per_problem_max, args.min_users)
    print("\n" + results_text)

    with open(f'{output_dir}/results.txt', 'w') as f:
        f.write(results_text)
    print(f"Saved: {output_dir}/results.txt")

    plot_convergence_per_model(aggregated, args.budget, output_dir,
                               args.per_problem_max, args.min_users)
    plot_heatmaps_all_metrics(aggregated, output_dir,
                              args.per_problem_max, args.min_users)
    plot_strategy_comparison(aggregated, output_dir,
                            args.per_problem_max, args.min_users)

    print("\n" + "="*70)
    print("DONE! All outputs saved to:", output_dir)
    print("="*70)


if __name__ == "__main__":
    main()
