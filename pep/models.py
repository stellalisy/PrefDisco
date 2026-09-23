"""
Likelihood Models for Preference Prediction.

Each model predicts:
- P(user has criterion c) for all criteria
- E[value | user has criterion c] for all criteria

Models use global statistics computed from training problems.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod


class BaseLikelihoodModel(ABC):
    """Base class for likelihood models."""

    name: str = "Base"

    def __init__(self, data: Dict, global_stats: Dict):
        """
        Initialize model.

        Args:
            data: Problem-specific data from prepare_problem_data()
            global_stats: Global statistics from compute_global_statistics()
        """
        self.data = data
        self.criteria = data['criteria']
        self.n_criteria = data['n_criteria']
        self.criterion_to_idx = data['criterion_to_idx']
        self.global_stats = global_stats

    @abstractmethod
    def predict(
        self,
        observed_has: List[Tuple[int, float]],
        observed_not_has: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict P(has) and E[value] for all criteria.

        Args:
            observed_has: List of (criterion_idx, value) for criteria user has
            observed_not_has: List of criterion indices user doesn't have

        Returns:
            pred_has: Array of P(has criterion) for each criterion
            pred_val: Array of E[value | has criterion] for each criterion
        """
        pass


class Uniform(BaseLikelihoodModel):
    """Uniform prior: P(has) = 0.5, E[val] = 3.0 for all criteria."""

    name = "Uniform"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)
        self.prior_has = np.full(self.n_criteria, 0.5)
        self.prior_val = np.full(self.n_criteria, 3.0)

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class Prior(BaseLikelihoodModel):
    """Global prior: uses P(has) and E[val] from training data."""

    name = "Prior"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class NaiveBayes(BaseLikelihoodModel):
    """Naive Bayes: updates P(has) using conditional probabilities."""

    name = "Naive Bayes"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

        self.pairwise_cond = global_stats['pairwise_cond']

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        obs_has_names = set(self.criteria[c] for c, v in observed_has)
        obs_not_names = set(self.criteria[c] for c in observed_not_has)

        for c in range(self.n_criteria):
            c_name = self.criteria[c]
            if c_name in obs_has_names or c_name in obs_not_names:
                continue

            log_odds = np.log(pred_has[c] / (1 - pred_has[c] + 1e-10) + 1e-10)

            for obs_name in obs_has_names:
                key = (obs_name, c_name)
                if key in self.pairwise_cond:
                    p_c_given_obs = self.pairwise_cond[key]['p_c2_given_c1']
                    p_c_given_not_obs = self.pairwise_cond[key]['p_c2_given_not_c1']
                    if p_c_given_obs > 0 and p_c_given_not_obs > 0:
                        log_odds += np.log(p_c_given_obs / p_c_given_not_obs)

            for obs_name in obs_not_names:
                key = (obs_name, c_name)
                if key in self.pairwise_cond:
                    p_c_given_not_obs = self.pairwise_cond[key]['p_c2_given_not_c1']
                    p_c_given_obs = self.pairwise_cond[key]['p_c2_given_c1']
                    if p_c_given_not_obs > 0 and p_c_given_obs > 0:
                        log_odds += np.log(p_c_given_not_obs / p_c_given_obs)

            pred_has[c] = 1 / (1 + np.exp(-np.clip(log_odds, -10, 10)))

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class ItemCF(BaseLikelihoodModel):
    """Item-based Collaborative Filtering using criterion similarity."""

    name = "Item CF"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

        # Build similarity matrix
        sim = global_stats['similarity']
        self.item_sim = np.zeros((self.n_criteria, self.n_criteria))
        for i, c1 in enumerate(self.criteria):
            for j, c2 in enumerate(self.criteria):
                if i != j and (c1, c2) in sim:
                    self.item_sim[i, j] = sim[(c1, c2)]

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        obs_has_set = set(x[0] for x in observed_has)
        obs_not_set = set(observed_not_has)

        for c in range(self.n_criteria):
            if c in obs_has_set or c in obs_not_set:
                continue
            if obs_has_set:
                sim_sum = sum(self.item_sim[c, c1] for c1 in obs_has_set)
                if obs_not_set:
                    sim_sum -= sum(self.item_sim[c, c1] for c1 in obs_not_set)
                n_obs = len(obs_has_set) + len(obs_not_set)
                pred_has[c] = np.clip(self.prior_has[c] + sim_sum / (n_obs + 1), 0.01, 0.99)

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class ItemCFValue(BaseLikelihoodModel):
    """Item CF with value correlation for better value prediction."""

    name = "Item CF + Value"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

        sim = global_stats['similarity']
        self.item_sim = np.zeros((self.n_criteria, self.n_criteria))
        for i, c1 in enumerate(self.criteria):
            for j, c2 in enumerate(self.criteria):
                if i != j and (c1, c2) in sim:
                    self.item_sim[i, j] = sim[(c1, c2)]

        self.value_corr = global_stats['value_corr']

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        obs_has_set = set(x[0] for x in observed_has)
        obs_not_set = set(observed_not_has)
        obs_values = {c: v for c, v in observed_has}

        for c in range(self.n_criteria):
            if c in obs_has_set or c in obs_not_set:
                continue

            c_name = self.criteria[c]

            # Has prediction
            if obs_has_set:
                sim_sum = sum(self.item_sim[c, c1] for c1 in obs_has_set)
                if obs_not_set:
                    sim_sum -= sum(self.item_sim[c, c1] for c1 in obs_not_set)
                n_obs = len(obs_has_set) + len(obs_not_set)
                pred_has[c] = np.clip(self.prior_has[c] + sim_sum / (n_obs + 1), 0.01, 0.99)

            # Value prediction using correlation
            if obs_has_set:
                val_adjustments = []
                for c1 in obs_has_set:
                    c1_name = self.criteria[c1]
                    key = (c1_name, c_name)
                    if key in self.value_corr:
                        corr = self.value_corr[key]
                        obs_val = obs_values[c1]
                        deviation = obs_val - self.prior_val[c1]
                        val_adjustments.append(corr * deviation * 0.5)

                if val_adjustments:
                    pred_val[c] = np.clip(self.prior_val[c] + np.mean(val_adjustments), 1, 5)

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class HybridNBCF(BaseLikelihoodModel):
    """Hybrid of Naive Bayes and Item CF."""

    name = "Hybrid NB+CF"

    def __init__(self, data: Dict, global_stats: Dict, alpha: float = 0.5):
        super().__init__(data, global_stats)
        self.nb = NaiveBayes(data, global_stats)
        self.cf = ItemCF(data, global_stats)
        self.alpha = alpha
        self.prior_has = self.cf.prior_has
        self.prior_val = self.cf.prior_val

    def predict(self, observed_has, observed_not_has):
        nb_has, nb_val = self.nb.predict(observed_has, observed_not_has)
        cf_has, cf_val = self.cf.predict(observed_has, observed_not_has)

        pred_has = self.alpha * nb_has + (1 - self.alpha) * cf_has
        pred_val = self.alpha * nb_val + (1 - self.alpha) * cf_val

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class LogisticCF(BaseLikelihoodModel):
    """Logistic CF: updates log-odds using similarity."""

    name = "Logistic CF"

    def __init__(self, data: Dict, global_stats: Dict):
        super().__init__(data, global_stats)

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

        sim = global_stats['similarity']
        self.item_sim = np.zeros((self.n_criteria, self.n_criteria))
        for i, c1 in enumerate(self.criteria):
            for j, c2 in enumerate(self.criteria):
                if i != j and (c1, c2) in sim:
                    self.item_sim[i, j] = sim[(c1, c2)]

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        obs_has_set = set(x[0] for x in observed_has)
        obs_not_set = set(observed_not_has)

        for c in range(self.n_criteria):
            if c in obs_has_set or c in obs_not_set:
                continue

            log_odds = np.log(pred_has[c] / (1 - pred_has[c] + 1e-10) + 1e-10)

            for c1 in obs_has_set:
                log_odds += self.item_sim[c, c1] * 2
            for c1 in obs_not_set:
                log_odds -= self.item_sim[c, c1] * 2

            pred_has[c] = 1 / (1 + np.exp(-np.clip(log_odds, -10, 10)))

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


class BayesianCF(BaseLikelihoodModel):
    """Bayesian CF: uses Beta prior with similarity updates."""

    name = "Bayesian CF"

    def __init__(self, data: Dict, global_stats: Dict, prior_strength: float = 2.0):
        super().__init__(data, global_stats)
        self.prior_strength = prior_strength

        priors = global_stats['priors']
        self.prior_has = np.zeros(self.n_criteria)
        self.prior_val = np.zeros(self.n_criteria)

        for i, c_name in enumerate(self.criteria):
            if c_name in priors:
                self.prior_has[i] = priors[c_name]['prior_has']
                self.prior_val[i] = priors[c_name]['prior_val']
            else:
                self.prior_has[i] = 0.3
                self.prior_val[i] = 3.0

        sim = global_stats['similarity']
        self.item_sim = np.zeros((self.n_criteria, self.n_criteria))
        for i, c1 in enumerate(self.criteria):
            for j, c2 in enumerate(self.criteria):
                if i != j and (c1, c2) in sim:
                    self.item_sim[i, j] = sim[(c1, c2)]

    def predict(self, observed_has, observed_not_has):
        pred_has = self.prior_has.copy()
        pred_val = self.prior_val.copy()

        obs_has_set = set(x[0] for x in observed_has)
        obs_not_set = set(observed_not_has)

        for c in range(self.n_criteria):
            if c in obs_has_set or c in obs_not_set:
                continue

            alpha = self.prior_has[c] * self.prior_strength
            beta = (1 - self.prior_has[c]) * self.prior_strength

            for c1 in obs_has_set:
                weight = self.item_sim[c, c1]
                alpha += weight
            for c1 in obs_not_set:
                weight = self.item_sim[c, c1]
                beta += weight

            pred_has[c] = alpha / (alpha + beta) if (alpha + beta) > 0 else self.prior_has[c]

        for (c, v) in observed_has:
            pred_has[c], pred_val[c] = 1.0, v
        for c in observed_not_has:
            pred_has[c] = 0.0

        return pred_has, pred_val


# Registry of all models
ALL_MODELS = [
    Uniform,
    Prior,
    NaiveBayes,
    ItemCF,
    ItemCFValue,
    HybridNBCF,
    LogisticCF,
    BayesianCF,
]

MODEL_REGISTRY = {M.name: M for M in ALL_MODELS}
