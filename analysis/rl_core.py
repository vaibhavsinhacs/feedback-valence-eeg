"""Trial-by-trial reinforcement learning for the probabilistic selection task.

Two models are used: a single learning rate, or separate rates for positive
and negative prediction errors. Only the chosen option is updated. All option
values start at 0.5. Choices are evaluated before the feedback update.

The original model implementation was unavailable. These functions preserve
the reconstructed equations used in the study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit


ModelName = Literal["one_rate", "separate_rate"]
DEFAULT_INITIAL_VALUE = 0.5
PROBABILITY_EPSILON = 1e-12
BETA_FLOOR = 1e-8


@dataclass(frozen=True)
class RLParameters:
    """Constrained parameters for one learning model."""
    beta: float
    alpha: float | None = None
    alpha_positive: float | None = None
    alpha_negative: float | None = None


@dataclass(frozen=True)
class TrialSequence:
    """Integer-coded trial sequence used by the pure RL evaluator."""
    chosen_index: np.ndarray
    unchosen_index: np.ndarray
    reward: np.ndarray
    n_options: int = 6

    def __post_init__(self) -> None:
        chosen = np.asarray(self.chosen_index)
        unchosen = np.asarray(self.unchosen_index)
        reward = np.asarray(self.reward)
        if chosen.ndim != 1:
            raise ValueError('chosen_index must be one-dimensional.')
        if unchosen.ndim != 1:
            raise ValueError('unchosen_index must be one-dimensional.')
        if reward.ndim != 1:
            raise ValueError('reward must be one-dimensional.')
        if not len(chosen) == len(unchosen) == len(reward):
            raise ValueError('chosen_index, unchosen_index, and reward must have equal lengths.')
        if len(reward) == 0:
            raise ValueError('TrialSequence must contain at least one trial.')
        if not isinstance(self.n_options, int) or self.n_options < 2:
            raise ValueError('n_options must be an integer of at least 2.')
        if not np.issubdtype(chosen.dtype, np.integer):
            raise TypeError('chosen_index must contain integers.')
        if not np.issubdtype(unchosen.dtype, np.integer):
            raise TypeError('unchosen_index must contain integers.')
        if np.any(chosen < 0) or np.any(chosen >= self.n_options):
            raise ValueError('chosen_index contains an index outside [0, n_options).')
        if np.any(unchosen < 0) or np.any(unchosen >= self.n_options):
            raise ValueError('unchosen_index contains an index outside [0, n_options).')
        if np.any(chosen == unchosen):
            raise ValueError('Chosen and unchosen option indices must differ on every trial.')
        reward_numeric = reward.astype(float)
        if not np.isfinite(reward_numeric).all():
            raise ValueError('reward must contain only finite values.')
        if not np.isin(reward_numeric, [0.0, 1.0]).all():
            raise ValueError('reward must be binary, coded 0 or 1.')
        object.__setattr__(self, 'chosen_index', chosen.astype(np.int64, copy=True))
        object.__setattr__(self, 'unchosen_index', unchosen.astype(np.int64, copy=True))
        object.__setattr__(self, 'reward', reward_numeric.astype(float, copy=True))

    @property
    def n_trials(self) -> int:
        """Number of trials in the sequence."""
        return len(self.reward)


@dataclass(frozen=True)
class EvaluationResult:
    """Likelihood and latent variables returned by model evaluation."""
    log_likelihood: float
    negative_log_likelihood: float
    trialwise: pd.DataFrame
    final_q_values: np.ndarray


def stable_softplus(x: float) -> float:
    """Numerically stable scalar softplus."""
    return float(np.logaddexp(0.0, float(x)))


def inverse_softplus(y: float) -> float:
    """Numerically stable inverse softplus for strictly positive y."""
    value = float(y)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError('inverse_softplus requires a finite positive value.')
    if value > 50.0:
        return value
    return float(value + np.log(-np.expm1(-value)))


def learning_rate_to_unconstrained(alpha: float) -> float:
    """Map a learning rate in the open interval (0, 1) to the real line."""
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError('Learning-rate inverse transform requires 0 < alpha < 1.')
    return float(np.log(value) - np.log1p(-value))


def beta_to_unconstrained(beta: float) -> float:
    """Map beta > BETA_FLOOR to the real line."""
    value = float(beta)
    if not np.isfinite(value) or value <= BETA_FLOOR:
        raise ValueError(f'beta inverse transform requires beta > {BETA_FLOOR}.')
    return inverse_softplus(value - BETA_FLOOR)


def unpack_parameters(theta: np.ndarray, model: ModelName) -> RLParameters:
    """Transform unconstrained optimizer coordinates to model parameters."""
    values = np.asarray(theta, dtype=float)
    if values.ndim != 1:
        raise ValueError('theta must be one-dimensional.')
    if not np.isfinite(values).all():
        raise ValueError('theta must contain only finite values.')
    if model == 'one_rate':
        if len(values) != 2:
            raise ValueError('one_rate requires theta=[alpha, beta] coordinates.')
        return RLParameters(alpha=float(expit(values[0])), beta=stable_softplus(values[1]) + BETA_FLOOR)
    if model == 'separate_rate':
        if len(values) != 3:
            raise ValueError('separate_rate requires theta=[alpha_positive, alpha_negative, beta] coordinates.')
        return RLParameters(
            alpha_positive=float(expit(values[0])),
            alpha_negative=float(expit(values[1])),
            beta=stable_softplus(values[2]) + BETA_FLOOR
        )
    raise ValueError(f'Unknown model: {model}')


def pack_parameters(parameters: RLParameters, model: ModelName) -> np.ndarray:
    """Transform constrained parameters to unconstrained coordinates."""
    validate_parameters(parameters, model)
    if model == 'one_rate':
        assert parameters.alpha is not None
        return np.array(
            [learning_rate_to_unconstrained(parameters.alpha), beta_to_unconstrained(parameters.beta)],
            dtype=float
        )
    if model == 'separate_rate':
        assert parameters.alpha_positive is not None
        assert parameters.alpha_negative is not None
        return np.array(
            [
                learning_rate_to_unconstrained(parameters.alpha_positive),
                learning_rate_to_unconstrained(parameters.alpha_negative),
                beta_to_unconstrained(parameters.beta)
            ],
            dtype=float
        )
    raise ValueError(f'Unknown model: {model}')


def validate_parameters(parameters: RLParameters, model: ModelName) -> None:
    """Validate constrained parameter values for one model."""
    beta = float(parameters.beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError('beta must be finite and strictly positive.')
    if model == 'one_rate':
        if parameters.alpha is None:
            raise ValueError('one_rate requires alpha.')
        alpha = float(parameters.alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError('alpha must lie in [0, 1].')
        return
    if model == 'separate_rate':
        if parameters.alpha_positive is None:
            raise ValueError('separate_rate requires alpha_positive.')
        if parameters.alpha_negative is None:
            raise ValueError('separate_rate requires alpha_negative.')
        alpha_positive = float(parameters.alpha_positive)
        alpha_negative = float(parameters.alpha_negative)
        if not np.isfinite(alpha_positive) or not 0.0 <= alpha_positive <= 1.0:
            raise ValueError('alpha_positive must lie in [0, 1].')
        if not np.isfinite(alpha_negative) or not 0.0 <= alpha_negative <= 1.0:
            raise ValueError('alpha_negative must lie in [0, 1].')
        return
    raise ValueError(f'Unknown model: {model}')


def binary_choice_probability(q_chosen: float, q_unchosen: float, beta: float) -> float:
    """Probability assigned to the observed chosen option."""
    q_c = float(q_chosen)
    q_u = float(q_unchosen)
    inverse_temperature = float(beta)
    if not np.isfinite(q_c) or not np.isfinite(q_u):
        raise ValueError('Q values must be finite.')
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0.0:
        raise ValueError('beta must be finite and strictly positive.')
    probability = float(expit(inverse_temperature * (q_c - q_u)))
    return float(np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))


def select_learning_rate(prediction_error: float, parameters: RLParameters, model: ModelName) -> float:
    """Select the learning rate used for one feedback update."""
    validate_parameters(parameters, model)
    if model == 'one_rate':
        assert parameters.alpha is not None
        return float(parameters.alpha)
    if model == 'separate_rate':
        if prediction_error >= 0.0:
            assert parameters.alpha_positive is not None
            return float(parameters.alpha_positive)
        assert parameters.alpha_negative is not None
        return float(parameters.alpha_negative)
    raise ValueError(f'Unknown model: {model}')


def log_likelihood_only(
    sequence: TrialSequence,
    parameters: RLParameters,
    model: ModelName,
    *,
    initial_value: float=DEFAULT_INITIAL_VALUE
) -> float:
    """Evaluate log likelihood without constructing trialwise output.

    This is the optimizer-facing evaluator. It uses the same pre-update
    likelihood and chosen-option-only update equations as evaluate_sequence,
    but avoids DataFrame allocation on every optimizer function evaluation.
    """
    validate_parameters(parameters, model)
    q_initial = float(initial_value)
    if not np.isfinite(q_initial) or not 0.0 <= q_initial <= 1.0:
        raise ValueError('initial_value must lie in [0, 1].')
    q_values = np.full(sequence.n_options, q_initial, dtype=float)
    chosen_indices = sequence.chosen_index
    unchosen_indices = sequence.unchosen_index
    rewards = sequence.reward
    beta = float(parameters.beta)
    log_likelihood = 0.0
    if model == 'one_rate':
        assert parameters.alpha is not None
        alpha = float(parameters.alpha)
        for trial_index in range(sequence.n_trials):
            chosen = int(chosen_indices[trial_index])
            unchosen = int(unchosen_indices[trial_index])
            reward = float(rewards[trial_index])
            q_chosen = float(q_values[chosen])
            q_unchosen = float(q_values[unchosen])
            probability = binary_choice_probability(q_chosen, q_unchosen, beta)
            log_likelihood += float(np.log(probability))
            prediction_error = reward - q_chosen
            q_values[chosen] = q_chosen + alpha * prediction_error
        return float(log_likelihood)
    if model == 'separate_rate':
        assert parameters.alpha_positive is not None
        assert parameters.alpha_negative is not None
        alpha_positive = float(parameters.alpha_positive)
        alpha_negative = float(parameters.alpha_negative)
        for trial_index in range(sequence.n_trials):
            chosen = int(chosen_indices[trial_index])
            unchosen = int(unchosen_indices[trial_index])
            reward = float(rewards[trial_index])
            q_chosen = float(q_values[chosen])
            q_unchosen = float(q_values[unchosen])
            probability = binary_choice_probability(q_chosen, q_unchosen, beta)
            log_likelihood += float(np.log(probability))
            prediction_error = reward - q_chosen
            alpha_used = alpha_positive if prediction_error >= 0.0 else alpha_negative
            q_values[chosen] = q_chosen + alpha_used * prediction_error
        return float(log_likelihood)
    raise ValueError(f'Unknown model: {model}')


def evaluate_sequence(
    sequence: TrialSequence,
    parameters: RLParameters,
    model: ModelName,
    *,
    initial_value: float=DEFAULT_INITIAL_VALUE
) -> EvaluationResult:
    """Evaluate likelihood and trialwise latent variables.

    The likelihood is evaluated before each feedback update. Only the chosen
    option is updated.
    """
    validate_parameters(parameters, model)
    q_initial = float(initial_value)
    if not np.isfinite(q_initial) or not 0.0 <= q_initial <= 1.0:
        raise ValueError('initial_value must lie in [0, 1].')
    q_values = np.full(sequence.n_options, q_initial, dtype=float)
    log_likelihood = 0.0
    rows: list[dict[str, float | int | str]] = []
    for trial_index in range(sequence.n_trials):
        chosen = int(sequence.chosen_index[trial_index])
        unchosen = int(sequence.unchosen_index[trial_index])
        reward = float(sequence.reward[trial_index])
        q_chosen_pre = float(q_values[chosen])
        q_unchosen_pre = float(q_values[unchosen])
        choice_probability = binary_choice_probability(
            q_chosen=q_chosen_pre,
            q_unchosen=q_unchosen_pre,
            beta=parameters.beta
        )
        choice_log_probability = float(np.log(choice_probability))
        prediction_error = reward - q_chosen_pre
        alpha_used = select_learning_rate(prediction_error, parameters, model)
        q_chosen_post = q_chosen_pre + alpha_used * prediction_error
        log_likelihood += choice_log_probability
        q_values[chosen] = q_chosen_post
        rows.append({
            'trial_index_zero_based': trial_index,
            'chosen_index': chosen,
            'unchosen_index': unchosen,
            'reward': reward,
            'q_chosen_pre': q_chosen_pre,
            'q_unchosen_pre': q_unchosen_pre,
            'choice_probability': choice_probability,
            'choice_log_probability': choice_log_probability,
            'prediction_error_signed': prediction_error,
            'prediction_error_absolute': abs(prediction_error),
            'prediction_error_positive': max(prediction_error, 0.0),
            'prediction_error_negative': min(prediction_error, 0.0),
            'prediction_error_negative_absolute': abs(min(prediction_error, 0.0)),
            'alpha_used': alpha_used,
            'q_chosen_post': q_chosen_post,
            'model': model
        })
    return EvaluationResult(
        log_likelihood=float(log_likelihood),
        negative_log_likelihood=float(-log_likelihood),
        trialwise=pd.DataFrame.from_records(rows),
        final_q_values=q_values.copy()
    )


def negative_log_likelihood(
    theta: np.ndarray,
    sequence: TrialSequence,
    model: ModelName,
    *,
    initial_value: float=DEFAULT_INITIAL_VALUE
) -> float:
    """Optimizer-facing negative log likelihood."""
    parameters = unpack_parameters(theta, model)
    log_likelihood = log_likelihood_only(sequence, parameters, model, initial_value=initial_value)
    negative_log_likelihood_value = float(-log_likelihood)
    if not np.isfinite(negative_log_likelihood_value):
        return 1000000000000.0
    return negative_log_likelihood_value
