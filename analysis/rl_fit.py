"""Maximum-likelihood fitting of the behavioral learning models.

Fits use ten starting points and Nelder-Mead optimization. Random starting
points are seeded. A separate set of explicit starts can also be supplied.
The original participant-specific starting-value table is not distributed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import rl_core
from .rl_core import ModelName, RLParameters, TrialSequence


DEFAULT_N_STARTS = 10
DEFAULT_START_SEED = 20_260_717
ALPHA_START_MIN, ALPHA_START_MAX = 0.02, 0.98
BETA_START_MIN, BETA_START_MAX = 0.10, 20.0
EXPLICIT_START_ALPHA_EPSILON = 1e-10
EXPLICIT_START_BETA_EPSILON = 1e-10
SUCCESSFUL_START_NLL_TOLERANCE = 1e-4
NELDER_MEAD_OPTIONS = {
    "maxiter": 10_000, "maxfev": 20_000, "xatol": 1e-8,
    "fatol": 1e-8, "disp": False, "adaptive": False,
}


@dataclass(frozen=True)
class ModelFit:
    """Best fit and complete multistart diagnostics for one model."""
    model: ModelName
    parameters: RLParameters
    log_likelihood: float
    negative_log_likelihood: float
    aic: float
    n_trials: int
    n_parameters: int
    n_starts: int
    seed: int
    best_start_index: int
    converged: bool
    optimizer_message: str
    diagnostics: pd.DataFrame


def parameter_count(model: ModelName) -> int:
    """Number of fitted parameters in one learning model."""
    if model == 'one_rate':
        return 2
    if model == 'separate_rate':
        return 3
    raise ValueError(f'Unknown model: {model}')


def _validate_start_configuration(n_starts: int, seed: int) -> None:
    """Validate deterministic multistart configuration."""
    if not isinstance(n_starts, int) or n_starts < 1:
        raise ValueError('n_starts must be a positive integer.')
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')


def _sample_beta(rng: np.random.Generator) -> float:
    """Sample beta log-uniformly across the documented positive domain."""
    log_beta = rng.uniform(np.log(BETA_START_MIN), np.log(BETA_START_MAX))
    return float(np.exp(log_beta))


def make_start_parameters(
    model: ModelName,
    *,
    n_starts: int=DEFAULT_N_STARTS,
    seed: int=DEFAULT_START_SEED
) -> list[RLParameters]:
    """Create deterministic constrained start parameters.

    Start 0 is a fixed central anchor. Remaining starts are seeded draws:
    learning rates are uniform on [0.02, 0.98], and beta is log-uniform on
    [0.10, 20]. The exact start coordinates are a clean-reconstruction choice,
    not recovered original-code behavior.
    """
    _validate_start_configuration(n_starts, seed)
    rng = np.random.default_rng(seed)
    if model == 'one_rate':
        starts = [RLParameters(alpha=0.5, beta=1.0)]
        while len(starts) < n_starts:
            starts.append(RLParameters(
                alpha=float(rng.uniform(ALPHA_START_MIN, ALPHA_START_MAX)),
                beta=_sample_beta(rng)
            ))
        return starts
    if model == 'separate_rate':
        starts = [RLParameters(alpha_positive=0.5, alpha_negative=0.5, beta=1.0)]
        while len(starts) < n_starts:
            starts.append(RLParameters(
                alpha_positive=float(rng.uniform(ALPHA_START_MIN, ALPHA_START_MAX)),
                alpha_negative=float(rng.uniform(ALPHA_START_MIN, ALPHA_START_MAX)),
                beta=_sample_beta(rng)
            ))
        return starts
    raise ValueError(f'Unknown model: {model}')


def make_start_points(
    model: ModelName,
    *,
    n_starts: int=DEFAULT_N_STARTS,
    seed: int=DEFAULT_START_SEED
) -> list[np.ndarray]:
    """Create deterministic unconstrained optimizer start vectors."""
    return [rl_core.pack_parameters(parameters, model) for parameters in make_start_parameters(model, n_starts=n_starts, seed=seed)]


def _diagnostic_parameter_columns(parameters: RLParameters) -> dict[str, float]:
    """Flatten constrained parameters for one diagnostics row."""
    return {
        'alpha': np.nan if parameters.alpha is None else float(parameters.alpha),
        'alpha_positive': np.nan if parameters.alpha_positive is None else float(parameters.alpha_positive),
        'alpha_negative': np.nan if parameters.alpha_negative is None else float(parameters.alpha_negative),
        'beta': float(parameters.beta)
    }


def project_parameters_to_transform_interior(parameters: RLParameters, model: ModelName) -> RLParameters:
    """Project a valid constrained start into the inverse-transform domain.

    The RL models permit learning rates on the closed interval [0, 1] and
    beta values down to the numerical floor produced by the forward
    transform. Exact boundary values cannot be inverted to finite optimizer
    coordinates. This helper deterministically moves only those start values
    into the open transform domain.
    """
    rl_core.validate_parameters(parameters, model)
    beta = max(float(parameters.beta), float(rl_core.BETA_FLOOR + EXPLICIT_START_BETA_EPSILON))
    if model == 'one_rate':
        assert parameters.alpha is not None
        alpha = float(np.clip(
            float(parameters.alpha),
            EXPLICIT_START_ALPHA_EPSILON,
            1.0 - EXPLICIT_START_ALPHA_EPSILON
        ))
        return RLParameters(alpha=alpha, beta=beta)
    if model == 'separate_rate':
        assert parameters.alpha_positive is not None
        assert parameters.alpha_negative is not None
        alpha_positive = float(np.clip(
            float(parameters.alpha_positive),
            EXPLICIT_START_ALPHA_EPSILON,
            1.0 - EXPLICIT_START_ALPHA_EPSILON
        ))
        alpha_negative = float(np.clip(
            float(parameters.alpha_negative),
            EXPLICIT_START_ALPHA_EPSILON,
            1.0 - EXPLICIT_START_ALPHA_EPSILON
        ))
        return RLParameters(alpha_positive=alpha_positive, alpha_negative=alpha_negative, beta=beta)
    raise ValueError(f'Unknown model: {model}')


def prepare_start_points(
    model: ModelName,
    *,
    n_starts: int,
    seed: int,
    start_parameters: Sequence[RLParameters] | None
) -> list[np.ndarray]:
    """Return validated optimizer starts.

    When ``start_parameters`` is omitted, the seeded start design is
    used. When explicit starts are provided, their order is preserved and the
    list must contain exactly ``n_starts`` valid parameter objects.
    """
    _validate_start_configuration(n_starts, seed)
    if start_parameters is None:
        return make_start_points(model, n_starts=n_starts, seed=seed)
    starts = list(start_parameters)
    if len(starts) != n_starts:
        raise ValueError(f'Explicit start_parameters must contain exactly {n_starts} starts; received {len(starts)}.')
    packed: list[np.ndarray] = []
    for start_index, parameters in enumerate(starts):
        try:
            projected = project_parameters_to_transform_interior(parameters, model)
            packed.append(rl_core.pack_parameters(projected, model))
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid explicit start at index {start_index}: {error}') from error
    return packed


def select_best_start_index(
    optimization_results: Sequence[dict[str, Any]],
    *,
    successful_nll_tolerance: float=SUCCESSFUL_START_NLL_TOLERANCE
) -> int:
    """Select the fit endpoint using a success-within-basin rule.

    The absolute best finite endpoint defines the reference objective. When
    at least one successfully terminated finite endpoint lies within
    ``successful_nll_tolerance`` of that reference, the best such successful
    endpoint is selected. Otherwise, the absolute best finite endpoint is
    retained. Ties are resolved by lower start index.
    """
    if not np.isfinite(successful_nll_tolerance) or successful_nll_tolerance < 0.0:
        raise ValueError('successful_nll_tolerance must be finite and nonnegative.')
    finite_indices = [index for index, item in enumerate(optimization_results) if bool(item['finite'])]
    if not finite_indices:
        raise RuntimeError('No finite optimization result.')
    best_finite_index = min(
        finite_indices,
        key=lambda index: (float(optimization_results[index]['nll']), index)
    )
    best_finite_nll = float(optimization_results[best_finite_index]['nll'])
    eligible_successful_indices = [index for index in finite_indices if bool(optimization_results[index]['result'].success) and float(optimization_results[index]['nll']) <= best_finite_nll + successful_nll_tolerance]
    if eligible_successful_indices:
        return min(
            eligible_successful_indices,
            key=lambda index: (float(optimization_results[index]['nll']), index)
        )
    return best_finite_index


def fit_model(
    sequence: TrialSequence,
    model: ModelName,
    *,
    n_starts: int=DEFAULT_N_STARTS,
    seed: int=DEFAULT_START_SEED,
    optimizer_options: dict[str, Any] | None=None,
    start_parameters: Sequence[RLParameters] | None=None
) -> ModelFit:
    """Fit one model by deterministic multistart Nelder-Mead."""
    _validate_start_configuration(n_starts, seed)
    options = dict(NELDER_MEAD_OPTIONS)
    if optimizer_options is not None:
        options.update(optimizer_options)
    starts = prepare_start_points(model, n_starts=n_starts, seed=seed, start_parameters=start_parameters)
    records: list[dict[str, Any]] = []
    optimization_results = []
    for start_index, start_theta in enumerate(starts):
        initial_parameters = rl_core.unpack_parameters(start_theta, model)
        initial_nll = rl_core.negative_log_likelihood(start_theta, sequence, model)
        result = minimize(
            rl_core.negative_log_likelihood,
            x0=start_theta,
            args=(sequence, model),
            method='Nelder-Mead',
            options=options
        )
        final_theta = np.asarray(result.x, dtype=float)
        final_parameters = rl_core.unpack_parameters(final_theta, model)
        final_nll = rl_core.negative_log_likelihood(final_theta, sequence, model)
        finite_result = bool(np.isfinite(final_nll) and np.isfinite(final_theta).all())
        records.append({
            'model': model,
            'start_index': start_index,
            'seed': seed,
            'initial_negative_log_likelihood': float(initial_nll),
            'final_negative_log_likelihood': float(final_nll),
            'improvement': float(initial_nll - final_nll),
            'optimizer_success': bool(result.success),
            'finite_result': finite_result,
            'optimizer_status': int(result.status),
            'optimizer_message': str(result.message),
            'optimizer_iterations': int(result.nit),
            'optimizer_function_evaluations': int(result.nfev),
            **{
                f'initial_{key}': value for key,
                value in _diagnostic_parameter_columns(initial_parameters).items()
            },
            **{
                f'final_{key}': value for key,
                value in _diagnostic_parameter_columns(final_parameters).items()
            }
        })
        optimization_results.append({
            'result': result,
            'theta': final_theta,
            'parameters': final_parameters,
            'nll': float(final_nll),
            'finite': finite_result
        })
    try:
        best_start_index = select_best_start_index(optimization_results)
    except RuntimeError as error:
        raise RuntimeError(f'No finite optimization result for model {model}.') from error
    best = optimization_results[best_start_index]
    evaluated = rl_core.evaluate_sequence(sequence, best['parameters'], model)
    n_parameters = parameter_count(model)
    aic = 2.0 * n_parameters - 2.0 * evaluated.log_likelihood
    diagnostics = pd.DataFrame.from_records(records).sort_values('start_index', kind='stable').reset_index(drop=True)
    optimizer_result = best['result']
    return ModelFit(
        model=model,
        parameters=best['parameters'],
        log_likelihood=evaluated.log_likelihood,
        negative_log_likelihood=evaluated.negative_log_likelihood,
        aic=float(aic),
        n_trials=sequence.n_trials,
        n_parameters=n_parameters,
        n_starts=n_starts,
        seed=seed,
        best_start_index=best_start_index,
        converged=bool(optimizer_result.success),
        optimizer_message=str(optimizer_result.message),
        diagnostics=diagnostics
    )


def fit_benchmark_models(
    sequence: TrialSequence,
    *,
    n_starts: int=DEFAULT_N_STARTS,
    seed: int=DEFAULT_START_SEED,
    optimizer_options: dict[str, Any] | None=None
) -> dict[ModelName, ModelFit]:
    """Fit both learning-model variants."""
    return {
        'one_rate': fit_model(
            sequence,
            'one_rate',
            n_starts=n_starts,
            seed=seed,
            optimizer_options=optimizer_options
        ),
        'separate_rate': fit_model(
            sequence,
            'separate_rate',
            n_starts=n_starts,
            seed=seed,
            optimizer_options=optimizer_options
        )
    }


def simulate_task_sequence(
    *,
    offered_pairs: np.ndarray,
    reward_probabilities: np.ndarray,
    parameters: RLParameters,
    model: ModelName,
    seed: int,
    initial_value: float=rl_core.DEFAULT_INITIAL_VALUE
) -> TrialSequence:
    """Simulate choices and binary feedback for recovery tests.

    Parameters
    ----------
    offered_pairs
        Integer array with shape (n_trials, 2).
    reward_probabilities
        One Bernoulli reward probability per option.
    """
    rl_core.validate_parameters(parameters, model)
    pairs = np.asarray(offered_pairs)
    probabilities = np.asarray(reward_probabilities, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('offered_pairs must have shape (n_trials, 2).')
    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError('offered_pairs must contain integer indices.')
    if probabilities.ndim != 1:
        raise ValueError('reward_probabilities must be one-dimensional.')
    if len(probabilities) < 2:
        raise ValueError('At least two reward probabilities are required.')
    if not np.isfinite(probabilities).all() or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise ValueError('Reward probabilities must lie in [0, 1].')
    if np.any(pairs < 0) or np.any(pairs >= len(probabilities)):
        raise ValueError('offered_pairs contains an invalid option index.')
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError('Each offered pair must contain distinct options.')
    rng = np.random.default_rng(seed)
    q_values = np.full(len(probabilities), float(initial_value), dtype=float)
    chosen_indices: list[int] = []
    unchosen_indices: list[int] = []
    rewards: list[float] = []
    for first, second in pairs:
        first_index = int(first)
        second_index = int(second)
        probability_choose_first = rl_core.binary_choice_probability(
            q_values[first_index],
            q_values[second_index],
            parameters.beta
        )
        choose_first = bool(rng.random() < probability_choose_first)
        chosen = first_index if choose_first else second_index
        unchosen = second_index if choose_first else first_index
        reward = float(rng.random() < probabilities[chosen])
        prediction_error = reward - q_values[chosen]
        alpha_used = rl_core.select_learning_rate(prediction_error, parameters, model)
        q_values[chosen] = q_values[chosen] + alpha_used * prediction_error
        chosen_indices.append(chosen)
        unchosen_indices.append(unchosen)
        rewards.append(reward)
    return TrialSequence(
        chosen_index=np.asarray(chosen_indices, dtype=int),
        unchosen_index=np.asarray(unchosen_indices, dtype=int),
        reward=np.asarray(rewards, dtype=float),
        n_options=len(probabilities)
    )
