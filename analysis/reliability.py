"""Within-session EEG reliability and randomization methods.

The scientific routines for session contrasts, split-half measures,
participant bootstrap, and sign-flip tests are separated here from the
original study's input-archive and reporting code.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from scipy import stats


JOIN_KEYS = ["analysis_role", "participant_id", "session_id", "pooled_event_index"]
FEATURE_COLUMN = "fcz_theta_4_8Hz_200_400ms_pooled_baseline_db_within_session_z"
SIGN_FLIP_REPLICATES = 9999
SIGN_FLIP_SEED = 2026073101
PARTICIPANT_BOOTSTRAP_REPLICATES = 9999
PARTICIPANT_BOOTSTRAP_SEED = 2026073103
RELIABILITY_BOOTSTRAP_REPLICATES = 9999
RELIABILITY_BOOTSTRAP_SEED = 2026073102
MINIMUM_VALID_BOOTSTRAP_FRACTION = 0.95
CONFIDENCE_LEVEL = 0.95
ALPHA = 1.0 - CONFIDENCE_LEVEL


def session_key(participant_id: Any, session_id: Any) -> str:
    return f'{participant_id}::{session_id}'


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2 or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x_centered, x_centered)) * float(np.dot(y_centered, y_centered)))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman_brown(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    denominator = 1.0 + float(r)
    if np.isclose(denominator, 0.0):
        return np.nan
    return float(2.0 * float(r) / denominator)


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered ** 2, axis=1) * np.sum(y_centered ** 2, axis=1))
    result = np.full(x.shape[0], np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > np.finfo(float).tiny)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def vector_spearman_brown(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    result = np.full_like(r, np.nan, dtype=float)
    denominator = 1.0 + r
    valid = np.isfinite(r) & np.isfinite(denominator) & (np.abs(denominator) > np.finfo(float).tiny)
    result[valid] = 2.0 * r[valid] / denominator[valid]
    return result


def percentile_interval(
    values: np.ndarray,
    confidence_level: float=CONFIDENCE_LEVEL
) -> tuple[float, float, int, float]:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    fraction = float(len(valid) / len(values)) if len(values) else 0.0
    if len(valid) == 0:
        return (np.nan, np.nan, 0, fraction)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(valid, [tail, 1.0 - tail])
    return (float(lower), float(upper), int(len(valid)), fraction)


def participant_session_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (participant_id, session_id), session in frame.groupby(['participant_id', 'session_id'], sort=True):
        positive = pd.to_numeric(
            session.loc[session['feedback_valence'].eq('positive'), FEATURE_COLUMN],
            errors='raise'
        ).to_numpy(dtype=float)
        negative = pd.to_numeric(
            session.loc[session['feedback_valence'].eq('negative'), FEATURE_COLUMN],
            errors='raise'
        ).to_numpy(dtype=float)
        if len(positive) == 0 or len(negative) == 0:
            raise RuntimeError('A measurement session lacks one valence.')
        rows.append({
            'analysis_role': str(session['analysis_role'].iloc[0]),
            'participant_id': str(participant_id),
            'session_id': str(session_id),
            'positive_trials': int(len(positive)),
            'negative_trials': int(len(negative)),
            'positive_mean_z': float(np.mean(positive)),
            'negative_mean_z': float(np.mean(negative)),
            'negative_minus_positive_z': float(np.mean(negative) - np.mean(positive))
        })
    return pd.DataFrame(rows)


def fit_fixed_effects_cr1(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.sort_values(JOIN_KEYS).reset_index(drop=True)
    y = pd.to_numeric(frame[FEATURE_COLUMN], errors='raise').to_numpy(dtype=float)
    negative = pd.to_numeric(frame['negative_indicator'], errors='raise').to_numpy(dtype=float)
    cluster_labels = np.asarray(
        [
            session_key(participant_id, session_id) for participant_id,
            session_id in zip(frame['participant_id'].astype(str), frame['session_id'].astype(str))
        ],
        dtype=object
    )
    cluster_levels = np.asarray(sorted(set(cluster_labels.tolist())), dtype=object)
    cluster_map = {label: index for index, label in enumerate(cluster_levels)}
    cluster_codes = np.asarray([cluster_map[label] for label in cluster_labels], dtype=int)
    n_observations = int(len(frame))
    n_clusters = int(len(cluster_levels))
    n_parameters = int(n_clusters + 1)
    x = np.zeros((n_observations, n_parameters), dtype=float)
    x[np.arange(n_observations), cluster_codes] = 1.0
    x[:, -1] = negative
    rank = int(np.linalg.matrix_rank(x))
    if rank != n_parameters:
        raise RuntimeError('The fixed-effects model matrix is rank deficient.')
    xtx = x.T @ x
    xty = x.T @ y
    beta = np.linalg.solve(xtx, xty)
    residual = y - x @ beta
    bread = np.linalg.inv(xtx)
    cluster_scores = np.zeros((n_clusters, n_parameters), dtype=float)
    for cluster_index in range(n_clusters):
        mask = cluster_codes == cluster_index
        x_g = x[mask]
        u_g = residual[mask]
        cluster_scores[cluster_index] = x_g.T @ u_g
    meat = cluster_scores.T @ cluster_scores
    if n_clusters <= 1 or n_observations <= n_parameters:
        raise RuntimeError('CR1 small-sample correction is not estimable.')
    cr1_factor = float(n_clusters / (n_clusters - 1) * ((n_observations - 1) / (n_observations - n_parameters)))
    covariance = cr1_factor * bread @ meat @ bread
    estimate = float(beta[-1])
    variance = float(covariance[-1, -1])
    if not np.isfinite(variance) or variance < 0.0:
        raise RuntimeError('The CR1 variance is invalid.')
    standard_error = float(math.sqrt(max(variance, 0.0)))
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        raise RuntimeError('The CR1 standard error is invalid.')
    degrees_of_freedom = int(n_clusters - 1)
    t_statistic = float(estimate / standard_error)
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom))
    critical = float(stats.t.ppf(1.0 - ALPHA / 2.0, df=degrees_of_freedom))
    ci_lower = float(estimate - critical * standard_error)
    ci_upper = float(estimate + critical * standard_error)
    y_within = y.copy()
    x_within = negative.copy()
    for cluster_index in range(n_clusters):
        mask = cluster_codes == cluster_index
        y_within[mask] -= np.mean(y[mask])
        x_within[mask] -= np.mean(negative[mask])
    denominator = float(np.dot(x_within, x_within))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError('The within-transformed estimator is invalid.')
    within_estimate = float(np.dot(x_within, y_within) / denominator)
    condition_number = float(np.linalg.cond(xtx))
    return {
        'n_observations': n_observations,
        'n_clusters': n_clusters,
        'n_parameters': n_parameters,
        'model_rank': rank,
        'xtx_condition_number': condition_number,
        'estimate': estimate,
        'within_transform_estimate': within_estimate,
        'estimate_difference': float(estimate - within_estimate),
        'CR1_factor': cr1_factor,
        'CR1_standard_error': standard_error,
        't_statistic': t_statistic,
        'degrees_of_freedom': degrees_of_freedom,
        'two_sided_p_value': p_value,
        'CI_level': CONFIDENCE_LEVEL,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper
    }


def randomization_and_bootstrap(contrast_values: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contrast_values = np.asarray(contrast_values, dtype=float)
    if len(contrast_values) < 2 or not np.isfinite(contrast_values).all():
        raise RuntimeError('The participant-session contrast vector is invalid.')
    observed_mean = float(np.mean(contrast_values))
    sign_rng = np.random.default_rng(SIGN_FLIP_SEED)
    signs = sign_rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(SIGN_FLIP_REPLICATES, len(contrast_values)),
        replace=True
    )
    sign_flip_means = np.mean(signs * contrast_values[None, :], axis=1)
    exceedances = int(np.sum(np.abs(sign_flip_means) >= abs(observed_mean)))
    sign_flip_p = float((1 + exceedances) / (SIGN_FLIP_REPLICATES + 1))
    bootstrap_rng = np.random.default_rng(PARTICIPANT_BOOTSTRAP_SEED)
    bootstrap_indices = bootstrap_rng.integers(
        0,
        len(contrast_values),
        size=(PARTICIPANT_BOOTSTRAP_REPLICATES, len(contrast_values))
    )
    bootstrap_means = np.mean(contrast_values[bootstrap_indices], axis=1)
    bootstrap_lower, bootstrap_upper, bootstrap_valid, bootstrap_valid_fraction = percentile_interval(bootstrap_means)
    result = {
        'n_sessions': int(len(contrast_values)),
        'unweighted_mean_contrast': observed_mean,
        'contrast_sample_SD': float(np.std(contrast_values, ddof=1)),
        'contrast_standard_error': float(stats.sem(contrast_values, ddof=1)),
        'sign_flip_replicates': SIGN_FLIP_REPLICATES,
        'sign_flip_seed': SIGN_FLIP_SEED,
        'sign_flip_exceedances': exceedances,
        'sign_flip_two_sided_p_value': sign_flip_p,
        'participant_bootstrap_replicates': PARTICIPANT_BOOTSTRAP_REPLICATES,
        'participant_bootstrap_seed': PARTICIPANT_BOOTSTRAP_SEED,
        'participant_bootstrap_valid_replicates': bootstrap_valid,
        'participant_bootstrap_valid_fraction': bootstrap_valid_fraction,
        'participant_bootstrap_CI_level': CONFIDENCE_LEVEL,
        'participant_bootstrap_CI_lower': bootstrap_lower,
        'participant_bootstrap_CI_upper': bootstrap_upper
    }
    sign_flip_draws = pd.DataFrame({
        'draw_index': np.arange(1, SIGN_FLIP_REPLICATES + 1, dtype=int),
        'sign_flip_mean_contrast': sign_flip_means,
        'absolute_at_least_observed': np.abs(sign_flip_means) >= abs(observed_mean)
    })
    bootstrap_draws = pd.DataFrame({
        'draw_index': np.arange(1, PARTICIPANT_BOOTSTRAP_REPLICATES + 1, dtype=int),
        'bootstrap_mean_contrast': bootstrap_means
    })
    return (result, sign_flip_draws, bootstrap_draws)


def build_half_markers(frame: pd.DataFrame, split_scheme: str) -> pd.DataFrame:
    if split_scheme not in {'odd_even', 'first_second'}:
        raise ValueError(f'Unknown split scheme: {split_scheme}')
    rows: list[dict[str, Any]] = []
    for (participant_id, session_id), session in frame.groupby(['participant_id', 'session_id'], sort=True):
        session = session.sort_values('pooled_event_index').reset_index(drop=True)
        split_n = int(2 * (len(session) // 2))
        if split_n < 4:
            raise RuntimeError('A reliability session has too few retained trials.')
        session = session.iloc[:split_n].copy()
        rank = np.arange(1, split_n + 1, dtype=int)
        if split_scheme == 'odd_even':
            first_mask = rank % 2 == 1
            second_mask = rank % 2 == 0
            first_name = 'odd'
            second_name = 'even'
        else:
            half_n = int(split_n // 2)
            first_mask = rank <= half_n
            second_mask = rank > half_n
            first_name = 'first'
            second_name = 'second'
        half_values: list[dict[str, Any]] = []
        for half_name, mask in [(first_name, first_mask), (second_name, second_mask)]:
            half = session.loc[mask].copy()
            positive = pd.to_numeric(
                half.loc[half['feedback_valence'].eq('positive'), FEATURE_COLUMN],
                errors='raise'
            ).to_numpy(dtype=float)
            negative = pd.to_numeric(
                half.loc[half['feedback_valence'].eq('negative'), FEATURE_COLUMN],
                errors='raise'
            ).to_numpy(dtype=float)
            if len(positive) == 0 or len(negative) == 0:
                raise RuntimeError(f'{participant_id} {session_id} lacks one valence in {split_scheme} half {half_name}.')
            half_values.append({
                'half_name': half_name,
                'half_total_trials': int(len(half)),
                'half_positive_trials': int(len(positive)),
                'half_negative_trials': int(len(negative)),
                'half_positive_mean': float(np.mean(positive)),
                'half_negative_mean': float(np.mean(negative)),
                'half_marker': float(np.mean(negative) - np.mean(positive))
            })
        rows.append({
            'analysis_role': str(session['analysis_role'].iloc[0]),
            'participant_id': str(participant_id),
            'session_id': str(session_id),
            'split_scheme': split_scheme,
            'retained_trials_available': int(len(frame.loc[frame['participant_id'].eq(participant_id) & frame['session_id'].eq(session_id)])),
            'split_trials_used': split_n,
            'half_1_name': half_values[0]['half_name'],
            'half_1_total_trials': half_values[0]['half_total_trials'],
            'half_1_positive_trials': half_values[0]['half_positive_trials'],
            'half_1_negative_trials': half_values[0]['half_negative_trials'],
            'half_1_positive_mean': half_values[0]['half_positive_mean'],
            'half_1_negative_mean': half_values[0]['half_negative_mean'],
            'half_1_marker': half_values[0]['half_marker'],
            'half_2_name': half_values[1]['half_name'],
            'half_2_total_trials': half_values[1]['half_total_trials'],
            'half_2_positive_trials': half_values[1]['half_positive_trials'],
            'half_2_negative_trials': half_values[1]['half_negative_trials'],
            'half_2_positive_mean': half_values[1]['half_positive_mean'],
            'half_2_negative_mean': half_values[1]['half_negative_mean'],
            'half_2_marker': half_values[1]['half_marker']
        })
    return pd.DataFrame(rows)


def reliability_result(markers: pd.DataFrame, role: str, split_scheme: str) -> tuple[dict[str, Any], pd.DataFrame]:
    x = pd.to_numeric(markers['half_1_marker'], errors='raise').to_numpy(dtype=float)
    y = pd.to_numeric(markers['half_2_marker'], errors='raise').to_numpy(dtype=float)
    raw_r = pearson_r(x, y)
    sb_r = spearman_brown(raw_r)
    if not np.isfinite(raw_r) or not np.isfinite(sb_r):
        raise RuntimeError(f'{role} {split_scheme} reliability is undefined.')
    rng = np.random.default_rng(RELIABILITY_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(markers), size=(RELIABILITY_BOOTSTRAP_REPLICATES, len(markers)))
    bootstrap_raw = rowwise_pearson(x[indices], y[indices])
    bootstrap_sb = vector_spearman_brown(bootstrap_raw)
    raw_lower, raw_upper, raw_valid, raw_valid_fraction = percentile_interval(bootstrap_raw)
    sb_lower, sb_upper, sb_valid, sb_valid_fraction = percentile_interval(bootstrap_sb)
    result = {
        'analysis_role': role,
        'split_scheme': split_scheme,
        'n_sessions': int(len(markers)),
        'raw_Pearson_r': raw_r,
        'Spearman_Brown_r': sb_r,
        'bootstrap_replicates': RELIABILITY_BOOTSTRAP_REPLICATES,
        'bootstrap_seed': RELIABILITY_BOOTSTRAP_SEED,
        'raw_valid_bootstrap_replicates': raw_valid,
        'raw_valid_bootstrap_fraction': raw_valid_fraction,
        'raw_CI_level': CONFIDENCE_LEVEL,
        'raw_CI_lower': raw_lower,
        'raw_CI_upper': raw_upper,
        'SB_valid_bootstrap_replicates': sb_valid,
        'SB_valid_bootstrap_fraction': sb_valid_fraction,
        'SB_CI_level': CONFIDENCE_LEVEL,
        'SB_CI_lower': sb_lower,
        'SB_CI_upper': sb_upper
    }
    draws = pd.DataFrame({
        'analysis_role': role,
        'split_scheme': split_scheme,
        'draw_index': np.arange(1, RELIABILITY_BOOTSTRAP_REPLICATES + 1, dtype=int),
        'raw_Pearson_r': bootstrap_raw,
        'Spearman_Brown_r': bootstrap_sb,
        'raw_defined': np.isfinite(bootstrap_raw),
        'Spearman_Brown_defined': np.isfinite(bootstrap_sb)
    })
    return (result, draws)
