"""Session fixed-effects models for the independent EEG cohort.

Session means are removed before regression. Cluster-robust (CR1) standard
errors are calculated at the session level. The model contrasts feedback
valence with trialwise signed prediction error.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd
from scipy import stats


OUTCOME_COLUMN = "fcz_theta_4_8Hz_200_400ms_pooled_baseline_db_within_session_z"
VALENCE_COLUMN = "negative_indicator"
CONFIDENCE_LEVEL = 0.95
ALPHA = 1.0 - CONFIDENCE_LEVEL


def session_codes(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    keys = frame['participant_id'].astype(str) + '::' + frame['session_id'].astype(str)
    levels = sorted(keys.unique())
    mapping = {level: index for index, level in enumerate(levels)}
    return (keys.map(mapping).to_numpy(dtype=int), levels)


def demean_by_session(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    for code in np.unique(codes):
        mask = codes == code
        output[mask] -= np.mean(output[mask], axis=0)
    return output


def fit_absorbed_cr1(
    frame: pd.DataFrame,
    predictor_columns: list[str],
    analysis_role: str,
    latent_model: str,
    eeg_model: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = pd.to_numeric(frame[OUTCOME_COLUMN], errors='raise').to_numpy(dtype=float)
    x = frame[predictor_columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    codes, levels = session_codes(frame)
    y_within = demean_by_session(y, codes)
    x_within = demean_by_session(x, codes)
    n_observations = len(y)
    n_sessions = len(levels)
    n_predictors = x_within.shape[1]
    n_parameters = n_sessions + n_predictors
    rank = int(np.linalg.matrix_rank(x_within))
    if rank != n_predictors:
        raise RuntimeError(f'Rank deficiency for {analysis_role} {latent_model} {eeg_model}: rank={rank}, predictors={n_predictors}.')
    xtx = x_within.T @ x_within
    xtx_inverse = np.linalg.inv(xtx)
    beta = xtx_inverse @ (x_within.T @ y_within)
    residual = y_within - x_within @ beta
    meat = np.zeros((n_predictors, n_predictors), dtype=float)
    for code in np.unique(codes):
        mask = codes == code
        score = x_within[mask].T @ residual[mask]
        meat += np.outer(score, score)
    if n_sessions <= 1 or n_observations <= n_parameters:
        raise RuntimeError('CR1 small-sample correction is not estimable.')
    cr1_factor = float(n_sessions / (n_sessions - 1.0) * ((n_observations - 1.0) / (n_observations - n_parameters)))
    covariance = cr1_factor * xtx_inverse @ meat @ xtx_inverse
    variances = np.diag(covariance)
    if not np.isfinite(variances).all() or np.any(variances < -1e-12):
        raise RuntimeError('Invalid CR1 covariance diagonal.')
    standard_errors = np.sqrt(np.clip(variances, 0.0, None))
    if np.any(standard_errors <= 0.0):
        raise RuntimeError('A CR1 standard error is nonpositive.')
    degrees_of_freedom = n_sessions - 1
    t_statistics = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df=degrees_of_freedom)
    critical = float(stats.t.ppf(1.0 - ALPHA / 2.0, df=degrees_of_freedom))
    ci_lower = beta - critical * standard_errors
    ci_upper = beta + critical * standard_errors
    rss = float(np.dot(residual, residual))
    tss = float(np.dot(y_within, y_within))
    within_r2 = float(1.0 - rss / tss)
    residual_variance_ml = max(rss / n_observations, np.finfo(float).tiny)
    log_likelihood = float(-0.5 * n_observations * (math.log(2.0 * math.pi) + 1.0 + math.log(residual_variance_ml)))
    aic = float(-2.0 * log_likelihood + 2.0 * n_parameters)
    bic = float(-2.0 * log_likelihood + n_parameters * math.log(n_observations))
    x_scales = np.std(x_within, axis=0, ddof=0)
    y_scale = float(np.std(y_within, ddof=0))
    standardized_x = x_within / x_scales
    condition_number = float(np.linalg.cond(standardized_x.T @ standardized_x))
    if n_predictors == 2:
        predictor_correlation = float(np.corrcoef(x_within[:, 0], x_within[:, 1])[0, 1])
        vif = float(1.0 / (1.0 - predictor_correlation ** 2))
    else:
        predictor_correlation = math.nan
        vif = 1.0
    model_row = {
        'analysis_role': analysis_role,
        'latent_model': latent_model,
        'EEG_model': eeg_model,
        'predictor_columns': ';'.join(predictor_columns),
        'n_sessions': n_sessions,
        'n_trials': n_observations,
        'n_predictors': n_predictors,
        'full_parameter_count': n_parameters,
        'predictor_rank': rank,
        'CR1_factor': cr1_factor,
        'degrees_of_freedom': degrees_of_freedom,
        'RSS_within': rss,
        'TSS_within': tss,
        'within_R2': within_r2,
        'residual_RMSE_within': float(math.sqrt(rss / n_observations)),
        'Gaussian_log_likelihood_ML': log_likelihood,
        'AIC': aic,
        'BIC': bic,
        'standardized_design_condition_number': condition_number,
        'joint_predictor_correlation': predictor_correlation,
        'joint_VIF': vif
    }
    coefficient_rows: list[dict[str, Any]] = []
    for index, predictor_column in enumerate(predictor_columns):
        predictor = 'valence' if predictor_column == VALENCE_COLUMN else 'signed_PE'
        if predictor == 'valence':
            scale_interpretation = 'negative-minus-positive FCz theta contrast in within-session outcome SD units'
        else:
            scale_interpretation = 'FCz theta change per one within-session SD increase in signed PE'
        coefficient_rows.append({
            'analysis_role': analysis_role,
            'latent_model': latent_model,
            'EEG_model': eeg_model,
            'predictor': predictor,
            'predictor_column': predictor_column,
            'n_sessions': n_sessions,
            'n_trials': n_observations,
            'estimate': float(beta[index]),
            'CR1_standard_error': float(standard_errors[index]),
            't_statistic': float(t_statistics[index]),
            'degrees_of_freedom': degrees_of_freedom,
            'two_sided_p_value': float(p_values[index]),
            'CI_level': CONFIDENCE_LEVEL,
            'CI_lower': float(ci_lower[index]),
            'CI_upper': float(ci_upper[index]),
            'predictor_within_role_SD': float(x_scales[index]),
            'outcome_within_role_SD': y_scale,
            'standardized_estimate': float(beta[index] * x_scales[index] / y_scale),
            'scale_interpretation': scale_interpretation,
            'Holm_adjusted_p_within_primary_joint': math.nan
        })
    return (model_row, coefficient_rows)


def holm_adjust_two(p_values: list[float]) -> list[float]:
    if len(p_values) != 2:
        raise ValueError('This stage only applies Holm adjustment to two joint coefficients.')
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(2, dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        multiplier = 2 - rank
        value = min(1.0, multiplier * p_values[index])
        value = max(value, previous)
        adjusted[index] = value
        previous = value
    return adjusted.tolist()
