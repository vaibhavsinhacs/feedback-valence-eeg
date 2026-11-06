"""Participant fixed-effects regression and cluster-robust inference.

Formulas are passed directly to the regression design function.
The CR1 covariance estimator clusters observations by participant and uses
clusters minus one degrees of freedom for the reported t tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import patsy
from scipy import stats


def make_design(formula: str, sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    working = sample.reset_index(drop=True).copy()
    outcome_frame, design_frame = patsy.dmatrices(
        formula,
        working,
        return_type='dataframe',
        NA_action='raise'
    )
    outcome = outcome_frame.iloc[:, 0].to_numpy(dtype=float)
    design = design_frame.to_numpy(dtype=float)
    terms = list(design_frame.columns)
    groups = working['participant_id'].astype('string').str.strip().reset_index(drop=True)
    if not np.isfinite(outcome).all() or not np.isfinite(design).all():
        raise RuntimeError('Nonfinite design or outcome values.')
    return (outcome, design, terms, groups)


def fit_ols_cr1(outcome: np.ndarray, design: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    outcome = np.asarray(outcome, dtype=float)
    design = np.asarray(design, dtype=float)
    rows, columns = design.shape
    rank = int(np.linalg.matrix_rank(design))
    if rank != columns:
        raise RuntimeError('Design is not full column rank.')
    group_codes, group_levels = pd.factorize(groups, sort=True)
    clusters = int(len(group_levels))
    if clusters <= 1:
        raise RuntimeError('Insufficient clusters.')
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    fitted = design @ coefficients
    residuals = outcome - fitted
    xtx_inverse = np.linalg.inv(design.T @ design)
    meat = np.zeros((columns, columns), dtype=float)
    for cluster_code in range(clusters):
        mask = group_codes == cluster_code
        cluster_score = design[mask].T @ residuals[mask]
        meat += np.outer(cluster_score, cluster_score)
    correction = clusters / (clusters - 1) * (rows - 1) / (rows - columns)
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    covariance = (covariance + covariance.T) / 2.0
    variances = np.diag(covariance)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise RuntimeError('Invalid CR1 coefficient variance.')
    standard_errors = np.sqrt(variances)
    t_statistics = coefficients / standard_errors
    inference_df = clusters - 1
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df=inference_df)
    critical_value = float(stats.t.ppf(0.975, df=inference_df))
    ci_lower = coefficients - critical_value * standard_errors
    ci_upper = coefficients + critical_value * standard_errors
    if not all((np.isfinite(values).all() for values in [
        coefficients,
        standard_errors,
        t_statistics,
        p_values,
        ci_lower,
        ci_upper
    ])):
        raise RuntimeError('Nonfinite OLS or CR1 results.')
    return {
        'coefficients': coefficients,
        'fitted': fitted,
        'residuals': residuals,
        'xtx_inverse': xtx_inverse,
        'covariance': covariance,
        'standard_errors': standard_errors,
        't_statistics': t_statistics,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rank': rank,
        'rows': rows,
        'columns': columns,
        'clusters': clusters,
        'group_codes': group_codes,
        'group_levels': [str(value) for value in group_levels],
        'correction': float(correction),
        'inference_df': int(inference_df),
        'condition_number': float(np.linalg.cond(design))
    }


def result_for_term(fit: dict[str, Any], terms: list[str], term: str) -> dict[str, Any]:
    if term not in terms:
        raise RuntimeError(f'Term missing from design: {term}')
    index = terms.index(term)
    return {
        'term': term,
        'estimate': float(fit['coefficients'][index]),
        'cluster_robust_se_cr1': float(fit['standard_errors'][index]),
        't_statistic': float(fit['t_statistics'][index]),
        'inference_df': int(fit['inference_df']),
        'p_value_two_sided': float(fit['p_values'][index]),
        'ci_95_lower': float(fit['ci_lower'][index]),
        'ci_95_upper': float(fit['ci_upper'][index]),
        'rows': int(fit['rows']),
        'clusters': int(fit['clusters'])
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or (not np.isfinite(values).all()) or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError('Invalid Holm p-value family.')
    count = len(values)
    order = np.argsort(values, kind='mergesort')
    ordered = values[order]
    adjusted_ordered = np.empty(count, dtype=float)
    running_maximum = 0.0
    for index, value in enumerate(ordered):
        multiplier = count - index
        candidate = min(1.0, multiplier * float(value))
        running_maximum = max(running_maximum, candidate)
        adjusted_ordered[index] = running_maximum
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted.tolist()
