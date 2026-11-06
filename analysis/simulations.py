"""Simulation methods for distinguishing feedback-valence and prediction-error accounts.

These functions contain the estimators, generative mechanisms, recovery
statistics, and task designs used in the identifiability analyses. The
original batch runner, plotting, input hashes, and output packaging have
been omitted.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd
from scipy import stats


EMPIRICAL_REPLICATES = 120
EMPIRICAL_COUNT_DRAWS = 8
GENERIC_REPLICATES = 200
SPECTRAL_REPLICATES = 3000
SEED_EMPIRICAL = 2026072906
SEED_GENERIC = 2026072907
SEED_SPECTRAL = 2026072908
SYMPTOM_SLOPE = 0.275
BASE_EFFECT = 0.35
DEFAULT_HETEROGENEITY = 0.35
EMPIRICAL_NOISE_LEVELS = {"low_noise": 0.75, "moderate_noise": 1.25, "high_noise": 2.0}
MECHANISMS = [
    "null",
    "valence",
    "signed_pe",
    "unsigned_surprise",
    "within_loss_magnitude",
    "mixed_valence_signed_pe"
]
ESTIMATORS = [
    "null",
    "valence_direct",
    "signed_pe_only",
    "valence_adjusted_for_pe",
    "signed_pe_adjusted_for_valence",
    "unsigned_surprise",
    "within_loss_magnitude"
]
MODEL_IDS = [
    "null",
    "valence",
    "signed_pe",
    "joint_valence_signed_pe",
    "unsigned_surprise",
    "within_loss_magnitude"
]
TRUE_MODEL = {
    "null": "null",
    "valence": "valence",
    "signed_pe": "signed_pe",
    "unsigned_surprise": "unsigned_surprise",
    "within_loss_magnitude": "within_loss_magnitude",
    "mixed_valence_signed_pe": "joint_valence_signed_pe"
}
PRIMARY_ESTIMATOR = {
    "null": "null",
    "valence": "valence_adjusted_for_pe",
    "signed_pe": "signed_pe_adjusted_for_valence",
    "unsigned_surprise": "unsigned_surprise",
    "within_loss_magnitude": "within_loss_magnitude",
    "mixed_valence_signed_pe": "valence_adjusted_for_pe"
}
SPECTRAL_PROFILES = {
    "theta_specific": {"delta": 0.0, "theta": 1.0, "alpha": 0.0},
    "broad_low_frequency": {"delta": 0.9, "theta": 1.0, "alpha": 0.7},
    "delta_dominant_broad": {"delta": 1.0, "theta": 0.75, "alpha": 0.35},
    "theta_alpha_spread": {"delta": 0.35, "theta": 1.0, "alpha": 0.85}
}
SPECTRAL_COUNTS = [20, 40, 60, 100]
SPECTRAL_NOISE = {"low_noise": 0.65, "moderate_noise": 1.0, "high_noise": 1.5}
IDENTIFICATION_THRESHOLDS = {
    "recovery_identified": 0.60,
    "recovery_ambiguous": 0.40,
    "detection_identified": 0.75,
    "model_selection_identified": 0.60,
    "false_attribution_identified": 0.20
}
EPS = 1e-12


def assert_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {', '.join(missing)}")


def zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    sd = float(array.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError('Cannot standardize a constant/nonfinite vector.')
    return (array - float(array.mean())) / sd


def zscore_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    means = array.mean(axis=0, keepdims=True)
    sds = array.std(axis=0, ddof=1, keepdims=True)
    if not np.isfinite(sds).all() or (sds <= 0).any():
        raise RuntimeError('Cannot standardize one or more simulation columns.')
    return (array - means) / sds


def safe_corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise RuntimeError('Correlation matrices must have identical shape.')
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc, axis=0))
    result = np.divide(
        np.sum(xc * yc, axis=0),
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=denominator > 0
    )
    return result


def spearman_brown_array(r: np.ndarray) -> np.ndarray:
    values = np.asarray(r, dtype=float)
    return np.divide(
        2.0 * values,
        1.0 + values,
        out=np.full_like(values, np.nan),
        where=np.abs(1.0 + values) > EPS
    )


def slope_test_matrix(predictor: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != len(x):
        raise RuntimeError('Slope-test dimensions do not align.')
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ (design.T @ y)
    residual = y - design @ beta
    df = len(x) - design.shape[1]
    sigma2 = np.sum(residual ** 2, axis=0) / df
    se = np.sqrt(np.maximum(sigma2 * bread[1, 1], 0.0))
    slopes = beta[1, :]
    t_stat = np.divide(slopes, se, out=np.zeros_like(slopes), where=se > 0)
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    return (slopes, se, p_value)


def within_participant_standardize_matrix(values: np.ndarray, participant_slices: list[slice]) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for participant_slice in participant_slices:
        result[participant_slice, :] = zscore_columns(result[participant_slice, :])
    return result


def condition_number_and_vif(valence: np.ndarray, signed_pe: np.ndarray) -> tuple[float, float, float]:
    matrix = np.column_stack([np.ones(len(valence)), valence, signed_pe])
    condition = float(np.linalg.cond(matrix))
    correlation = float(stats.pearsonr(valence, signed_pe).statistic)
    vif = float(1.0 / max(1.0 - correlation ** 2, EPS))
    return (correlation, vif, condition)


def model_weight(design: np.ndarray, coefficient_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return the minimum-norm coefficient weight and rank diagnostics.

    A rank-deficient design is not treated as a software error because
    non-unique adjusted coefficients are themselves an identifiability
    result. The target coefficient is uniquely estimable only when its
    basis vector lies in the row space of the design matrix.
    """
    x = np.asarray(design, dtype=float)
    if x.ndim != 2:
        raise RuntimeError('Estimator design must be two-dimensional.')
    if not np.isfinite(x).all():
        raise RuntimeError('Estimator design contains nonfinite values.')
    rank = int(np.linalg.matrix_rank(x))
    n_columns = int(x.shape[1])
    pseudoinverse = np.linalg.pinv(x)
    weight = pseudoinverse[int(coefficient_index), :]
    row_space_projection = pseudoinverse @ x
    target = np.zeros(n_columns, dtype=float)
    target[int(coefficient_index)] = 1.0
    estimability_residual = float(np.linalg.norm(target - row_space_projection @ target))
    target_estimable = bool(estimability_residual <= 1e-08)
    singular_values = np.linalg.svd(x, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps * max(x.shape) * singular_values.max()]
    condition_number = float(positive.max() / positive.min()) if len(positive) > 0 else float('inf')
    diagnostics = {
        'rank': rank,
        'n_columns': n_columns,
        'rank_deficient': bool(rank < n_columns),
        'target_estimable': target_estimable,
        'estimability_residual': estimability_residual,
        'condition_number_effective': condition_number
    }
    return (weight, diagnostics)


def model_matrix(frame: pd.DataFrame, model_id: str) -> tuple[np.ndarray, list[str]]:
    covariates = [frame['rt_z'].to_numpy(dtype=float), frame['progression_z'].to_numpy(dtype=float)]
    names = ['intercept', 'rt_z', 'progression_z']
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float), *covariates]
    if model_id == 'null':
        pass
    elif model_id == 'valence':
        columns.append(frame['valence_z'].to_numpy(dtype=float))
        names.append('valence')
    elif model_id == 'signed_pe':
        columns.append(frame['signed_pe_z'].to_numpy(dtype=float))
        names.append('signed_pe')
    elif model_id == 'joint_valence_signed_pe':
        columns.extend([
            frame['valence_z'].to_numpy(dtype=float),
            frame['signed_pe_z'].to_numpy(dtype=float)
        ])
        names.extend(['valence', 'signed_pe'])
    elif model_id == 'unsigned_surprise':
        columns.append(frame['unsigned_surprise_z'].to_numpy(dtype=float))
        names.append('unsigned_surprise')
    elif model_id == 'within_loss_magnitude':
        columns.append(frame['within_loss_magnitude_z'].to_numpy(dtype=float))
        names.append('within_loss_magnitude')
    else:
        raise ValueError(f'Unknown model ID: {model_id}')
    return (np.column_stack(columns), names)


def estimator_specification(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, int]]:
    specifications: dict[str, tuple[np.ndarray, int]] = {}
    x, names = model_matrix(frame, 'null')
    specifications['null'] = (x, 0)
    x, names = model_matrix(frame, 'valence')
    specifications['valence_direct'] = (x, names.index('valence'))
    x, names = model_matrix(frame, 'signed_pe')
    specifications['signed_pe_only'] = (x, names.index('signed_pe'))
    x, names = model_matrix(frame, 'joint_valence_signed_pe')
    specifications['valence_adjusted_for_pe'] = (x, names.index('valence'))
    specifications['signed_pe_adjusted_for_valence'] = (x, names.index('signed_pe'))
    x, names = model_matrix(frame, 'unsigned_surprise')
    specifications['unsigned_surprise'] = (x, names.index('unsigned_surprise'))
    x, names = model_matrix(frame, 'within_loss_magnitude')
    specifications['within_loss_magnitude'] = (x, names.index('within_loss_magnitude'))
    return specifications


def prepare_design(frame: pd.DataFrame, design_id: str) -> dict[str, Any]:
    required = [
        'participant_id',
        'stai_z',
        'valence_z',
        'signed_pe_z',
        'unsigned_surprise_z',
        'within_loss_magnitude_z',
        'rt_z',
        'progression_z'
    ]
    assert_columns(frame, required, design_id)
    ordered_parts: list[pd.DataFrame] = []
    participant_slices: list[slice] = []
    participants: list[str] = []
    stai_values: list[float] = []
    estimator_weights = {estimator_id: [] for estimator_id in ESTIMATORS}
    estimator_rank_diagnostics = {estimator_id: [] for estimator_id in ESTIMATORS}
    start = 0
    for participant_id, part in frame.groupby('participant_id', sort=True):
        part = part.reset_index(drop=True)
        stop = start + len(part)
        participant_slices.append(slice(start, stop))
        participants.append(str(participant_id))
        stai_values.append(float(part['stai_z'].iloc[0]))
        ordered_parts.append(part)
        specs = estimator_specification(part)
        for estimator_id in ESTIMATORS:
            x, coefficient_index = specs[estimator_id]
            weight, diagnostics = model_weight(x, coefficient_index)
            estimator_weights[estimator_id].append(weight)
            estimator_rank_diagnostics[estimator_id].append({
                'participant_id': str(participant_id),
                **diagnostics
            })
        start = stop
    ordered = pd.concat(ordered_parts, ignore_index=True)
    stai = np.asarray(stai_values, dtype=float)
    if len(participants) < 20:
        raise RuntimeError(f'{design_id}: too few participants.')
    global_models: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    global_model_diagnostics: dict[str, dict[str, Any]] = {}
    participant_codes = pd.Categorical(
        ordered['participant_id'],
        categories=participants,
        ordered=True
    ).codes
    participant_dummies = pd.get_dummies(participant_codes, drop_first=True, dtype=float).to_numpy(dtype=float)
    base = np.column_stack([
        np.ones(len(ordered)),
        participant_dummies,
        ordered['rt_z'].to_numpy(dtype=float),
        ordered['progression_z'].to_numpy(dtype=float)
    ])
    for model_id in MODEL_IDS:
        if model_id == 'null':
            x = base
        elif model_id == 'valence':
            x = np.column_stack([base, ordered['valence_z'].to_numpy(dtype=float)])
        elif model_id == 'signed_pe':
            x = np.column_stack([base, ordered['signed_pe_z'].to_numpy(dtype=float)])
        elif model_id == 'joint_valence_signed_pe':
            x = np.column_stack([
                base,
                ordered['valence_z'].to_numpy(dtype=float),
                ordered['signed_pe_z'].to_numpy(dtype=float)
            ])
        elif model_id == 'unsigned_surprise':
            x = np.column_stack([base, ordered['unsigned_surprise_z'].to_numpy(dtype=float)])
        elif model_id == 'within_loss_magnitude':
            x = np.column_stack([base, ordered['within_loss_magnitude_z'].to_numpy(dtype=float)])
        else:
            raise ValueError(model_id)
        rank = int(np.linalg.matrix_rank(x))
        pinv = np.linalg.pinv(x)
        global_models[model_id] = (x, pinv, rank)
        global_model_diagnostics[model_id] = {
            'rank': rank,
            'n_columns': int(x.shape[1]),
            'rank_deficient': bool(rank < x.shape[1]),
            'condition_number': float(np.linalg.cond(x))
        }
    correlation, vif, condition = condition_number_and_vif(
        ordered['valence_z'].to_numpy(dtype=float),
        ordered['signed_pe_z'].to_numpy(dtype=float)
    )
    estimator_estimability_summary = {}
    for estimator_id, diagnostic_rows in estimator_rank_diagnostics.items():
        estimable = np.asarray([bool(row['target_estimable']) for row in diagnostic_rows], dtype=bool)
        rank_deficient = np.asarray([bool(row['rank_deficient']) for row in diagnostic_rows], dtype=bool)
        residuals = np.asarray(
            [float(row['estimability_residual']) for row in diagnostic_rows],
            dtype=float
        )
        estimator_estimability_summary[estimator_id] = {
            'participant_estimable_fraction': float(estimable.mean()),
            'participant_rank_deficient_fraction': float(rank_deficient.mean()),
            'maximum_estimability_residual': float(residuals.max()),
            'rank_deficient_participant_count': int(rank_deficient.sum()),
            'nonestimable_participant_count': int((~estimable).sum())
        }
    return {
        'design_id': design_id,
        'frame': ordered,
        'participants': participants,
        'participant_slices': participant_slices,
        'stai': stai,
        'estimator_weights': estimator_weights,
        'estimator_rank_diagnostics': estimator_rank_diagnostics,
        'estimator_estimability_summary': estimator_estimability_summary,
        'global_models': global_models,
        'global_model_diagnostics': global_model_diagnostics,
        'predictor_correlation': correlation,
        'joint_vif': vif,
        'joint_condition_number': condition,
        'n_participants': len(participants),
        'n_trials': len(ordered)
    }


def estimate_participant_coefficients(design: dict[str, Any], outcomes: np.ndarray) -> dict[str, np.ndarray]:
    coefficients: dict[str, np.ndarray] = {}
    for estimator_id in ESTIMATORS:
        rows = []
        for participant_index, participant_slice in enumerate(design['participant_slices']):
            weights = design['estimator_weights'][estimator_id][participant_index]
            rows.append(weights @ outcomes[participant_slice, :])
        coefficients[estimator_id] = np.vstack(rows)
    return coefficients


def model_selection(design: dict[str, Any], outcomes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bic_rows: list[np.ndarray] = []
    bic_by_model: dict[str, np.ndarray] = {}
    n = outcomes.shape[0]
    for model_id in MODEL_IDS:
        x, pinv, k = design['global_models'][model_id]
        beta = pinv @ outcomes
        residual = outcomes - x @ beta
        sse = np.sum(residual ** 2, axis=0)
        bic = n * np.log(np.maximum(sse / n, EPS)) + k * np.log(n)
        bic_rows.append(bic)
        bic_by_model[model_id] = bic
    matrix = np.vstack(bic_rows)
    selected = np.argmin(matrix, axis=0)
    return (selected, bic_by_model)


def mechanism_signal(frame: pd.DataFrame, latent: np.ndarray, mechanism: str) -> np.ndarray:
    n_replicates = latent.shape[1]
    n_rows = len(frame)
    participant_codes = pd.Categorical(
        frame['participant_id'],
        categories=sorted(frame['participant_id'].unique()),
        ordered=True
    ).codes
    latent_rows = latent[participant_codes, :]
    valence = frame['valence_z'].to_numpy(dtype=float)[:, None]
    signed_pe = frame['signed_pe_z'].to_numpy(dtype=float)[:, None]
    surprise = frame['unsigned_surprise_z'].to_numpy(dtype=float)[:, None]
    loss = frame['within_loss_magnitude_z'].to_numpy(dtype=float)[:, None]
    if mechanism == 'null':
        return np.zeros((n_rows, n_replicates), dtype=float)
    if mechanism == 'valence':
        return latent_rows * valence
    if mechanism == 'signed_pe':
        return latent_rows * signed_pe
    if mechanism == 'unsigned_surprise':
        return latent_rows * surprise
    if mechanism == 'within_loss_magnitude':
        return latent_rows * loss
    if mechanism == 'mixed_valence_signed_pe':
        return latent_rows * (0.7 * valence + 0.7 * signed_pe)
    raise ValueError(mechanism)


def summarize_batch(
    design: dict[str, Any],
    mechanism: str,
    noise_sd: float,
    heterogeneity_sd: float,
    n_replicates: int,
    rng: np.random.Generator,
    module: str,
    scenario_metadata: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stai = design['stai']
    n_participants = len(stai)
    latent = BASE_EFFECT + SYMPTOM_SLOPE * stai[:, None] + heterogeneity_sd * rng.normal(size=(n_participants, n_replicates))
    if mechanism == 'null':
        latent = heterogeneity_sd * rng.normal(size=(n_participants, n_replicates))
    signal = mechanism_signal(design['frame'], latent, mechanism)
    noise_1 = noise_sd * rng.normal(size=signal.shape)
    noise_2 = noise_sd * rng.normal(size=signal.shape)
    y1 = within_participant_standardize_matrix(signal + noise_1, design['participant_slices'])
    y2 = within_participant_standardize_matrix(signal + noise_2, design['participant_slices'])
    coefficients_1 = estimate_participant_coefficients(design, y1)
    coefficients_2 = estimate_participant_coefficients(design, y2)
    selected_model_index, _ = model_selection(design, y1)
    selection_counts = Counter((MODEL_IDS[int(index)] for index in selected_model_index))
    true_model = TRUE_MODEL[mechanism]
    correct_selection_rate = selection_counts.get(true_model, 0) / n_replicates
    rows: list[dict[str, Any]] = []
    for estimator_id in ESTIMATORS:
        estimate_1 = coefficients_1[estimator_id]
        estimate_2 = coefficients_2[estimator_id]
        recovery = safe_corr_columns(latent, estimate_1)
        session_correlation = safe_corr_columns(estimate_1, estimate_2)
        reliability = spearman_brown_array(session_correlation)
        symptom_slope, _, symptom_p = slope_test_matrix(stai, estimate_1)
        detection = symptom_p < 0.05
        expected_positive = mechanism != 'null' and estimator_id != 'null'
        sign_accuracy = symptom_slope > 0 if expected_positive else np.abs(symptom_slope) < np.inf
        estimator_diagnostics = design['estimator_estimability_summary'][estimator_id]
        true_model_diagnostics = design['global_model_diagnostics'][true_model]
        row = {
            'module': module,
            'design_id': design['design_id'],
            'mechanism': mechanism,
            'estimator_id': estimator_id,
            'noise_sd': noise_sd,
            'heterogeneity_sd': heterogeneity_sd,
            'n_replicates': n_replicates,
            'n_participants': design['n_participants'],
            'n_trials': design['n_trials'],
            'predictor_correlation': design['predictor_correlation'],
            'joint_vif': design['joint_vif'],
            'joint_condition_number': design['joint_condition_number'],
            'recovery_mean': float(np.nanmean(recovery)),
            'recovery_median': float(np.nanmedian(recovery)),
            'recovery_q025': float(np.nanquantile(recovery, 0.025)),
            'recovery_q975': float(np.nanquantile(recovery, 0.975)),
            'reliability_median': float(np.nanmedian(reliability)),
            'reliability_q025': float(np.nanquantile(reliability, 0.025)),
            'reliability_q975': float(np.nanquantile(reliability, 0.975)),
            'symptom_slope_mean': float(np.mean(symptom_slope)),
            'symptom_detection_rate': float(np.mean(detection)),
            'positive_slope_rate': float(np.mean(sign_accuracy)),
            'correct_model_selection_rate': float(correct_selection_rate),
            'participant_estimable_fraction': float(estimator_diagnostics['participant_estimable_fraction']),
            'participant_rank_deficient_fraction': float(estimator_diagnostics['participant_rank_deficient_fraction']),
            'nonestimable_participant_count': int(estimator_diagnostics['nonestimable_participant_count']),
            'global_true_model_rank': int(true_model_diagnostics['rank']),
            'global_true_model_columns': int(true_model_diagnostics['n_columns']),
            'global_true_model_rank_deficient': bool(true_model_diagnostics['rank_deficient']),
            'true_model': true_model,
            'is_primary_estimator': estimator_id == PRIMARY_ESTIMATOR[mechanism],
            **scenario_metadata
        }
        rows.append(row)
    selection_summary = {
        'module': module,
        'design_id': design['design_id'],
        'mechanism': mechanism,
        'noise_sd': noise_sd,
        'heterogeneity_sd': heterogeneity_sd,
        'n_replicates': n_replicates,
        'n_participants': design['n_participants'],
        'n_trials': design['n_trials'],
        'predictor_correlation': design['predictor_correlation'],
        'joint_vif': design['joint_vif'],
        'joint_condition_number': design['joint_condition_number'],
        'true_model': true_model,
        'correct_model_selection_rate': float(correct_selection_rate),
        'true_model_global_rank': int(design['global_model_diagnostics'][true_model]['rank']),
        'true_model_global_columns': int(design['global_model_diagnostics'][true_model]['n_columns']),
        'true_model_global_rank_deficient': bool(design['global_model_diagnostics'][true_model]['rank_deficient']),
        **{f'selection_rate__{model_id}': selection_counts.get(model_id, 0) / n_replicates for model_id in MODEL_IDS},
        **scenario_metadata
    }
    return (rows, selection_summary)


def aggregate_simulation_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric_average = [
        'recovery_mean',
        'recovery_median',
        'reliability_median',
        'symptom_slope_mean',
        'symptom_detection_rate',
        'positive_slope_rate',
        'correct_model_selection_rate',
        'participant_estimable_fraction',
        'participant_rank_deficient_fraction',
        'nonestimable_participant_count',
        'global_true_model_rank',
        'global_true_model_columns',
        'predictor_correlation',
        'joint_vif',
        'joint_condition_number',
        'n_participants',
        'n_trials'
    ]
    available = [column for column in numeric_average if column in frame.columns]
    result = frame.groupby(group_columns, dropna=False, sort=True)[available].mean().reset_index()
    counts = frame.groupby(group_columns, dropna=False).size().rename('batch_count').reset_index()
    return result.merge(counts, on=group_columns, validate='one_to_one')


def empirical_design_variants(primary: pd.DataFrame, rng: np.random.Generator) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    variants: list[tuple[pd.DataFrame, dict[str, Any]]] = [(
        primary.copy(),
        {
            'design_family': 'empirical_exact',
            'count_per_valence': 'all',
            'cohort_rule': 'accepted_N46',
            'design_draw': 0
        }
    )]
    counts = primary.groupby(['participant_id', 'feedback_negative']).size().unstack(fill_value=0)
    counts['minimum_valence_count'] = counts.min(axis=1)
    fixed_definitions = [('fixed_min40', 40, [20, 40]), ('fixed_min60', 60, [20, 40, 60])]
    for cohort_rule, minimum, target_counts in fixed_definitions:
        participants = sorted(counts.index[counts['minimum_valence_count'] >= minimum].tolist())
        cohort = primary.loc[primary['participant_id'].isin(participants)].copy()
        for target_count in target_counts:
            for draw in range(EMPIRICAL_COUNT_DRAWS):
                sampled_parts = []
                for participant_id, part in cohort.groupby('participant_id', sort=True):
                    for feedback_value, valence_part in part.groupby('feedback_negative', sort=True):
                        selected = rng.choice(
                            valence_part.index.to_numpy(),
                            size=target_count,
                            replace=False
                        )
                        sampled_parts.append(valence_part.loc[selected])
                sampled = pd.concat(sampled_parts, ignore_index=True).sort_values(['participant_id', 'trial_number_phase'])
                variants.append((
                    sampled.reset_index(drop=True),
                    {
                        'design_family': 'empirical_fixed_count',
                        'count_per_valence': target_count,
                        'cohort_rule': cohort_rule,
                        'design_draw': draw
                    }
                ))
    return variants


def generic_scenario_library() -> list[dict[str, Any]]:
    baseline = {
        'scenario_id': 'baseline',
        'probability_good': 0.75,
        'trials_per_valence': 60,
        'learning_rate': 0.3,
        'asymmetry': 'symmetric',
        'structure': 'stationary',
        'outcome_coding': 'absolute',
        'noise_sd': 1.25,
        'heterogeneity_sd': 0.35
    }
    scenarios = [baseline.copy()]

    def modified(scenario_id: str, **updates: Any) -> None:
        scenario = baseline.copy()
        scenario['scenario_id'] = scenario_id
        scenario.update(updates)
        scenarios.append(scenario)
    modified('probability_060', probability_good=0.6)
    modified('probability_090', probability_good=0.9)
    modified('trials_020', trials_per_valence=20)
    modified('trials_040', trials_per_valence=40)
    modified('trials_100', trials_per_valence=100)
    modified('learning_rate_010', learning_rate=0.1)
    modified('learning_rate_060', learning_rate=0.6)
    modified('positive_fast', asymmetry='positive_fast')
    modified('negative_fast', asymmetry='negative_fast')
    modified('single_reversal', structure='single_reversal')
    modified('volatile', structure='volatile')
    modified('context_normalized_outcomes', outcome_coding='context_normalized')
    modified('low_noise', noise_sd=0.75)
    modified('high_noise', noise_sd=2.0)
    modified('low_heterogeneity', heterogeneity_sd=0.1)
    modified('high_heterogeneity', heterogeneity_sd=0.7)
    modified('challenge_low_trials_high_noise', trials_per_valence=20, noise_sd=2.0)
    modified('challenge_low_probability_high_noise', probability_good=0.6, noise_sd=2.0)
    modified('challenge_reversal_low_trials', structure='single_reversal', trials_per_valence=20)
    modified('challenge_normalized_high_noise', outcome_coding='context_normalized', noise_sd=2.0)
    modified('challenge_volatile_positive_fast', structure='volatile', asymmetry='positive_fast')
    modified('favorable_high_trials_low_noise', trials_per_valence=100, noise_sd=0.75)
    return scenarios


def alpha_pair(learning_rate: float, asymmetry: str) -> tuple[float, float]:
    if asymmetry == 'symmetric':
        return (learning_rate, learning_rate)
    if asymmetry == 'positive_fast':
        return (min(0.85, learning_rate + 0.35), max(0.05, learning_rate - 0.1))
    if asymmetry == 'negative_fast':
        return (max(0.05, learning_rate - 0.1), min(0.85, learning_rate + 0.35))
    raise ValueError(asymmetry)


def probability_schedule(n_trials: int, probability_good: float, structure: str) -> np.ndarray:
    if structure == 'stationary':
        return np.full(n_trials, probability_good, dtype=float)
    if structure == 'single_reversal':
        schedule = np.full(n_trials, probability_good, dtype=float)
        schedule[n_trials // 2:] = 1.0 - probability_good
        return schedule
    if structure == 'volatile':
        schedule = np.empty(n_trials, dtype=float)
        edges = np.linspace(0, n_trials, 5, dtype=int)
        for block in range(4):
            value = probability_good if block % 2 == 0 else 1.0 - probability_good
            schedule[edges[block]:edges[block + 1]] = value
        return schedule
    raise ValueError(structure)


def simulate_generic_design(
    scenario: dict[str, Any],
    stai: np.ndarray,
    participant_ids: list[str],
    rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_per_valence = int(scenario['trials_per_valence'])
    alpha_positive, alpha_negative = alpha_pair(
        float(scenario['learning_rate']),
        str(scenario['asymmetry'])
    )
    for participant_index, participant_id in enumerate(participant_ids):
        participant_alpha_shift = rng.normal(loc=0.0, scale=0.04)
        alpha_pos = float(np.clip(alpha_positive + participant_alpha_shift, 0.02, 0.95))
        alpha_neg = float(np.clip(alpha_negative + participant_alpha_shift, 0.02, 0.95))
        participant_rows = []
        for feedback_negative in [0, 1]:
            schedule = probability_schedule(
                n_per_valence,
                float(scenario['probability_good']),
                str(scenario['structure'])
            )
            if scenario['outcome_coding'] == 'absolute':
                q_value = -0.5 if feedback_negative else 0.5
            else:
                q_value = 0.0
            for trial_within_valence in range(n_per_valence):
                good = bool(rng.random() < schedule[trial_within_valence])
                if scenario['outcome_coding'] == 'absolute':
                    if feedback_negative:
                        outcome = 0.0 if good else -1.0
                    else:
                        outcome = 1.0 if good else 0.0
                elif scenario['outcome_coding'] == 'context_normalized':
                    outcome = 1.0 if good else -1.0
                else:
                    raise ValueError(scenario['outcome_coding'])
                prediction_error = outcome - q_value
                learning_rate = alpha_pos if prediction_error >= 0 else alpha_neg
                q_value = q_value + learning_rate * prediction_error
                participant_rows.append({
                    'participant_id': participant_id,
                    'stai_z': float(stai[participant_index]),
                    'feedback_negative': feedback_negative,
                    'signed_pe_raw': prediction_error,
                    'trial_within_valence': trial_within_valence
                })
        participant_frame = pd.DataFrame(participant_rows)
        participant_frame = participant_frame.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2 ** 31 - 1))
        ).reset_index(drop=True)
        participant_frame['progression_z'] = zscore(np.arange(len(participant_frame), dtype=float))
        participant_frame['valence_z'] = zscore(participant_frame['feedback_negative'].to_numpy(dtype=float))
        participant_frame['signed_pe_z'] = zscore(participant_frame['signed_pe_raw'].to_numpy(dtype=float))
        participant_frame['unsigned_surprise_z'] = zscore(np.abs(participant_frame['signed_pe_raw'].to_numpy(dtype=float)))
        loss_magnitude = participant_frame['feedback_negative'].to_numpy(dtype=float) * np.abs(participant_frame['signed_pe_raw'].to_numpy(dtype=float))
        participant_frame['within_loss_magnitude_z'] = zscore(loss_magnitude)
        participant_frame['rt_z'] = zscore(0.2 * participant_frame['unsigned_surprise_z'].to_numpy(dtype=float) + rng.normal(size=len(participant_frame)))
        rows.append(participant_frame)
    return pd.concat(rows, ignore_index=True)


def spectral_module(stai: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_participants = len(stai)
    for profile_id, weights in SPECTRAL_PROFILES.items():
        for count in SPECTRAL_COUNTS:
            for noise_id, base_noise in SPECTRAL_NOISE.items():
                latent = BASE_EFFECT + SYMPTOM_SLOPE * stai[:, None] + DEFAULT_HETEROGENEITY * rng.normal(size=(n_participants, SPECTRAL_REPLICATES))
                count_scale = math.sqrt(20.0 / count)
                session_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for band in ['delta', 'theta', 'alpha']:
                    signal = weights[band] * latent
                    noise_sd = base_noise * count_scale
                    first = signal + noise_sd * rng.normal(size=signal.shape)
                    second = signal + noise_sd * rng.normal(size=signal.shape)
                    session_values[band] = (first, second)
                slopes: dict[str, np.ndarray] = {}
                p_values: dict[str, np.ndarray] = {}
                reliabilities: dict[str, np.ndarray] = {}
                for band in ['delta', 'theta', 'alpha']:
                    first, second = session_values[band]
                    slope, _, p_value = slope_test_matrix(stai, first)
                    slopes[band] = slope
                    p_values[band] = p_value
                    reliabilities[band] = spearman_brown_array(safe_corr_columns(first, second))
                theta_minus_delta = session_values['theta'][0] - session_values['delta'][0]
                theta_minus_alpha = session_values['theta'][0] - session_values['alpha'][0]
                theta_delta_slope, _, theta_delta_p = slope_test_matrix(stai, theta_minus_delta)
                theta_alpha_slope, _, theta_alpha_p = slope_test_matrix(stai, theta_minus_alpha)
                theta_detected = p_values['theta'] < 0.05
                delta_detected = p_values['delta'] < 0.05
                alpha_detected = p_values['alpha'] < 0.05
                theta_specific_supported = theta_detected & (theta_delta_p < 0.05) & (theta_alpha_p < 0.05) & (theta_delta_slope > 0) & (theta_alpha_slope > 0)
                apparent_theta_without_specificity = theta_detected & ~theta_specific_supported
                rows.append({
                    'profile_id': profile_id,
                    'trials_per_valence': count,
                    'noise_id': noise_id,
                    'noise_sd_at_count': base_noise * count_scale,
                    'n_replicates': SPECTRAL_REPLICATES,
                    'theta_detection_rate': float(theta_detected.mean()),
                    'delta_detection_rate': float(delta_detected.mean()),
                    'alpha_detection_rate': float(alpha_detected.mean()),
                    'theta_specificity_support_rate': float(theta_specific_supported.mean()),
                    'apparent_theta_without_specificity_rate': float(apparent_theta_without_specificity.mean()),
                    'theta_reliability_median': float(np.nanmedian(reliabilities['theta'])),
                    'delta_reliability_median': float(np.nanmedian(reliabilities['delta'])),
                    'alpha_reliability_median': float(np.nanmedian(reliabilities['alpha'])),
                    'theta_slope_mean': float(slopes['theta'].mean()),
                    'theta_minus_delta_slope_mean': float(theta_delta_slope.mean()),
                    'theta_minus_alpha_slope_mean': float(theta_alpha_slope.mean())
                })
    return pd.DataFrame(rows)


def classify_identification(
    primary_row: pd.Series,
    max_false_attribution: float,
    model_selection_rate: float
) -> str:
    thresholds = IDENTIFICATION_THRESHOLDS
    recovery = float(primary_row['recovery_mean'])
    detection = float(primary_row['symptom_detection_rate'])
    estimable_fraction = float(primary_row.get('participant_estimable_fraction', 1.0))
    if recovery >= thresholds['recovery_identified'] and detection >= thresholds['detection_identified'] and (model_selection_rate >= thresholds['model_selection_identified']) and (max_false_attribution <= thresholds['false_attribution_identified']) and (estimable_fraction >= 0.95):
        return 'identified'
    if recovery >= thresholds['recovery_ambiguous'] or detection >= 0.4:
        return 'ambiguous'
    return 'poor_recovery'


def build_identification_map(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_columns = group_columns + ['mechanism']
    for keys, mechanism_frame in summary.groupby(key_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_data = dict(zip(key_columns, keys))
        mechanism = str(key_data['mechanism'])
        primary_id = PRIMARY_ESTIMATOR[mechanism]
        primary_rows = mechanism_frame.loc[mechanism_frame['estimator_id'].astype(str).eq(primary_id)]
        if len(primary_rows) != 1:
            raise RuntimeError(f'Identification map expected one primary row: {key_data}')
        primary_row = primary_rows.iloc[0]
        other = mechanism_frame.loc[~mechanism_frame['estimator_id'].astype(str).isin([primary_id, 'null'])]
        max_false = float(other['symptom_detection_rate'].max()) if not other.empty else 0.0
        selection_mask = np.ones(len(selection), dtype=bool)
        for column, value in key_data.items():
            if column not in selection.columns:
                continue
            selection_mask &= selection[column].astype(str) == str(value)
        selected_rows = selection.loc[selection_mask]
        if len(selected_rows) != 1:
            raise RuntimeError(f'Identification map selection alignment failed: {key_data}, rows={len(selected_rows)}')
        selection_rate = float(selected_rows['correct_model_selection_rate'].iloc[0])
        status = classify_identification(primary_row, max_false, selection_rate)
        rows.append({
            **key_data,
            'primary_estimator': primary_id,
            'primary_recovery': float(primary_row['recovery_mean']),
            'primary_detection_rate': float(primary_row['symptom_detection_rate']),
            'primary_reliability': float(primary_row['reliability_median']),
            'primary_participant_estimable_fraction': float(primary_row.get('participant_estimable_fraction', 1.0)),
            'primary_participant_rank_deficient_fraction': float(primary_row.get('participant_rank_deficient_fraction', 0.0)),
            'maximum_competing_detection_rate': max_false,
            'correct_model_selection_rate': selection_rate,
            'identification_status': status
        })
    return pd.DataFrame(rows)
