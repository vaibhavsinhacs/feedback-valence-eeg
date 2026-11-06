"""Spatial, temporal, and frequency-domain EEG comparisons.

Methods for channel-region contrasts, time-frequency summaries, reliability,
and the time-domain feedback potential. The full dataset runner and
results-archive code are not part of this selection.
"""

from __future__ import annotations

from typing import Any, Iterable
from pathlib import Path
import math

import mne
import numpy as np
import pandas as pd
from scipy import stats
from mne.time_frequency import tfr_array_morlet


FREQUENCIES = np.logspace(0.01, 1.7, 50)
N_CYCLES = np.logspace(np.log10(3.0), np.log10(10.0), 50)
THETA_MASK = (FREQUENCIES >= 4.0) & (FREQUENCIES <= 8.0)
TF_MASK = FREQUENCIES <= 20.0
THETA_FREQS, THETA_CYCLES = FREQUENCIES[THETA_MASK], N_CYCLES[THETA_MASK]
TF_FREQS, TF_CYCLES = FREQUENCIES[TF_MASK], N_CYCLES[TF_MASK]
BASELINE_WINDOW, PRIMARY_WINDOW = (-0.30, -0.20), (0.20, 0.40)
TEMPORAL_WINDOWS = {
    "early_000_200": (0.0, 0.2),
    "primary_200_400": (0.2, 0.4),
    "late_400_600": (0.4, 0.6)
}
SPECTRAL_BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_12": (8.0, 12.0),
    "beta_13_20": (13.0, 20.0)
}
TF_TIME_EDGES = np.arange(0.0, 0.6000001, 0.05)
FRONTOCENTRAL_CORE_REQUESTED = ["FZ", "FCZ", "CZ", "FC1", "FC2", "C1", "C2"]
MIDLINE_FRONTOCENTRAL_REQUESTED = ["FZ", "FCZ", "CZ"]
POSTERIOR_CONTROL_REQUESTED = ["P3", "P4", "P7", "P8", "PO3", "PO4", "O1", "O2"]
FRONTOPOLAR_REQUESTED = ["FP1", "FPZ", "FP2", "AF3", "AF4"]
BOOTSTRAP_REPLICATES = 9999
PERMUTATION_REPLICATES = 9999
BOOTSTRAP_SEED, PERMUTATION_SEED, RELIABILITY_SEED = 2026072903, 2026072904, 2026072905
THETA_TOPO_DECIMATION, TF_DECIMATION = 4, 4
FLOAT_TOLERANCE, PRIMARY_REPRO_TOLERANCE = 1e-10, 1e-9


def normalize_participant_id(value: Any) -> str:
    text = str(value).strip()
    lowered = text.lower()
    suffix = lowered[4:] if lowered.startswith('sub-') else lowered
    return f'sub-{int(suffix):03d}' if suffix.isdigit() else text


def assert_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {', '.join(missing)}")


def standardize_within_participant(frame: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    grouped = frame.groupby('participant_id', sort=False)[source]
    means = grouped.transform('mean')
    sds = grouped.transform(lambda values: values.std(ddof=1))
    if not np.isfinite(sds).all() or (sds <= 0).any():
        raise RuntimeError(f'Invalid participant SD for {source}.')
    frame[destination] = (frame[source] - means) / sds
    return frame


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def available_channels(requested: list[str], available: list[str]) -> list[str]:
    available_set = set(available)
    return [channel for channel in requested if channel in available_set]


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, int]:
    required = [
        'participant_id',
        'feedback_negative',
        'stai_z_all_complete',
        'response_time_within_participant_z',
        'trial_progression_within_participant_z',
        'positive_pe_within_participant_valence_centered',
        'negative_pe_within_participant_valence_centered'
    ]
    assert_columns(frame, required, 'primary model table')
    participant_ids = sorted(frame['participant_id'].astype(str).unique())
    participant = frame['participant_id'].astype(str)
    columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    names = ['Intercept']
    for participant_id in participant_ids[1:]:
        columns.append(participant.eq(participant_id).to_numpy(dtype=float))
        names.append(f'participant[{participant_id}]')
    feedback = pd.to_numeric(frame['feedback_negative'], errors='raise').to_numpy(dtype=float)
    stai = pd.to_numeric(frame['stai_z_all_complete'], errors='raise').to_numpy(dtype=float)
    rt = pd.to_numeric(frame['response_time_within_participant_z'], errors='raise').to_numpy(dtype=float)
    progression = pd.to_numeric(
        frame['trial_progression_within_participant_z'],
        errors='raise'
    ).to_numpy(dtype=float)
    positive_pe = pd.to_numeric(
        frame['positive_pe_within_participant_valence_centered'],
        errors='raise'
    ).to_numpy(dtype=float)
    negative_pe = pd.to_numeric(
        frame['negative_pe_within_participant_valence_centered'],
        errors='raise'
    ).to_numpy(dtype=float)
    additional = [
        ('feedback_negative', feedback),
        ('feedback_negative:stai_z_all_complete', feedback * stai),
        ('response_time_within_participant_z', rt),
        ('trial_progression_within_participant_z', progression),
        ('positive_pe_within_participant_valence_centered', positive_pe),
        ('positive_pe_within_participant_valence_centered:stai_z_all_complete', positive_pe * stai),
        ('negative_pe_within_participant_valence_centered', negative_pe),
        ('negative_pe_within_participant_valence_centered:stai_z_all_complete', negative_pe * stai)
    ]
    for name, values in additional:
        columns.append(values)
        names.append(name)
    matrix = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise RuntimeError(f'Primary design is rank deficient: rank={rank}, columns={matrix.shape[1]}.')
    cluster_codes = pd.Categorical(
        frame['participant_id'].astype(str),
        categories=participant_ids,
        ordered=True
    ).codes
    target_index = names.index('feedback_negative:stai_z_all_complete')
    return (matrix, names, cluster_codes, target_index)


class CR1TargetFitter:

    def __init__(self, matrix: np.ndarray, cluster_codes: np.ndarray, target_index: int) -> None:
        self.x = np.asarray(matrix, dtype=float)
        self.cluster_codes = np.asarray(cluster_codes, dtype=int)
        self.target_index = int(target_index)
        self.n, self.k = self.x.shape
        self.groups = np.unique(self.cluster_codes)
        self.g = len(self.groups)
        self.xtx_inv = np.linalg.inv(self.x.T @ self.x)
        self.target_bread = self.xtx_inv[self.target_index, :]
        self.factor = self.g / (self.g - 1.0) * (self.n - 1.0) / (self.n - self.k)
        self.df = self.g - 1

    def fit_matrix(self, outcomes: np.ndarray, labels: list[str]) -> pd.DataFrame:
        y = np.asarray(outcomes, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[0] != self.n:
            raise RuntimeError(f'Outcome rows {y.shape[0]} do not match design {self.n}.')
        if y.shape[1] != len(labels):
            raise RuntimeError('Outcome label count mismatch.')
        if not np.isfinite(y).all():
            raise RuntimeError('Outcome matrix contains nonfinite values.')
        beta = self.xtx_inv @ (self.x.T @ y)
        residual = y - self.x @ beta
        target_variance = np.zeros(y.shape[1], dtype=float)
        for group in self.groups:
            mask = self.cluster_codes == group
            score = self.x[mask, :].T @ residual[mask, :]
            projected = self.target_bread @ score
            target_variance += projected ** 2
        target_variance *= self.factor
        se = np.sqrt(np.maximum(target_variance, 0.0))
        estimate = beta[self.target_index, :]
        t_stat = estimate / se
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=self.df)
        critical = stats.t.ppf(0.975, df=self.df)
        ci_low = estimate - critical * se
        ci_high = estimate + critical * se
        return pd.DataFrame({
            'outcome_id': labels,
            'estimate': estimate,
            'CR1_SE': se,
            't_statistic': t_stat,
            'df': self.df,
            'p_value': p_value,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_rows': self.n,
            'n_clusters': self.g
        })


def participant_gain(
    frame: pd.DataFrame,
    outcome: str,
    subset_column: str | None=None,
    subset_value: str | None=None
) -> pd.DataFrame:
    required = [
        'participant_id',
        'feedback_negative',
        'response_time_within_participant_z',
        'trial_progression_within_participant_z',
        outcome
    ]
    assert_columns(frame, required, f'participant gain {outcome}')
    use = frame.copy()
    if subset_column is not None:
        use = use.loc[use[subset_column].astype(str).eq(str(subset_value))].copy()
    rows: list[dict[str, Any]] = []
    for participant_id, part in use.groupby('participant_id', sort=True):
        y = pd.to_numeric(part[outcome], errors='raise').to_numpy(dtype=float)
        x = np.column_stack([
            np.ones(len(part)),
            pd.to_numeric(part['feedback_negative'], errors='raise').to_numpy(dtype=float),
            pd.to_numeric(
                part['response_time_within_participant_z'],
                errors='raise'
            ).to_numpy(dtype=float),
            pd.to_numeric(
                part['trial_progression_within_participant_z'],
                errors='raise'
            ).to_numpy(dtype=float)
        ])
        if np.linalg.matrix_rank(x) != x.shape[1]:
            raise RuntimeError(f'{participant_id}: participant gain model rank deficient.')
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        rows.append({
            'participant_id': participant_id,
            'outcome_id': outcome,
            'positive_intercept': float(beta[0]),
            'negative_minus_positive_gain': float(beta[1]),
            'n_trials': len(part)
        })
    return pd.DataFrame(rows)


def spearman_test(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_perm: int=PERMUTATION_REPLICATES,
    n_boot: int=BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 4:
        raise RuntimeError('Invalid Spearman inputs.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError('Nonfinite Spearman inputs.')
    observed = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm, dtype=float)
    for index in range(n_perm):
        perm[index] = stats.spearmanr(x[rng.permutation(len(x))], y).statistic
    p_value = float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1))
    boot = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, len(x), size=len(x))
        value = stats.spearmanr(x[sample], y[sample]).statistic
        boot[index] = value
    finite = boot[np.isfinite(boot)]
    ci_low, ci_high = np.quantile(finite, [0.025, 0.975])
    return {
        'rho': observed,
        'permutation_p': p_value,
        'bootstrap_ci_low': float(ci_low),
        'bootstrap_ci_high': float(ci_high),
        'n': len(x),
        'n_perm': n_perm,
        'n_boot': n_boot
    }


def partial_spearman_test(
    x: np.ndarray,
    y: np.ndarray,
    control: np.ndarray,
    seed: int
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    control = np.asarray(control, dtype=float)
    if not len(x) == len(y) == len(control):
        raise RuntimeError('Partial Spearman input length mismatch.')

    def residual_rho(x_values: np.ndarray, y_values: np.ndarray, c_values: np.ndarray) -> float:
        rx = stats.rankdata(x_values)
        ry = stats.rankdata(y_values)
        rc = stats.rankdata(c_values)
        design = np.column_stack([np.ones(len(rc)), rc])
        ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
        ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
        return float(stats.pearsonr(ex, ey).statistic)
    observed = residual_rho(x, y, control)
    rng = np.random.default_rng(seed)
    perm = np.empty(PERMUTATION_REPLICATES, dtype=float)
    for index in range(PERMUTATION_REPLICATES):
        perm[index] = residual_rho(x[rng.permutation(len(x))], y, control)
    p_value = float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (PERMUTATION_REPLICATES + 1))
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(x), size=len(x))
        boot[index] = residual_rho(x[sample], y[sample], control[sample])
    finite = boot[np.isfinite(boot)]
    ci_low, ci_high = np.quantile(finite, [0.025, 0.975])
    return {
        'partial_rho': observed,
        'permutation_p': p_value,
        'bootstrap_ci_low': float(ci_low),
        'bootstrap_ci_high': float(ci_high),
        'n': len(x)
    }


def spearman_brown(r: float) -> float:
    if not np.isfinite(r) or math.isclose(1.0 + r, 0.0):
        return np.nan
    return float(2.0 * r / (1.0 + r))


def reliability_from_halves(
    marker: pd.DataFrame,
    half_column: str,
    halves: tuple[str, str],
    seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    wide = marker.pivot(
        index='participant_id',
        columns=half_column,
        values='negative_minus_positive_gain'
    )
    wide = wide.loc[:, list(halves)].dropna()
    x = wide[halves[0]].to_numpy(dtype=float)
    y = wide[halves[1]].to_numpy(dtype=float)
    r = float(stats.pearsonr(x, y).statistic)
    sb = spearman_brown(r)
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(x), size=len(x))
        value = stats.pearsonr(x[sample], y[sample]).statistic
        boot[index] = spearman_brown(float(value))
    finite = boot[np.isfinite(boot)]
    ci_low, ci_high = np.quantile(finite, [0.025, 0.975])
    return (
        {
            'pearson_r': r,
            'spearman_brown': sb,
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'n_participants': len(x)
        },
        boot
    )


def pooled_db(power: np.ndarray, times: np.ndarray) -> np.ndarray:
    baseline_mask = (times >= BASELINE_WINDOW[0]) & (times <= BASELINE_WINDOW[1])
    if not baseline_mask.any():
        raise RuntimeError('Baseline window has no samples.')
    baseline = power[..., baseline_mask].mean(axis=(0, -1))
    if not np.isfinite(baseline).all() or (baseline <= 0).any():
        raise RuntimeError('Invalid pooled baseline.')
    return 10.0 * np.log10(power / baseline[None, ..., None])


def fit_morlet_cycles_to_signal(
    frequencies: np.ndarray,
    requested_cycles: np.ndarray,
    n_times: int,
    sfreq: float,
    safety_factor: float=0.95
) -> np.ndarray:
    """
    Cap low-frequency Morlet cycles so every wavelet is shorter than
    the available epoch. MNE's zero-mean Morlet length is approximately
    10 * sigma_t, with sigma_t = n_cycles / (2*pi*f).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    requested_cycles = np.asarray(requested_cycles, dtype=float)
    epoch_duration = (int(n_times) - 1) / float(sfreq)
    maximum_cycles = safety_factor * epoch_duration * 2.0 * np.pi * frequencies / 10.0
    effective_cycles = np.minimum(requested_cycles, maximum_cycles)
    if not np.isfinite(effective_cycles).all():
        raise RuntimeError('Nonfinite effective Morlet cycles.')
    if (effective_cycles <= 0).any():
        raise RuntimeError('Nonpositive effective Morlet cycles.')
    return effective_cycles


def extract_n46_tf_features(
    participants: list[str],
    primary_trials: pd.DataFrame,
    epoch_inventory: pd.DataFrame,
    metadata_inventory: pd.DataFrame,
    checkpoint_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, mne.Info, list[str], dict[str, list[str]]]:
    epoch_map = {normalize_participant_id(row.participant_id): Path(row.path) for row in epoch_inventory.itertuples(index=False)}
    metadata_map = {normalize_participant_id(row.participant_id): Path(row.path) for row in metadata_inventory.itertuples(index=False)}
    indexed_trials = primary_trials.set_index(['participant_id', 'trial_number_phase'])
    if not indexed_trials.index.is_unique:
        raise RuntimeError('Primary trial keys are not unique for S4 extraction.')
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    feature_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    canonical_info: mne.Info | None = None
    canonical_channels: list[str] | None = None
    channel_sets: dict[str, list[str]] | None = None
    required_metadata = [
        'participant_id',
        'trial_number_phase',
        'feedback_valence',
        'feedback_negative',
        'event_sample',
        'trial_id'
    ]
    print('Extracting N=46 theta topography and frontocentral time-frequency features...')
    for index, participant_id in enumerate(participants, start=1):
        epochs = mne.read_epochs(epoch_map[participant_id], preload=True, verbose='ERROR')
        csv_metadata = pd.read_csv(metadata_map[participant_id], low_memory=False)
        if epochs.metadata is None:
            raise RuntimeError(f'{participant_id}: missing FIF metadata.')
        epoch_metadata = epochs.metadata.reset_index(drop=True).copy()
        assert_columns(epoch_metadata, required_metadata, f'{participant_id} metadata')
        assert_columns(csv_metadata, required_metadata, f'{participant_id} CSV metadata')
        if len(epochs) != len(epoch_metadata) or len(epoch_metadata) != len(csv_metadata):
            raise RuntimeError(f'{participant_id}: epoch/metadata length mismatch.')
        for frame in [epoch_metadata, csv_metadata]:
            frame['participant_id'] = frame['participant_id'].map(normalize_participant_id)
            frame['trial_number_phase'] = pd.to_numeric(frame['trial_number_phase'], errors='raise').astype(int)
        comparison = [
            'participant_id',
            'trial_number_phase',
            'feedback_valence',
            'feedback_negative',
            'event_sample',
            'trial_id'
        ]
        if not epoch_metadata[comparison].astype(str).equals(csv_metadata[comparison].astype(str)):
            raise RuntimeError(f'{participant_id}: FIF/CSV metadata mismatch.')
        metadata_index = epoch_metadata.set_index(['participant_id', 'trial_number_phase'])
        if not metadata_index.index.is_unique:
            raise RuntimeError(f'{participant_id}: metadata keys are not unique.')
        aligned = indexed_trials.reindex(metadata_index.index)
        if aligned['trial_id'].astype(str).tolist() != metadata_index['trial_id'].astype(str).tolist():
            raise RuntimeError(f'{participant_id}: primary trial alignment mismatch.')
        eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, stim=False, exclude=[])
        eeg_channels = [epochs.ch_names[pick] for pick in eeg_picks]
        if canonical_channels is None:
            canonical_channels = eeg_channels
            canonical_info = epochs.copy().pick(eeg_channels).info
            channel_sets = {
                'frontocentral_core': available_channels(FRONTOCENTRAL_CORE_REQUESTED, eeg_channels),
                'midline_frontocentral': available_channels(MIDLINE_FRONTOCENTRAL_REQUESTED, eeg_channels),
                'posterior_control': available_channels(POSTERIOR_CONTROL_REQUESTED, eeg_channels),
                'frontopolar': available_channels(FRONTOPOLAR_REQUESTED, eeg_channels)
            }
            if 'FCZ' not in eeg_channels:
                raise RuntimeError('FCZ is absent.')
            if len(channel_sets['frontocentral_core']) < 5:
                raise RuntimeError('Fewer than five frontocentral-core channels.')
            if len(channel_sets['posterior_control']) < 4:
                raise RuntimeError('Fewer than four posterior-control channels.')
        elif eeg_channels != canonical_channels:
            raise RuntimeError(f'{participant_id}: EEG channel order differs from canonical order.')
        assert canonical_channels is not None
        assert channel_sets is not None
        times = epochs.times.astype(float)
        if not math.isclose(float(epochs.info['sfreq']), 500.0, abs_tol=1e-12):
            raise RuntimeError(f'{participant_id}: sampling rate is not 500 Hz.')
        raw = epochs.get_data(picks=canonical_channels, copy=True)
        fcz_index = canonical_channels.index('FCZ')
        exact_fcz_power = tfr_array_morlet(
            raw[:, fcz_index:fcz_index + 1, :],
            sfreq=float(epochs.info['sfreq']),
            freqs=THETA_FREQS,
            n_cycles=THETA_CYCLES,
            zero_mean=True,
            use_fft=True,
            decim=1,
            output='power',
            n_jobs=1,
            verbose=False
        )
        exact_fcz_db = pooled_db(exact_fcz_power, times)
        exact_roi_mask = (times >= PRIMARY_WINDOW[0]) & (times <= PRIMARY_WINDOW[1])
        exact_fcz_feature = exact_fcz_db[:, 0, :, :][:, :, exact_roi_mask].mean(axis=(1, 2))
        del exact_fcz_power, exact_fcz_db
        theta_topography_power = tfr_array_morlet(
            raw,
            sfreq=float(epochs.info['sfreq']),
            freqs=THETA_FREQS,
            n_cycles=THETA_CYCLES,
            zero_mean=True,
            use_fft=True,
            decim=THETA_TOPO_DECIMATION,
            output='power',
            n_jobs=1,
            verbose=False
        )
        theta_topography_times = times[::THETA_TOPO_DECIMATION]
        theta_topography_db = pooled_db(theta_topography_power, theta_topography_times)
        topography_roi_mask = (theta_topography_times >= PRIMARY_WINDOW[0]) & (theta_topography_times <= PRIMARY_WINDOW[1])
        topography = theta_topography_db[..., topography_roi_mask].mean(axis=(2, 3))
        topography[:, fcz_index] = exact_fcz_feature
        del theta_topography_power, theta_topography_db
        cluster_indices = [canonical_channels.index(channel) for channel in channel_sets['frontocentral_core']]
        cluster_raw = raw[:, cluster_indices, :]
        effective_tf_cycles = fit_morlet_cycles_to_signal(
            TF_FREQS,
            TF_CYCLES,
            n_times=len(times),
            sfreq=float(epochs.info['sfreq'])
        )
        tf_power = tfr_array_morlet(
            cluster_raw,
            sfreq=float(epochs.info['sfreq']),
            freqs=TF_FREQS,
            n_cycles=effective_tf_cycles,
            zero_mean=True,
            use_fft=True,
            decim=TF_DECIMATION,
            output='power',
            n_jobs=1,
            verbose=False
        )
        tf_times = times[::TF_DECIMATION]
        tf_db = pooled_db(tf_power, tf_times)
        tf_cluster = tf_db.mean(axis=1)
        del tf_power, tf_db, cluster_raw, raw
        output = epoch_metadata[required_metadata].copy()
        output['theta_primary_exact_FCZ'] = exact_fcz_feature
        for channel_index, channel in enumerate(canonical_channels):
            output[f'theta_topography__{channel}'] = topography[:, channel_index]
        for label, (start_time, stop_time) in TEMPORAL_WINDOWS.items():
            time_mask = (tf_times >= start_time) & (tf_times <= stop_time)
            frequency_mask = (TF_FREQS >= 4.0) & (TF_FREQS <= 8.0)
            output[f'fc_theta__{label}'] = tf_cluster[:, frequency_mask, :][:, :, time_mask].mean(axis=(1, 2))
        primary_time_mask = (tf_times >= PRIMARY_WINDOW[0]) & (tf_times <= PRIMARY_WINDOW[1])
        for label, (low, high) in SPECTRAL_BANDS.items():
            frequency_mask = (TF_FREQS >= low) & (TF_FREQS <= high)
            if not frequency_mask.any():
                raise RuntimeError(f'No frequencies for {label}.')
            output[f'fc_band__{label}'] = tf_cluster[:, frequency_mask, :][:, :, primary_time_mask].mean(axis=(1, 2))
        for frequency_index, frequency in enumerate(TF_FREQS):
            for bin_index in range(len(TF_TIME_EDGES) - 1):
                low = TF_TIME_EDGES[bin_index]
                high = TF_TIME_EDGES[bin_index + 1]
                time_mask = (tf_times >= low) & (tf_times < high)
                column = f'tf__f{frequency_index:02d}_t{bin_index:02d}'
                output[column] = tf_cluster[:, frequency_index, :][:, time_mask].mean(axis=1)
        output.to_parquet(
            checkpoint_dir / f'{participant_id}_time_frequency_features.parquet',
            index=False
        )
        feature_parts.append(output)
        audit_rows.append({
            'participant_id': participant_id,
            'epochs': len(epochs),
            'channels': len(canonical_channels),
            'frontocentral_channels': ','.join(channel_sets['frontocentral_core']),
            'theta_frequency_count': int(len(THETA_FREQS)),
            'theta_topography_decimation': THETA_TOPO_DECIMATION,
            'tf_frequency_count': int(len(TF_FREQS)),
            'tf_decimation': TF_DECIMATION,
            'tf_requested_min_cycles': float(TF_CYCLES.min()),
            'tf_effective_min_cycles': float(effective_tf_cycles.min()),
            'tf_cycles_capped_count': int(np.sum(effective_tf_cycles < TF_CYCLES)),
            'metadata_exact': True,
            'trial_identity_exact': True,
            'status': 'ok'
        })
        if index % 2 == 0 or index == len(participants):
            print(f'  processed {index} of {len(participants)} participants')
    assert canonical_info is not None
    assert canonical_channels is not None
    assert channel_sets is not None
    return (
        pd.concat(feature_parts, ignore_index=True),
        pd.DataFrame(audit_rows),
        canonical_info,
        canonical_channels,
        channel_sets
    )


def extract_rewp_audit_features(
    participants: list[str],
    complete_trials: pd.DataFrame,
    epoch_inventory: pd.DataFrame,
    metadata_inventory: pd.DataFrame,
    frontopolar_channels: list[str],
    checkpoint_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_map = {normalize_participant_id(row.participant_id): Path(row.path) for row in epoch_inventory.itertuples(index=False)}
    metadata_map = {normalize_participant_id(row.participant_id): Path(row.path) for row in metadata_inventory.itertuples(index=False)}
    indexed_trials = complete_trials.set_index(['participant_id', 'trial_number_phase'])
    if not indexed_trials.index.is_unique:
        raise RuntimeError('Complete-symptom trial keys are not unique for RewP audit.')
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    feature_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    required_metadata = ['participant_id', 'trial_number_phase', 'feedback_negative', 'trial_id']
    print('Extracting N=121 frontopolar RewP discrepancy-audit features...')
    for index, participant_id in enumerate(participants, start=1):
        epochs = mne.read_epochs(epoch_map[participant_id], preload=True, verbose='ERROR')
        metadata = epochs.metadata.reset_index(drop=True).copy()
        assert_columns(metadata, required_metadata, f'{participant_id} metadata')
        metadata['participant_id'] = metadata['participant_id'].map(normalize_participant_id)
        metadata['trial_number_phase'] = pd.to_numeric(metadata['trial_number_phase'], errors='raise').astype(int)
        metadata_index = metadata.set_index(['participant_id', 'trial_number_phase'])
        if not metadata_index.index.is_unique:
            raise RuntimeError(f'{participant_id}: nonunique ERP metadata.')
        aligned = indexed_trials.reindex(metadata_index.index)
        if aligned['trial_id'].astype(str).tolist() != metadata_index['trial_id'].astype(str).tolist():
            raise RuntimeError(f'{participant_id}: ERP trial alignment mismatch.')
        selected = available_channels(frontopolar_channels, epochs.ch_names)
        if len(selected) < 3:
            raise RuntimeError(f'{participant_id}: insufficient frontopolar channels.')
        erp = epochs.copy().pick(selected)
        erp.filter(
            l_freq=None,
            h_freq=20.0,
            method='fir',
            phase='zero',
            fir_design='firwin',
            picks=selected,
            n_jobs=1,
            verbose='ERROR'
        )
        erp.apply_baseline((-0.2, 0.0), verbose='ERROR')
        uv = erp.get_data(copy=True).mean(axis=1) * 1000000.0
        times = erp.times.astype(float)
        output = metadata[required_metadata].copy()
        windows = {
            'frontopolar_250_300': (0.25, 0.3),
            'frontopolar_300_350': (0.3, 0.35),
            'frontopolar_350_400': (0.35, 0.4),
            'frontopolar_250_350': (0.25, 0.35)
        }
        for label, (low, high) in windows.items():
            mask = (times >= low) & (times <= high)
            output[label] = uv[:, mask].mean(axis=1)
        output.to_parquet(checkpoint_dir / f'{participant_id}_erp_features.parquet', index=False)
        feature_parts.append(output)
        audit_rows.append({
            'participant_id': participant_id,
            'epochs': len(epochs),
            'frontopolar_channels': ','.join(selected),
            'metadata_exact': True,
            'trial_identity_exact': True,
            'status': 'ok'
        })
        if index % 10 == 0 or index == len(participants):
            print(f'  processed {index} of {len(participants)} participants')
    return (pd.concat(feature_parts, ignore_index=True), pd.DataFrame(audit_rows))
