"""EEG preprocessing methods for the primary cohort.

Channel handling, feedback-event alignment, artifact detection, and ICA are
shown here independently of the private dataset paths and audit files. The
parameters match the original preprocessing implementation. The complete
subject-level file-processing pipeline is not included.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


EOG_CHANNELS = ["HEOG", "VEOG"]
ECG_CHANNELS = ["EKG", "ECG"]
MASTOID_CHANNELS = ["M1", "M2"]
FRONTOPOLAR_CHANNELS = ("FP1", "FPZ", "FP2")
DROP_CHANNELS = ["CB1", "CB2", "EKG", "ECG"]
BAD_CHANNEL_Z = 3.0
BAD_EPOCH_Z = 3.0
TMIN, TMAX = -2.0, 2.0
FILTER_L_FREQ, FILTER_H_FREQ, NOTCH_FREQ = 0.5, 100.0, 60.0
ICA_FIT_L_FREQ = 1.0
ICA_VARIANCE_COMPONENTS = 0.99
ICA_RANDOM_STATE = 97
ICA_EOG_THRESHOLD = 3.0


def canonical_channel(name: str) -> str:
    return str(name).strip().upper().replace(' ', '').replace('-', '').replace('_', '')


def safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
        return np.zeros_like(values, dtype=float)
    result = (values - mean) / standard_deviation
    result[~np.isfinite(result)] = 0.0
    return result


def normalize_raw_channels(raw: mne.io.BaseRaw) -> None:
    rename_map = {original: canonical_channel(original) for original in raw.ch_names}
    canonical_names = list(rename_map.values())
    if len(set(canonical_names)) != len(canonical_names):
        raise RuntimeError('Channel canonicalization would create duplicate labels.')
    raw.rename_channels(rename_map)


def assign_channel_types(raw: mne.io.BaseRaw) -> None:
    type_map: dict[str, str] = {}
    for channel in EOG_CHANNELS:
        if channel in raw.ch_names:
            type_map[channel] = 'eog'
    for channel in ECG_CHANNELS:
        if channel in raw.ch_names:
            type_map[channel] = 'ecg'
    if type_map:
        raw.set_channel_types(type_map, verbose='ERROR')


def get_scalp_channels(raw_or_epochs: Any) -> list[str]:
    eeg_channels = [raw_or_epochs.ch_names[index] for index in mne.pick_types(
        raw_or_epochs.info,
        eeg=True,
        eog=False,
        ecg=False,
        misc=False,
        exclude=[]
    )]
    scalp_channels = [channel for channel in eeg_channels if channel not in MASTOID_CHANNELS]
    return scalp_channels


def verify_scalp_locations(raw: mne.io.BaseRaw, scalp_channels: list[str]) -> None:
    missing = []
    for channel in scalp_channels:
        index = raw.ch_names.index(channel)
        location = np.asarray(raw.info['chs'][index]['loc'][:3], dtype=float)
        valid = bool(np.isfinite(location).all() and (not np.allclose(location, 0.0)))
        if not valid:
            missing.append(channel)
    if missing:
        raise RuntimeError('Missing standard-montage positions for scalp channels: ' + ', '.join(missing))


def build_events_and_metadata(
    participant_trials: pd.DataFrame,
    sfreq: float,
    first_samp: int
) -> tuple[np.ndarray, pd.DataFrame]:
    required_columns = [
        'participant_id',
        'trial_number_phase',
        'feedback_onset_sec',
        'feedback_valence',
        'reward'
    ]
    missing = [column for column in required_columns if column not in participant_trials.columns]
    if missing:
        raise RuntimeError('Canonical trial table is missing required columns: ' + ', '.join(missing))
    trials = participant_trials.sort_values('trial_number_phase').reset_index(drop=True).copy()
    participant_values = trials['participant_id'].astype(str).unique()
    if len(participant_values) != 1:
        raise RuntimeError('Participant trial subset contains multiple participant IDs.')
    trial_numbers = pd.to_numeric(trials['trial_number_phase'], errors='raise').astype(int)
    feedback_onsets = pd.to_numeric(trials['feedback_onset_sec'], errors='raise').astype(float)
    feedback_valence = trials['feedback_valence'].astype('string').str.strip().str.lower()
    reward = pd.to_numeric(trials['reward'], errors='raise').astype(int)
    if not feedback_valence.isin(['negative', 'positive']).all():
        invalid_labels = sorted(feedback_valence.loc[~feedback_valence.isin(['negative', 'positive'])].dropna().unique().tolist())
        raise RuntimeError('feedback_valence contains invalid labels: ' + ', '.join(invalid_labels))
    if not reward.isin([0, 1]).all():
        invalid_rewards = sorted(reward.loc[~reward.isin([0, 1])].unique().tolist())
        raise RuntimeError('reward contains values other than 0 and 1: ' + ', '.join((str(value) for value in invalid_rewards)))
    feedback_negative = (1 - reward).astype(int)
    valence_negative = (feedback_valence == 'negative').astype(int)
    if not np.array_equal(feedback_negative.to_numpy(), valence_negative.to_numpy()):
        mismatch_count = int((feedback_negative != valence_negative).sum())
        raise RuntimeError(f'Canonical reward and feedback_valence disagree for {mismatch_count} trials.')
    event_samples = np.rint(feedback_onsets.to_numpy() * float(sfreq)).astype(int) + int(first_samp)
    if (np.diff(event_samples) <= 0).any():
        raise RuntimeError('Canonical feedback events are not strictly increasing.')
    event_codes = np.where(feedback_negative.to_numpy() == 1, 1, 2).astype(int)
    events = np.column_stack([
        event_samples,
        np.zeros(len(event_samples), dtype=int),
        event_codes
    ]).astype(int)
    metadata = pd.DataFrame({
        'participant_id': trials['participant_id'].astype(str).to_numpy(),
        'trial_number_phase': trial_numbers.to_numpy(),
        'feedback_onset_sec': feedback_onsets.to_numpy(),
        'feedback_valence': feedback_valence.to_numpy(),
        'reward': reward.to_numpy(),
        'feedback_negative': feedback_negative.to_numpy(),
        'event_sample': event_samples,
        'event_code': event_codes,
        'canonical_event_index': np.arange(len(trials), dtype=int)
    })
    if 'trial_id' in trials.columns:
        metadata['trial_id'] = trials['trial_id'].astype(str).to_numpy()
    return (events, metadata)


def detect_bad_channels(epochs: mne.Epochs, scalp_channels: list[str]) -> tuple[list[str], pd.DataFrame]:
    data_uv = epochs.get_data(picks=scalp_channels, copy=True) * 1000000.0
    variance_uv2 = np.var(data_uv, axis=(0, 2))
    log_variance = np.log(np.maximum(variance_uv2, np.finfo(float).tiny))
    epoch_ptp_uv = np.ptp(data_uv, axis=2)
    ptp95_uv = np.percentile(epoch_ptp_uv, 95, axis=0)
    epoch_absmax_uv = np.max(np.abs(data_uv), axis=2)
    absmax95_uv = np.percentile(epoch_absmax_uv, 95, axis=0)
    median_scalp_signal = np.median(data_uv, axis=1)
    median_flat = median_scalp_signal.reshape(-1)
    correlation_with_median = []
    for channel_index in range(len(scalp_channels)):
        channel_flat = data_uv[:, channel_index, :].reshape(-1)
        if np.std(channel_flat) == 0.0 or np.std(median_flat) == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(channel_flat, median_flat)[0, 1])
        correlation_with_median.append(correlation)
    correlation_with_median = np.asarray(correlation_with_median, dtype=float)
    z_log_variance = safe_zscore(log_variance)
    z_ptp95 = safe_zscore(ptp95_uv)
    z_absmax95 = safe_zscore(absmax95_uv)
    z_negative_correlation = safe_zscore(-correlation_with_median)
    bad_by_variance = z_log_variance > BAD_CHANNEL_Z
    bad_by_ptp = z_ptp95 > BAD_CHANNEL_Z
    bad_by_absmax = z_absmax95 > BAD_CHANNEL_Z
    bad_by_correlation = z_negative_correlation > BAD_CHANNEL_Z
    frontopolar_channel = np.isin(
        np.asarray(scalp_channels, dtype=str),
        np.asarray(FRONTOPOLAR_CHANNELS, dtype=str)
    )
    bad_by_ptp_decision = bad_by_ptp & ~frontopolar_channel
    bad_by_absmax_decision = bad_by_absmax & ~frontopolar_channel
    bad_channel = bad_by_variance | bad_by_ptp_decision | bad_by_absmax_decision | bad_by_correlation
    audit = pd.DataFrame({
        'channel': scalp_channels,
        'variance_uv2': variance_uv2,
        'log_variance': log_variance,
        'ptp95_uv': ptp95_uv,
        'absmax95_uv': absmax95_uv,
        'correlation_with_median': correlation_with_median,
        'z_log_variance': z_log_variance,
        'z_ptp95': z_ptp95,
        'z_absmax95': z_absmax95,
        'z_negative_correlation': z_negative_correlation,
        'is_frontopolar_channel': frontopolar_channel,
        'frontopolar_amplitude_exception_applied': frontopolar_channel,
        'bad_by_variance': bad_by_variance,
        'bad_by_ptp': bad_by_ptp,
        'bad_by_absmax': bad_by_absmax,
        'bad_by_negative_correlation': bad_by_correlation,
        'bad_by_ptp_decision': bad_by_ptp_decision,
        'bad_by_absmax_decision': bad_by_absmax_decision,
        'bad_channel': bad_channel,
        'bad_channel_z_threshold': BAD_CHANNEL_Z
    })
    bad_channels = audit.loc[audit['bad_channel'], 'channel'].tolist()
    return (bad_channels, audit)


def detect_bad_epochs(epochs: mne.Epochs, scalp_channels: list[str]) -> pd.DataFrame:
    data_uv = epochs.get_data(picks=scalp_channels, copy=True) * 1000000.0
    mean_absolute_uv = np.mean(np.abs(data_uv), axis=(1, 2))
    variance_uv2 = np.var(data_uv, axis=(1, 2))
    log_variance = np.log(np.maximum(variance_uv2, np.finfo(float).tiny))
    maximum_absolute_uv = np.max(np.abs(data_uv), axis=(1, 2))
    maximum_channel_ptp_uv = np.max(np.ptp(data_uv, axis=2), axis=1)
    z_mean_absolute = safe_zscore(mean_absolute_uv)
    z_log_variance = safe_zscore(log_variance)
    z_maximum_absolute = safe_zscore(maximum_absolute_uv)
    z_maximum_channel_ptp = safe_zscore(maximum_channel_ptp_uv)
    bad_by_mean_absolute = z_mean_absolute > BAD_EPOCH_Z
    bad_by_variance = z_log_variance > BAD_EPOCH_Z
    bad_by_maximum_absolute = z_maximum_absolute > BAD_EPOCH_Z
    bad_by_ptp = z_maximum_channel_ptp > BAD_EPOCH_Z
    bad_epoch = bad_by_mean_absolute | bad_by_variance | bad_by_maximum_absolute | bad_by_ptp
    metadata = epochs.metadata.reset_index(drop=True).copy()
    audit = metadata[['participant_id', 'trial_number_phase', 'event_sample', 'event_code']].copy()
    audit['mean_absolute_uv'] = mean_absolute_uv
    audit['variance_uv2'] = variance_uv2
    audit['log_variance'] = log_variance
    audit['maximum_absolute_uv'] = maximum_absolute_uv
    audit['maximum_channel_ptp_uv'] = maximum_channel_ptp_uv
    audit['z_mean_absolute'] = z_mean_absolute
    audit['z_log_variance'] = z_log_variance
    audit['z_maximum_absolute'] = z_maximum_absolute
    audit['z_maximum_channel_ptp'] = z_maximum_channel_ptp
    audit['bad_by_mean_absolute'] = bad_by_mean_absolute
    audit['bad_by_variance'] = bad_by_variance
    audit['bad_by_maximum_absolute'] = bad_by_maximum_absolute
    audit['bad_by_ptp'] = bad_by_ptp
    audit['bad_epoch'] = bad_epoch
    audit['bad_epoch_z_threshold'] = BAD_EPOCH_Z
    audit['epoch_position_before_drop'] = np.arange(len(audit), dtype=int)
    return audit


def run_ica(epochs: mne.Epochs, scalp_channels: list[str]) -> tuple[mne.Epochs, pd.DataFrame, int]:
    epochs_for_ica = epochs.copy()
    epochs_for_ica.filter(
        l_freq=ICA_FIT_L_FREQ,
        h_freq=None,
        picks=scalp_channels,
        phase='zero',
        fir_design='firwin',
        verbose='ERROR'
    )
    ica = ICA(
        n_components=ICA_VARIANCE_COMPONENTS,
        method='fastica',
        random_state=ICA_RANDOM_STATE,
        max_iter='auto'
    )
    ica.fit(epochs_for_ica, picks=scalp_channels, verbose='ERROR')
    excluded_components: set[int] = set()
    audit_rows: list[dict[str, Any]] = []
    for eog_channel in EOG_CHANNELS:
        indices, scores = ica.find_bads_eog(
            epochs_for_ica,
            ch_name=eog_channel,
            threshold=ICA_EOG_THRESHOLD,
            verbose='ERROR'
        )
        indices = [int(index) for index in indices]
        excluded_components.update(indices)
        audit_rows.append({
            'eog_channel': eog_channel,
            'detected_components': ';'.join((str(index) for index in sorted(indices))),
            'n_detected_components': len(indices),
            'maximum_absolute_score': float(np.max(np.abs(scores))) if len(scores) else np.nan,
            'threshold': ICA_EOG_THRESHOLD
        })
    excluded = sorted(excluded_components)
    if excluded:
        ica.apply(epochs, exclude=excluded, verbose='ERROR')
    audit_rows.append({
        'eog_channel': 'UNION',
        'detected_components': ';'.join((str(index) for index in excluded)),
        'n_detected_components': len(excluded),
        'maximum_absolute_score': np.nan,
        'threshold': ICA_EOG_THRESHOLD,
        'n_ica_components': int(ica.n_components_),
        'ica_method': 'fastica',
        'ica_random_state': ICA_RANDOM_STATE,
        'ica_variance_components': ICA_VARIANCE_COMPONENTS,
        'ica_fit_channels': ';'.join(scalp_channels)
    })
    return (epochs, pd.DataFrame(audit_rows), len(excluded))
