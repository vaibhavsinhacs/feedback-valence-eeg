"""Feedback-locked theta power at FCz.

This is the measurement used in the primary analysis: Morlet power in the
4–8 Hz band and 200–400 ms after feedback, referenced to the participant's
pooled -300 to -200 ms baseline. Two sensitivity measurements are included.
Each input should contain epochs from one participant.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

BASELINE = (-0.300, -0.200)
ROI_TIME = (0.200, 0.400)
THETA_BAND = (4.0, 8.0)
FREQUENCIES = np.logspace(0.01, 1.7, 50)
CYCLES = np.logspace(np.log10(3.0), np.log10(10.0), 50)
THETA_MASK = (FREQUENCIES >= THETA_BAND[0]) & (FREQUENCIES <= THETA_BAND[1])
THETA_FREQUENCIES = FREQUENCIES[THETA_MASK]
THETA_CYCLES = CYCLES[THETA_MASK]


def measure_theta(epochs: mne.Epochs) -> dict[str, np.ndarray]:
    """Return three trialwise theta measurements for one participant.

    The primary measure uses one baseline per frequency pooled across trials.
    `within_trial_baseline_db` uses a separate baseline for each trial.
    """
    fcz = epochs.copy().pick(["FCZ"])
    data = fcz.get_data(copy=True)
    times = fcz.times.astype(float)
    baseline = (times >= BASELINE[0]) & (times <= BASELINE[1])
    response = (times >= ROI_TIME[0]) & (times <= ROI_TIME[1])
    if baseline.sum() != 51 or response.sum() != 101:
        raise ValueError("Expected 500 Hz epochs with 51 baseline and 101 response samples")

    power = tfr_array_morlet(
        data, sfreq=float(fcz.info["sfreq"]), freqs=THETA_FREQUENCIES,
        n_cycles=THETA_CYCLES, zero_mean=True, use_fft=True, decim=1,
        output="power", n_jobs=1, verbose=False,
    )[:, 0, :, :]

    pooled_baseline = np.mean(power[:, :, baseline], axis=(0, 2))
    primary_db = 10.0 * np.log10(power / pooled_baseline[None, :, None])
    primary_theta = np.mean(primary_db[:, :, response], axis=(1, 2))

    trial_baseline = np.mean(power[:, :, baseline], axis=2, keepdims=True)
    within_trial_db = 10.0 * np.log10(power / trial_baseline)
    trial_theta = np.mean(within_trial_db[:, :, response], axis=(1, 2))
    unbaselined_theta = np.mean(10.0 * np.log10(power[:, :, response]), axis=(1, 2))

    return {
        "pooled_baseline_db": primary_theta,
        "within_trial_baseline_db": trial_theta,
        "unbaselined_log_power_db": unbaselined_theta,
    }
