# Feedback Valence and Prediction Error in EEG

Selected Python analysis methods accompanying the preprint:

Sinha, V., & Ravindra, V. (2026). *Feedback Valence, Prediction Error, and the Limits of Computational Attribution in Symptom-Related EEG*. **bioRxiv**. https://doi.org/10.64898/2026.11.04.762513

## Code overview

| File | Contents |
| --- | --- |
| `analysis/rl_core.py` | Q-learning models, choice probabilities, and prediction errors |
| `analysis/rl_fit.py` | Multistart maximum-likelihood fitting and simulated choices |
| `analysis/eeg_preprocessing.py` | Feedback-event alignment, artifact screening, and ICA |
| `analysis/theta_power.py` | Feedback-locked FCz theta power and baseline measurements |
| `analysis/symptom_models.py` | Participant fixed-effects regression and cluster-robust inference |
| `analysis/spatial_spectral.py` | Spatial, spectral, time-frequency, and ERP analyses |
| `analysis/reliability.py` | Feedback contrasts, split-half reliability, and resampling |
| `analysis/independent_models.py` | Independent-cohort valence and prediction-error models |
| `analysis/simulations.py` | Model identification and parameter/recovery simulations |

## Data and requirements

The two EEG datasets are publicly available on OpenNeuro:

- [Primary dataset: ds003474, version 1.1.0](https://openneuro.org/datasets/ds003474/versions/1.1.0)
- [Independent dataset: ds004532, version 1.2.0](https://openneuro.org/datasets/ds004532/versions/1.2.0)

Use Python 3.11 and install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The modules can be imported, for example, with `from analysis import rl_core, rl_fit`. Their functions expect appropriate EEG epochs or analysis-ready data tables, including study-specific column names.

## Scope

This repository provides selected implementations of the study's computational and statistical methods. It is **not an end-to-end reproduction pipeline** and does not include raw EEG recordings, prepared analysis tables, or the scripts used to produce every published table and figure. Additional code or supporting materials may be requested from the corresponding author, subject to sharing restrictions.

## Citation and license

If you use this code, please cite the [associated preprint](https://doi.org/10.64898/2026.11.04.762513). Please also cite the original datasets when appropriate. The code is available under the [MIT license](LICENSE).
