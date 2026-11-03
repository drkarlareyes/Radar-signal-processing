# Radar vital-sign signal processing (SiViS)

Python processing for contactless heart-rate and respiration-rate estimation with a 60 GHz FMCW radar (Texas Instruments IWR6843), written for the multi-patient recordings of the **SiViS** dataset. SiViS contains radar recordings of simulated patients in different clinical states, recorded from several radar positions, with reference values for heart and respiration rate.

## Pipeline

`core_pipeline.py` processes each raw recording in these steps:

1. **Decoding.** Raw int16 I/Q samples into frames × 4 receivers × chirps × ADC samples.
2. **Range FFT** per chirp, then a range–angle map (angle FFT across receivers).
3. **Target detection.** Peak picking on the range–angle map, with minimum range and angle separation, so that more than one person can be monitored.
4. **Beamforming.** MVDR beamforming towards each detected target, with range gating.
5. **Heart rate.** Unwrapped phase, phase derivative, band-pass 0.8–3 Hz, peak of the Welch spectrum.
6. **Respiration rate.** Band-pass 0.1–0.6 Hz, Hilbert envelope, median filtering and mean interval between breaths (5–35 breaths/min).

The output is one row per detected target with the estimated heart and respiration rate next to the reference values in the annotation file.

The notebooks in `methods/` are exploratory comparisons of alternative steps (slow-time construction, range-bin and receiver selection, filtering, Goertzel checks of candidate rates).

## Data

Download SiViS from Harvard Dataverse: https://doi.org/10.7910/DVN/S9SNAK

The pipeline expects the `.bin` recordings in a `plots/` folder and an `annotations.xlsx` file with a `fname` column (optional: `num_adc_samples`, `num_chirps`, `framerate`, `gt_hr`, `gt_rr`). The number of people is read from the `ManN` part of each file name.

## Related publications

- Reyes-Leiva KM, Gupta A, Grosmaire A, Cerny M. SiViS: Simulated Multi-Patient Physiological Clinical States for Advanced Vital Sign Radar Monitoring Research. *Scientific Data* (2026). https://doi.org/10.1038/s41597-025-06449-9
- Reyes-Leiva KM, Gupta A, Cerny M. Radar Placement Effects on Multi-Patient Heart and Respiration Monitoring: SiViS Dataset Validation. In *Biomedical Engineering Systems and Technologies*, Springer (2026). https://doi.org/10.1007/978-3-032-13312-0_11
- Reyes-Leiva KM, Vondal M, Yang M, Wang Y. Estimating Respiratory Rate in Freely Moving Users Using Independent Component and Multi-Resolution Analysis. *Computers in Biology and Medicine* (2025). https://doi.org/10.1016/j.compbiomed.2025.110957
- Gupta A, Reyes-Leiva KM, Cerny M. Multi-Objective Undercomplete Independent Component Analysis for Radar Signals Based Heart Rate and Interbeat Interval Estimations. IEEE EMBC (2025). https://doi.org/10.1109/EMBC58623.2025.11253136

If you use the dataset, please cite the SiViS paper.

## Requirements

Python 3 with NumPy, SciPy, pandas and openpyxl.

## Author

Karla Miriam Reyes Leiva, PhD. [Website](https://drkarlareyes.github.io) · ORCID [0000-0002-3269-0830](https://orcid.org/0000-0002-3269-0830)

Code: MIT licence.
