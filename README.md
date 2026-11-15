# psa-3cams-dark-bright-calibration

Recording and preprocessing code, and the processed data, of a manuscript that is under peer review.

> **Not licensed for use.** This repository is public only so that the editor and the reviewers of the manuscript can inspect and run its contents. Until the manuscript is published, all rights are reserved, and the code and the data may not be used, copied, modified or redistributed for any other purpose. An open-source license will replace this notice when the paper is published.

In the experiment, participants looked at a nine-point grid on a dark and on a bright screen while an EyeLink 1000 Plus and two high-resolution cameras recorded their eyes. This repository holds the scripts that ran the recording, the pipeline that turns the recordings into pupil and corneal reflection measurements, and the resulting data. The analyses and the simulations of the manuscript are in the companion repository [psa-eye-model-evaluation](https://github.com/mh-salari/psa-eye-model-evaluation).

## What is included and what is not

Included:

- All recording and preprocessing scripts.
- The setup, protocol and calibration files of the experiment, and the processed measurements, in `data/`.
- The recordings of the artificial eye that the pixel-to-millimetre calibration uses.
- Example figures from one participant, who is the first author.

Not included:

- **The recordings of the participants** (eye videos, eye images and EyeLink files). They are personal data. The preprocessing steps therefore cannot be re-run from this repository; their results are the CSV files in `data/`.
- **SWAGS**, the software that records the two cameras. It is not public yet. The scripts that need it are `record_trial.py`, the scripts in `camera_setup/` and `pupil_size_to_mm/camera/capture_artificial_eye.py`. All other scripts run without it.

## Contents

| Path | What it holds |
| --- | --- |
| `docs/RUNBOOK.md` | The steps from the lab setup to the processed eye features, in order. |
| `camera_setup/` | Registering, configuring and previewing the cameras. |
| `record_trial.py`, `show_targets_dark_light.py`, `clickable/` | Recording one participant, checking the stimulus, and launchers for both. |
| `eye_images_processing/` | Extracting the eye images, annotating pupil and corneal reflection, aligning the recording phases, and exporting the measurements. |
| `pupil_size_to_mm/` | The calibrations that convert pupil size to millimetres, for the cameras and for the EyeLink 1000 Plus. |
| `figures/` | The example figures of the manuscript. |
| `utils/` | Helpers shared by the scripts. |
| `data/` | See below. |

## Data

| File | What it holds |
| --- | --- |
| `raw_eye_features.csv` | Pupil center, corneal reflection, pupil-corneal reflection vector and pupil diameter per camera, participant, eye, target and recording phase. The analyses of the manuscript use this file. |
| `aligned_features_all_steps.csv` | The same measurements after the recording phases are aligned on the corneal reflection. |
| `apparent_pupil_center_shift_estimates.csv` | The apparent pupil center shift between the dilated and the constricted condition, for every pair of recording phases it can be measured from. |
| `lab_setup.json`, `experiment_protocol.json` | The geometry of the setup and the parameters of the experiment. |
| `exclusions.json` | The excluded recordings and the reason for each. |
| `eye_image_px_per_mm.json`, `eyelink1000plus_units_per_mm.json` | The results of the pupil-size calibrations. |
| `camera_aliases.json`, `cameras/` | The camera names and settings, and the photos of the artificial eye. |
| `eyelink1000plus/` | The EyeLink 1000 Plus recordings of the artificial eye. |
| `cam1_and_cam2.eye_annotation_project.json` | The annotation settings. Its image paths refer to the recordings that are not included. |

Participants are identified by random three-digit numbers.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/). All code was written and run on macOS; it is not tested on other operating systems, and the launchers in `clickable/` work on macOS only. The dependencies are installed with:

```
uv sync
```

`pylink` is installed from SR Research's package index. Recording also needs the EyeLink 1000 Plus, its Developers Kit, and SWAGS.

## License

All rights reserved until the manuscript is published, see `LICENSE`.
