"""Shared helpers used by more than one workflow sub-dir of this repository.

Sub-dir-only helpers live flat inside the owning sub-dir; this package carries only what several
sub-dirs need: the SWAGS-key to camera-alias mapping, the per-camera px/mm calibration loaders,
the lab geometry and experiment-protocol loaders, the per-participant exclusions, the HV9 target
grid, the recording's phase and eye vocabulary, the pupil boundary geometry, and the apparent
pupil center shift.
"""
