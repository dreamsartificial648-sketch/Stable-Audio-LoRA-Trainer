# Validation

This document records checks for the standalone app, not ADAM's prior results.

## Automated checks

The test suite covers real FFmpeg dataset preparation and source preservation,
caption requirements and path containment, adapter continuation routing, event
parsing, cooperative stop/pause behavior, and the Qt desktop workflow.

## GPU checks

`scripts/smoke_gpu.py` verifies base inference, a short synthetic LoRA training run,
continuation into a new folder, stop/save while paused, and inference through the continued adapter.
It checks finite/non-silent 44.1 kHz stereo output with the exact requested frame
count, saved adapter files, completed epochs, and frozen-base training behavior.

These checks establish functional operation. They do not establish sound quality,
dataset suitability, a minimum GPU memory requirement, or stability over long runs.
Other machines, GPUs, and long training sessions require additional testing.

## Local verification, October 25, 2026

- CPU/Qt checks: **23 passed, 1 skipped**. The skipped test requires permission to
  create a Windows symlink; path traversal and out-of-dataset caption writes are tested.
- Desktop installed successfully into its own `.venv`, with `pip check` passing.
  All five pages were rendered and inspected. This environment contains no PyTorch.
- First GPU smoke pass: Windows, Python 3.10.6, PyTorch 2.7.1+cu126,
  RTX 3060 12 GB, pinned Stable Audio source in an existing runtime.
- Second GPU smoke pass: **all six checks passed in the app's newly installed,
  isolated runtime**. This used the standalone dependency checkout, not ADAM's
  Python environment. Both environments passed `pip check`.
- Base generation: finite, non-silent stereo WAV, exactly 88,200 frames at 44.1 kHz.
- One-step rank-4 LoRA: finite loss, 2,478,160 adapter parameters, frozen base checked.
- Continuation: adapter rank retained, epoch count advanced, new output folder.
- Continued adapter generation: finite, non-silent stereo WAV, exact requested length.
- Stop while paused: saved an adapter after one batch, marked the session stopped,
  and left the incomplete epoch out of `epochs_completed`.
- Peak allocated memory for the two-second rank-4 training check: **1.69 GiB**.
  This is PyTorch's peak allocated memory for a short check, not total process VRAM
  or a recommended GPU capacity.

The checks do not evaluate perceptual quality or prove convergence. Existing cached
weights were used; a first-time gated model download was not tested.
