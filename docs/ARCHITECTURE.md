# Architecture

`main.py` / `sfx_tuner.app` starts the PySide6 desktop. Qt's QProcess runs a single
worker at a time and consumes newline-delimited `SFX_EVENT:` JSON messages. Raw
output is retained in a local job log. Errors, success, and stopped jobs are distinct.

The UI imports NumPy and Qt, never PyTorch or Stable Audio. Dataset jobs use the
desktop interpreter; training, generation, and runtime checks use the selected
runtime interpreter. Requests are validated both before launch and in the worker.

- `audio.py`: FFmpeg import, PCM I/O, source deduplication and grouped holdouts.
- `contracts.py`: captions and ADAM-compatible dataset/adapter metadata.
- `requests.py`: operation bounds and preconditions.
- `jobs.py`: process ownership, streamed events, pause/stop flags, job records.
- `protocol.py`: JSON events and cooperative pause boundaries.
- `training.py`: upstream LoRA training wrapper, progress callbacks and atomic exports.
- `worker.py`: process entrypoint, runtime report and float32 generation.
- `widgets.py`: theme, bounded-memory waveform reader, playback and loss display.
- `scripts/install_runtime.py`: isolated CUDA runtime with a pinned upstream revision.

Checkpoint writes use a temporary adapter file followed by replacement. Metadata
also uses replacement; interruption between those two writes can leave metadata
behind the latest adapter. Intermediate epoch files are adapters, not optimizer
snapshots. Continue always creates a new output folder.

Source audio is decoded to a temporary WAV and cut in bounded blocks. Original
bytes are untouched. Prepared clips use a source-content hash to prevent duplicate
recordings from leaking between training and holdout splits. Captions refer to
clips inside the selected dataset; escaped paths are rejected.

First-release scope: Small-SFX LoRA training and new sound generation. Audio remix,
extension, true optimizer resume, validation metrics, multi-GPU, automatic sample
generation, packaged executables, and other model families are not included.
