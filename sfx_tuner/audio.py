"""Audio workspace storage and dataset preparation, without importing ML packages."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


MEDIA_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac', '.aiff', '.aif',
                    '.wma', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def audio_root(root: Path) -> Path:
    return root.resolve() / 'workspace'


def new_folder(root: Path, category: str, name: str) -> Path:
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', name).strip(' .')[:70] or category
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return audio_root(root) / category / f'{label}_{stamp}_{uuid4().hex[:6]}'


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError('Prepared audio must be 16-bit PCM WAV.')
    return np.frombuffer(raw, dtype='<i2').astype(np.float32).reshape(-1, channels).T / 32768.0, rate


def write_pcm(path: Path, samples: np.ndarray, rate: int) -> None:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] not in (1, 2) or not np.isfinite(values).all():
        raise ValueError('Audio must contain finite mono or stereo samples.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(values.shape[0]); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes((np.clip(values.T, -1, 1) * 32767).astype('<i2').tobytes())


def clip_starts(length: int, clip_length: int, overlap: float) -> list[int]:
    if length <= clip_length:
        return [0]
    stride = max(1, round(clip_length * (1 - overlap)))
    starts = list(range(0, length - clip_length + 1, stride))
    if starts[-1] != length - clip_length:
        starts.append(length - clip_length)
    return starts


def source_group(source: Path) -> str:
    """Hash contents so duplicate imports cannot leak across the validation split."""
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(request: dict, emit, stopped=lambda: False) -> dict:
    output = Path(request['output_dir'])
    if output.exists():
        raise ValueError('Choose a new dataset folder; existing datasets are never overwritten.')
    sources = [Path(p).expanduser().resolve() for p in request['sources']]
    if not sources or any(not p.is_file() or p.suffix.lower() not in MEDIA_EXTENSIONS for p in sources):
        raise ValueError('Select existing supported audio or video files.')
    rate, channels = int(request['sample_rate']), int(request['channels'])
    seconds, overlap = float(request['clip_seconds']), float(request['overlap'])
    if rate not in (16000, 24000, 44100, 48000) or channels not in (1, 2):
        raise ValueError('Choose a supported sample rate and mono or stereo.')
    if not math.isfinite(seconds) or not .5 <= seconds <= 120 or not 0 <= overlap <= .75:
        raise ValueError('Clip length must be 0.5–120 seconds and overlap 0–75%.')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Audio import needs ffmpeg. Install it and make it available on PATH.')
    # Determine source groups before cutting clips; identical files stay together.
    groups = [source_group(p) for p in sources]
    unique_groups = sorted(set(groups))
    validation_count = max(1, round(len(unique_groups) * .15)) if len(unique_groups) > 1 else 0
    validation_groups = set(unique_groups[:validation_count])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'clips').mkdir()
    manifest = {'version': 1, 'name': request.get('name', output.name), 'sample_rate': rate,
                'channels': channels, 'clip_seconds': seconds, 'overlap': overlap, 'clips': [], 'sources': [],
                'complete': False, 'created_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(output / 'dataset.json', manifest)
    total_seconds = 0.0
    seen = set()
    try:
        for index, (source, group) in enumerate(zip(sources, groups)):
            if stopped():
                raise InterruptedError('Audio preparation stopped; the partial dataset is not trainable.')
            if group in seen:
                emit(event='log', message=f'Skipping duplicate recording: {source.name}')
                continue
            seen.add(group)
            decoded = output / 'decoded.wav'
            command = [ffmpeg, '-nostdin', '-loglevel', 'error', '-y', '-i', str(source),
                       '-map', '0:a:0', '-vn', '-ac', str(channels), '-ar', str(rate), '-c:a', 'pcm_s16le', str(decoded)]
            # ffmpeg performs filtered resampling and handles floating-point WAV and video audio.
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            with (output / 'decode-error.txt').open('wb') as errors:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors, creationflags=creationflags)
                try:
                    while process.poll() is None:
                        if stopped():
                            raise InterruptedError('Audio preparation stopped.')
                        try:
                            process.wait(timeout=.15)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
            if process.returncode:
                detail = (output / 'decode-error.txt').read_text(encoding='utf-8', errors='replace')[-2000:]
                raise RuntimeError(f'Could not read audio from {source.name}: {detail}')
            with wave.open(str(decoded), 'rb') as handle:
                length = handle.getnframes()
                total_seconds += length / rate
                before = len(manifest['clips'])
                for start in clip_starts(length, round(rate * seconds), overlap):
                    if stopped():
                        raise InterruptedError('Audio preparation stopped.')
                    handle.setpos(start)
                    pcm = handle.readframes(round(rate * seconds))
                    samples = np.frombuffer(pcm, '<i2').astype(np.float32).reshape(-1, channels).T / 32768
                    if samples.size == 0 or (request.get('skip_silence', True) and float(np.max(np.abs(samples))) < .001):
                        continue
                    relative = f'clips/{len(manifest["clips"]):07d}.wav'
                    write_pcm(output / relative, samples, rate)
                    caption_file = source.with_suffix('.txt')
                    caption = caption_file.read_text(encoding='utf-8').strip() if caption_file.is_file() else request.get('caption', '').strip()
                    (output / relative).with_suffix('.txt').write_text(caption, encoding='utf-8')
                    manifest['clips'].append({'path': relative, 'source_id': group, 'source_index': index,
                                              'start_seconds': start / rate, 'samples': samples.shape[-1],
                                              'split': 'validation' if group in validation_groups else 'train'})
                manifest['sources'].append({'path': str(source), 'id': group, 'seconds': length / rate,
                                             'clips': len(manifest['clips']) - before})
            decoded.unlink(missing_ok=True)
            atomic_json(output / 'dataset.json', manifest)
            emit(event='prepare', current=index + 1, total=len(sources), clips=len(manifest['clips']))
        train_count = sum(c['split'] == 'train' for c in manifest['clips'])
        if not train_count:
            raise ValueError('No non-silent training clips were found. Check the source recordings.')
        manifest.update(complete=True, source_seconds=total_seconds, training_clips=train_count,
                        validation_clips=len(manifest['clips']) - train_count)
        atomic_json(output / 'dataset.json', manifest)
    finally:
        (output / 'decoded.wav').unlink(missing_ok=True)
        (output / 'decode-error.txt').unlink(missing_ok=True)
    emit(event='done', output_dir=str(output), clips=len(manifest['clips']))
    return manifest
