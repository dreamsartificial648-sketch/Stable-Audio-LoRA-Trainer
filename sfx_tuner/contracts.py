"""Small-SFX contracts; importing these never imports the audio runtime."""
from __future__ import annotations

from pathlib import Path
from sfx_tuner.audio import audio_root, read_json
from sfx_tuner.constants import ARCHITECTURE, BASE_MODEL, UPSTREAM_REVISION

BACKEND = 'stable_audio_3'
MAX_TRAIN_BATCH_SIZE = 256


def base_entry() -> dict:
    return dict(name='Stable Audio 3 Small-SFX · original base model', path='stable-audio-3:base',
                architecture=ARCHITECTURE, sample_rate=44100, channels=2, clip_seconds=120,
                base_model=True, legacy=False)


def caption_rows(folder: Path) -> list[dict]:
    manifest = read_json(folder / 'dataset.json')
    rows = []
    for clip in manifest.get('clips', []):
        path = (folder / clip['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Dataset contains a missing clip or a clip outside its folder.')
        caption_file = path.with_suffix('.txt')
        if not caption_file.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Caption points outside its dataset folder.')
        caption = caption_file.read_text(encoding='utf-8').strip() if caption_file.is_file() else ''
        rows.append(dict(path=str(path), caption=caption, split=clip.get('split', 'train')))
    return rows


def save_captions(folder: Path, captions: list[tuple[str, str]]) -> None:
    allowed = {row['path'] for row in caption_rows(folder)}
    for filename, caption in captions:
        path = Path(filename).resolve()
        if str(path) not in allowed:
            raise ValueError('Caption must belong to the selected dataset.')
        path.with_suffix('.txt').write_text(caption.strip(), encoding='utf-8')


def training_rows(folder: Path) -> list[dict]:
    manifest = read_json(folder / 'dataset.json')
    if not manifest.get('complete'):
        raise ValueError('Prepare and select a complete audio dataset first.')
    seconds = manifest.get('clip_seconds', 0)
    if not isinstance(seconds, (int, float)) or not .5 <= seconds <= 120:
        raise ValueError('Dataset clip length must be between 0.5 and 120 seconds.')
    rows = [row for row in caption_rows(folder) if row['split'] == 'train']
    if not rows:
        raise ValueError('The selected dataset has no training clips.')
    missing = [Path(row['path']).name for row in rows if not row['caption']]
    if missing:
        raise ValueError(f'Add and save sound descriptions in Dataset before fine-tuning. Missing {len(missing)} caption(s): ' + ', '.join(missing[:5]))
    return rows


def checkpoint_info(folder: Path) -> dict:
    info = read_json(folder / 'audio_model.json')
    if (info.get('architecture') != ARCHITECTURE or info.get('base_model') != BASE_MODEL
            or not (folder / 'adapter.safetensors').is_file()
            or type(info.get('rank')) is not int or not 1 <= info['rank'] <= 64):
        raise ValueError('Select a Stable Audio 3 Small-SFX LoRA compatible with this app or ADAM.')
    return info


def discover_models(root: Path) -> list[dict]:
    result = []
    for path in (audio_root(root) / 'models').glob('*/audio_model.json'):
        try:
            info = checkpoint_info(path.parent)
            result.append({**info, 'path': str(path.parent), 'legacy': False})
        except ValueError:
            continue
    return result
