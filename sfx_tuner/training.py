"""LoRA training adapted from ADAM; see THIRD_PARTY_NOTICES.md."""
from datetime import datetime, timezone
import math
from pathlib import Path
from sfx_tuner.audio import read_json, atomic_json, read_pcm
from sfx_tuner.contracts import ARCHITECTURE, BASE_MODEL, UPSTREAM_REVISION, checkpoint_info, training_rows
from sfx_tuner.protocol import emit, stopped, wait_if_paused

def train(request):
    import torch
    import torchaudio
    import pytorch_lightning as pl
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.models.lora.utils import load_lora_checkpoint
    from stable_audio_3.models.lora.utils import name_is_lora
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    if not torch.cuda.is_available():
        raise ValueError('Small-SFX LoRA training needs a CUDA GPU in its separate runtime.')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('This training runtime requires a CUDA GPU with bfloat16 support (Ampere or newer).')
    pl.seed_everything(request['seed'], workers=True)
    torch.set_float32_matmul_precision('high')
    rows = training_rows(Path(request['dataset_dir']))
    manifest = read_json(Path(request['dataset_dir']) / 'dataset.json')
    output = Path(request['output_dir']); output.mkdir(parents=True, exist_ok=False)
    emit(event='log', message='Loading Small-SFX base model and text encoder. First use downloads weights.')
    audio_model = StableAudioModel.from_pretrained(BASE_MODEL, device='cuda', model_half=False)
    model = audio_model.model.eval().requires_grad_(False)
    model.pretransform.enable_grad = False
    ratio = model.pretransform.downsampling_ratio
    sample_size = math.ceil(manifest['clip_seconds'] * model.sample_rate / ratio) * ratio

    class Clips(torch.utils.data.Dataset):
        def __len__(self): return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            values, rate = read_pcm(Path(row['path']))
            audio = torch.from_numpy(values)
            if rate != model.sample_rate:
                audio = torchaudio.functional.resample(audio, rate, model.sample_rate)
            if audio.shape[0] == 1: audio = audio.repeat(2, 1)
            count = min(audio.shape[-1], sample_size)
            audio = torch.nn.functional.pad(audio[:, :sample_size], (0, max(0, sample_size - audio.shape[-1])))
            mask = torch.arange(sample_size) < count
            return audio, dict(prompt=row['caption'], seconds_total=count / model.sample_rate,
                               seconds_start=0, padding_mask=[mask])

    def collate(batch):
        audio, metadata = zip(*batch)
        return torch.stack(audio), list(metadata)

    lora_config = dict(rank=request['rank'], alpha=request['rank'], adapter_type='lora',
                       dropout=0., include=['transformer.layers'], exclude=None)
    state = None; previous_epochs = 0
    if request.get('resume_from'):
        previous = checkpoint_info(Path(request['resume_from']))
        previous_epochs = previous.get('epochs_completed', 0)
        state, lora_config = load_lora_checkpoint(str(Path(request['resume_from']) / 'adapter.safetensors'))
    wrapper = DiffusionCondTrainingWrapper(model, mask_loss_weight=1., mask_padding_attention=True,
        silence_extension_scale_seconds=4., use_ema=False, log_loss_info=False,
        optimizer_configs={'diffusion': {'optimizer': {'type': 'AdamW', 'config': {
            'lr': request['learning_rate'], 'weight_decay': .01, 'betas': [.9, .95]}}}},
        timestep_sampler='trunc_logit_normal', timestep_sampler_options={},
        inpainting_config={'mask_kwargs': {'mask_type_probabilities': [.1, .8, .1]}},
        use_effective_length_for_schedule=True, sample_rate=model.sample_rate, sample_size=sample_size,
        lora_config=lora_config, lora_state_dict=state, log_every_n_steps=1, ot_coupling=True, base_precision='bf16')
    parameters = [p for p in wrapper.parameters() if p.requires_grad]
    if not parameters: raise ValueError('No LoRA parameters were enabled for this model.')
    unexpected = [name for name, p in wrapper.named_parameters() if p.requires_grad and not name_is_lora(name)]
    if unexpected:
        raise ValueError('Non-adapter parameters were unexpectedly trainable: ' + ', '.join(unexpected[:3]))
    emit(event='log', message=f'Training {sum(p.numel() for p in parameters):,} adapter parameters; base weights frozen.')
    info = dict(name=request['model_name'], architecture=ARCHITECTURE, base_model=BASE_MODEL,
                upstream_revision=UPSTREAM_REVISION, sample_rate=44100, channels=2,
                clip_seconds=manifest['clip_seconds'], dataset_dir=request['dataset_dir'],
                rank=lora_config['rank'], epochs_completed=previous_epochs, steps_completed=0,
                training_settings={key: request[key] for key in ('epochs', 'batch_size', 'learning_rate', 'rank', 'seed', 'save_every')},
                history=[],
                trainable_parameters=sum(p.numel() for p in parameters), base_frozen=True,
                created_at=datetime.now(timezone.utc).isoformat(), continuation='adapter weights; fresh optimizer')
    batches = math.ceil(len(rows) / request['batch_size'])

    class Progress(pl.Callback):
        def __init__(self):
            self.completed = 0

        def save(self, trainer, module):
            temporary = output / 'adapter.tmp.safetensors'
            module.export_lora_safetensors(str(temporary)); temporary.replace(output / 'adapter.safetensors')
            info.update(epochs_completed=previous_epochs + self.completed, steps_completed=int(trainer.global_step),
                        peak_vram_bytes=torch.cuda.max_memory_allocated(), stopped=stopped(request))
            atomic_json(output / 'audio_model.json', info)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = float(outputs['loss'].detach()) if isinstance(outputs, dict) else float(outputs.detach())
            if not math.isfinite(loss): raise ValueError('Training loss is not finite; reduce the learning rate.')
            emit(event='step', current=trainer.global_step, total=batches * request['epochs'], loss=loss)
            wait_if_paused(request)
            if stopped(request):
                self.save(trainer, module); trainer.should_stop = True

        def on_train_epoch_end(self, trainer, module):
            # A stopped partial epoch does not count as a completed epoch.
            if not stopped(request): self.completed = trainer.current_epoch + 1
            loss = float(trainer.callback_metrics.get('train/loss', 0))
            info['history'].append(dict(epoch=previous_epochs + self.completed, loss=loss, step=int(trainer.global_step)))
            emit(event='epoch', epoch=self.completed, total_epochs=request['epochs'], loss=loss)
            if self.completed and self.completed % request['save_every'] == 0:
                self.save(trainer, module)
                checkpoint = output / f'epoch_{previous_epochs + self.completed:06d}.safetensors'
                import shutil
                shutil.copy2(output / 'adapter.safetensors', checkpoint)

        def on_train_end(self, trainer, module): self.save(trainer, module)

    torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision='bf16-mixed', logger=False,
        callbacks=[Progress()], max_epochs=request['epochs'], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        gradient_clip_val=1., default_root_dir=str(output))
    if stopped(request): raise InterruptedError('Stopped before training.')
    trainer.fit(wrapper, torch.utils.data.DataLoader(Clips(), batch_size=request['batch_size'],
                shuffle=True, num_workers=0, drop_last=False, collate_fn=collate))
    emit(event='done', output_dir=str(output))
