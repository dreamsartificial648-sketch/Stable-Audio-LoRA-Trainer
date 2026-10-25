# Attribution and model terms

## ADAM

Dataset preparation (`sfx_tuner/audio.py`), adapter contracts (`sfx_tuner/contracts.py`),
and the training implementation (`sfx_tuner/training.py`) are adapted from ADAM.
Copyright (c) 2026 SyntheticMDProductions. Used under the MIT License, retained in LICENSE.
The remaining standalone application code is distributed under that same license.

## Stable Audio 3

The runtime installs Stability AI's [stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
at commit `77e0398c26f7d22d801ec1bb036307e8b231f741`.
Upstream code is MIT licensed, Copyright (c) 2026 Stability AI. It remains in a
separate dependency checkout with its own LICENSE; upstream code is not vendored here.

The app's MIT license does **not** relicense model weights or their components.
Review the [Small-SFX model card](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
the base model's terms, and the linked Stable Audio Community and Gemma terms
before downloading or using weights. Users accept those terms directly through
Hugging Face. No model weights, text encoders, access tokens, or datasets are bundled.

## Desktop dependencies

PySide6 / Qt is distributed under its own licensing terms, including LGPL/GPL or
commercial options. NumPy and the runtime packages retain their respective licenses.
This repository installs these packages rather than bundling an executable distribution.
FFmpeg is an external prerequisite and is not redistributed by this app.

This is an independent community tool derived from ADAM, not an official Stability AI product.

