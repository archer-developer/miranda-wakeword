# Miranda Wake Word

A custom Russian wake word ("**Миранда**") trained for use with
[ESPHome](https://esphome.io) / Home Assistant Voice satellites, built on top
of [microWakeWord](https://github.com/kahrendt/microWakeWord) and
[Piper sample generator](https://github.com/rhasspy/piper-sample-generator).

The pipeline synthesizes and records utterances of "Миранда", augments them
with background noise and room impulse responses, and trains a small
streaming TFLite model suitable for low-power microcontrollers (ESP32).

## Contents

- `miranda.tflite` / `miranda.json` — the trained model + ESPHome
  `micro_wake_word` manifest, trained on a mix of synthetic (TTS) and real
  recorded samples.
- `miranda_real_only.tflite` / `miranda_real_only.json` — the same model
  architecture trained using only real recordings (no synthetic TTS
  samples), for comparison.
- `training_parameters.yaml` / `training_parameters_real_only.yaml` —
  training configs for the two model variants above.
- Top-level Python scripts — the data preparation and training pipeline
  glue (sample slicing, feature extraction, holdout evaluation).

See [`CLAUDE.md`](CLAUDE.md) for a detailed breakdown of the pipeline
architecture and the exact commands used at each stage.

## Prerequisites

- Python 3.11
- Two vendored dependencies, cloned alongside this repo and installed in
  editable mode:
  - [microWakeWord](https://github.com/kahrendt/microWakeWord) → `micro-wake-word/`
  - [Piper sample generator](https://github.com/rhasspy/piper-sample-generator) → `piper-sample-generator/`

```bash
git clone https://github.com/kahrendt/microWakeWord micro-wake-word
git clone https://github.com/rhasspy/piper-sample-generator

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e micro-wake-word
pip install -e piper-sample-generator
```

## Using the model

Copy `miranda.tflite` and `miranda.json` into your Home Assistant
`custom_wake_words` (or ESPHome `micro_wake_word` component) configuration —
see the [ESPHome documentation](https://esphome.io/components/micro_wake_word/)
for how to point a voice satellite at a custom wake word model.

## Retraining / reproducing the pipeline

The full data preparation and training workflow (downloading negative audio,
generating synthetic samples with Piper, slicing real recordings, building
spectrogram features, training with microWakeWord, and evaluating against a
holdout set) is documented step by step in [`CLAUDE.md`](CLAUDE.md).

Note that the raw/generated datasets (recorded and synthetic samples,
downloaded background audio, extracted spectrogram features, training
checkpoints) are not included in this repository — only the pipeline code
and the final exported models are tracked.

## Credits

- [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin Ahrendt
- [Piper sample generator](https://github.com/rhasspy/piper-sample-generator) (Rhasspy / OHF Voice)
- Piper Russian voices from [piper-voices](https://huggingface.co/rhasspy/piper-voices)
