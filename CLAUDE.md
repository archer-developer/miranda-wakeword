# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A training pipeline for a custom Russian wake word, **"Миранда"**, targeting
ESP32-class devices via [ESPHome](https://esphome.io) / Home Assistant Voice.
It is not a library with tests to run — it's a working directory of scripts,
generated datasets, and training runs built on top of two vendored git
projects:

- `micro-wake-word/` — Kevin Ahrendt's [microWakeWord](https://github.com/kahrendt/microWakeWord)
  training framework (installed editable as the `microwakeword` package).
  Produces TFLite streaming models for TFLite-Micro.
- `piper-sample-generator/` — [Piper sample generator](https://github.com/rhasspy/piper-sample-generator)
  (installed editable as `piper_sample_generator`), used to synthesize TTS
  utterances of the wake word for positive training samples.

Top-level `.py` files are this project's own glue scripts around those two
packages, tailored to "Миранда" specifically (mirroring cells from
microWakeWord's `basic_training_notebook.ipynb`).

The environment is a local `.venv` (Python 3.11); there is no separate
requirements file — `microwakeword` and `piper_sample_generator` are
installed in editable mode from their respective subdirectories.

## Common commands

Activate the venv first: `source .venv/bin/activate`.

### 1. Download negative/augmentation audio (one-time, populates `mit_rirs/`, `audioset_16k/`, `fma/`, `fma_16k/`)
```bash
python download_augmentation_audio.py
```

### 2. Generate positive samples (TTS synthesis of "Миранда")
Use `piper_sample_generator` directly against the Russian voices in `voices/`
(e.g. `ru_RU-denis-medium.onnx`, `ru_RU-irina-medium.onnx`) — see
`piper-sample-generator/README.md` for full CLI flags:
```bash
python3 -m piper_sample_generator 'Миранда' --model voices/ru_RU-denis-medium.onnx \
    --max-samples 100 --output-dir synthetic_samples/
```

### 2b. Generate adversarial negatives (TTS of similar-sounding words)
`adversarial_phrases_ru.txt` lists words that share "Миранда"'s stress
pattern or ending (Марина, веранда, команда, ...). Synthesize each one into
`adversarial_samples/` with every available voice (Piper's Russian voices are
single-speaker, so also consider multi-voice TTS such as the Gemini voices
`miranda-tts` already uses), then build features with
`build_negative_features.py` (step 4b):
```bash
grep -v '^#' adversarial_phrases_ru.txt | while read -r phrase; do
  for v in voices/ru_RU-*.onnx; do
    python3 -m piper_sample_generator "$phrase" --model "$v" \
        --max-samples 20 --output-dir "adversarial_samples/$(basename "$v" .onnx)_${phrase// /_}"
  done
done
```

### 3. Slice multi-word real recordings into individual clips
When real recordings contain several repetitions of "Миранда" separated by
pauses, use WebRTC VAD to split them:
```bash
python slice_multi_word.py multi_word_raw/ multi_word_sliced/ \
    [--min-silence-ms 400] [--min-segment-ms 200] [--pad-ms 150] [--vad-aggressiveness 2]
```

### 4. Build spectrogram features (RaggedMmap) from positive samples
Builds `training`/`validation`/`testing` splits under the output dir. Run
once for synthetic samples and once for real recordings, so real samples can
be given a higher `sampling_weight` during training despite being scarcer:
```bash
python build_positive_features.py positive_samples generated_augmented_features
python build_positive_features.py positive_samples_real generated_augmented_features_real
```

### 4b. Build negative features from our own audio
Record what actually triggers the device — Russian TV, household
conversation, music — ideally through the Voice PE itself or at least in the
same room, plus the snippets dumped by `eval_false_accepts.py --dump-dir`
(step 7b). These are the negatives the stock `negative_datasets/*` lack:
```bash
python build_negative_features.py negative_recordings/ negative_features_home --eval-fraction 0.2
python build_negative_features.py adversarial_samples/ negative_features_adversarial
```
`--eval-fraction` holds files out into `validation_ambient`/`testing_ambient`,
so the FA/h metric that picks the best checkpoint during training is measured
on our own environment too. Enable the commented entries in
`training_parameters.yaml` once the dirs exist.

### 5. Build holdout evaluation features
Builds features for `holdout_real_fresh/` the same way as training positives,
for apples-to-apples evaluation of the trained model:
```bash
python build_holdout_features.py
```

### 6. Train
Training is driven by `microwakeword.model_train_eval`, configured via a
YAML file (see `training_parameters.yaml` / `training_parameters_real_only.yaml`
for the two datasets compared in this repo — full mix vs. real-recordings-only).
It requires a model subcommand (`mixednet` or `inception`); this project's
trained models use `mixednet`:
```bash
python -m microwakeword.model_train_eval \
    --training_config training_parameters.yaml \
    --train 1 --test_tflite_streaming_quantized 1 \
    mixednet
```
Key knobs live in the YAML: per-dataset `sampling_weight`/`penalty_weight`
(oversampling and error-weighting), `positive_class_weight` /
`negative_class_weight`, `training_steps`, `learning_rates`, and the
`minimization_metric` / `maximization_metric` pair that selects best weights
(this repo sets no minimization metric and maximizes `average_viable_recall`,
which is recall averaged over the 0–2 FA/h region of the *ambient* sets —
only `dinner_party_eval`, English, unless our own `negative_features_home`
is enabled). Output (checkpoints, TFLite models, logs) goes to `train_dir`
(e.g. `trained_models/miranda_v2/`). Training restores the latest
checkpoint from `<train_dir>/restore/`, so use a fresh `train_dir` for a
run that should start from scratch.

### 7. Sanity-check a trained model against holdout recordings
```bash
python check_holdout.py
```
Edit the `model_path` at the top of the script to point at a different
`trained_models/<name>/tflite_stream_state_internal_quant/*.tflite`.

### 7b. Measure false accepts per hour and pick the device threshold
`check_holdout.py` only looks at positives. `eval_false_accepts.py` runs the
streaming model over long negative recordings (that were *not* used for
training) and the holdout positives, applying the device's rule (mean of the
last `sliding_window_size` probabilities ≥ `probability_cutoff`), and prints
FA/h and recall for a grid of both:
```bash
python eval_false_accepts.py --model miranda.tflite \
    --negatives negative_recordings_eval/ --positives holdout_real_fresh/ \
    --report-cutoff 0.9 --report-window 5 --dump-dir fa_dumps/
```
Pick the lowest cutoff whose FA/h is acceptable (target well under 1/h) and
write it into the manifest. `--dump-dir` saves a snippet around every false
accept — listen to them, then feed them back into step 4b as hard negatives.

### 8. Package a model for ESPHome/Home Assistant
The trained `stream_state_internal_quant.tflite` is copied to the repo root
(e.g. `miranda.tflite`) alongside a manifest JSON (`miranda.json`) describing
it (`type: micro`, `wake_word`, `probability_cutoff`, `sliding_window_size`,
`tensor_arena_size`, `minimum_esphome_version`) — this is the ESPHome
`micro_wake_word` manifest format. This repo maintains two such pairs
(`miranda.*` and `miranda_real_only.*`) to compare the full-mix vs.
real-only trained models.

## Architecture / data flow

```
voices/*.onnx (Piper TTS)         real recordings
        │                          multi_word_raw*/
        ▼                                │
  piper_sample_generator          slice_multi_word.py (WebRTC VAD)
        │                                │
        ▼                                ▼
 synthetic_samples/,             positive_samples_real/,
 positive_samples/                multi_word_sliced*/
        │                                │
        └──────────► build_positive_features.py ◄──────┘
                     (Augmentation + SpectrogramGeneration)
                              │
                              ▼
        generated_augmented_features/ , generated_augmented_features_real/
        (RaggedMmap: training/validation/testing splits)
                              │
   negative_datasets/{speech,dinner_party,no_speech,dinner_party_eval}/
   (pre-built RaggedMmap negative feature sets, mixed in via training_parameters.yaml)
                              │
                              ▼
                 python -m microwakeword.model_train_eval (mixednet)
                 config: training_parameters*.yaml
                              │
                              ▼
             trained_models/<name>/tflite_stream_state_internal_quant/*.tflite
                              │
                              ▼
             check_holdout.py  ◄── holdout_real_features/ ◄── build_holdout_features.py ◄── holdout_real_fresh/
                              │
                              ▼
                 miranda*.tflite + miranda*.json  (ESPHome manifest)
```

Key points to understand before touching this pipeline:

- **Two-stage detection model** (from microWakeWord): a fixed audio
  preprocessor turns 16kHz mono audio into 40-value spectrogram features
  every 10ms (`micro_speech`/TFLite-Micro preprocessor, includes noise
  suppression + AGC), then a small streaming MixConv neural net predicts a
  wake-word probability per 30ms window. Models train in non-streaming mode
  (full spectrogram at once) and are then converted to a streaming model
  that only recomputes on the newest feature slice — this is why
  non-streaming training results transfer to streaming inference.
- **`mww_patches.py`** is a required compatibility shim (macOS
  `pymicro-features` fork exposes `ProcessSamples` instead of
  `process_samples`) — every script that touches `microwakeword.audio.*` or
  `microwakeword.inference` imports it first purely for its monkeypatch
  side effect. If you add a new script that generates features or runs
  inference, import `mww_patches` before importing `microwakeword`.
  `os.chdir(...)` to the script's own directory at the top of these scripts
  is intentional (paths in configs are relative to the repo root).
  Note: `check_holdout.py` uses `microwakeword.inference.Model` directly
  rather than these audio-pipeline modules, but still needs the same patch
  since `Model` also calls into `pymicro_features`.
- **Real vs. synthetic positive samples are kept in separate feature dirs**
  on purpose — real recordings are far fewer but more representative, so
  `training_parameters.yaml` gives `generated_augmented_features_real` a
  much higher `sampling_weight` (5.0) than the synthetic set (2.0) to
  balance their influence.
- **Negative datasets have different roles**: `speech`, `dinner_party`, and
  `no_speech` are used for training (nonzero `sampling_weight`);
  `dinner_party_eval` has `sampling_weight: 0.0` and
  `truncation_strategy: split` — it's held out purely for evaluation, not
  sampled during training.
- **Two parallel model variants are maintained** in this repo:
  `training_parameters.yaml` (full mix of synthetic + real positives) vs.
  `training_parameters_real_only.yaml` (drops the synthetic feature set) —
  compare `trained_models/miranda/` vs. `trained_models/miranda_real_only/`
  and the corresponding `train.log`/`train2.log` vs.
  `train3_realonly.log`/`train3_combined.log` to see how they were tuned.
- The many `resplit_out*/`, `one_file_resplit*/`, `holdout_real*/`
  directories reflect iterative experiments with how positive samples were
  split/resplit into train/validation/test and holdout sets — check
  filenames/timestamps rather than assuming any one is canonical when
  reusing this pattern.
