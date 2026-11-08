"""Measures false accepts per hour (FA/h) and recall of a trained streaming
model the way the ESP32 actually decides, for a grid of
probability_cutoff x sliding_window_size values.

Why this exists: microWakeWord's own training-time metrics
(`ambient_false_positives_per_hour`, `average_viable_recall`) are computed on
the non-streaming model, per independent window, against the *English*
`dinner_party_eval` set. None of that resembles what triggers the device at
home (Russian TV, household speech, our own room/mic). This script runs the
exported streaming TFLite model over our own long negative recordings and
our own positive clips, applies the same "mean of the last N probabilities
>= cutoff" rule ESPHome's `micro_wake_word` uses, and prints a table to pick
`probability_cutoff`/`sliding_window_size` from data instead of guessing.

Optionally (`--dump-dir`) it writes a short WAV around every false accept at
the chosen operating point - these are the best possible hard negatives and
should go straight into the negative set for the next training run (see
build_negative_features.py).

Usage:
    python eval_false_accepts.py \
        --model miranda.tflite \
        --negatives negative_recordings/ \
        --positives holdout_real_fresh/ \
        [--cutoffs 0.5 0.7 0.8 0.85 0.9 0.95 0.97] [--windows 3 5 8 10] \
        [--report-cutoff 0.9 --report-window 5 --dump-dir fa_dumps/]

All audio must be 16 kHz mono 16-bit WAV. Convert anything else first, e.g.:
    ffmpeg -i tv.mkv -ac 1 -ar 16000 -sample_fmt s16 tv.wav
Long recordings are fine (an hour of audio is ~120k model invocations).
"""
import argparse
import glob
import os

import mww_patches  # noqa: F401  (patches pymicro_features before microwakeword uses it)
import numpy as np
from scipy.io import wavfile

from microwakeword.inference import Model

SAMPLE_RATE = 16000
STEP_MS = 10  # feature window step, must match training (window_step_ms)

# Silence fed before/after each positive clip: the streaming model keeps
# internal state between invocations, so without it a clip's score would
# depend on whichever file was processed before it.
POSITIVE_PAD_S = 1.0


def load_wav(path):
    sr, data = wavfile.read(path)
    if sr != SAMPLE_RATE or data.ndim != 1 or data.dtype != np.int16:
        raise SystemExit(
            f"{path}: need 16 kHz mono int16 WAV (got sr={sr}, shape={data.shape}, "
            f"dtype={data.dtype}); convert with: ffmpeg -i IN -ac 1 -ar 16000 -sample_fmt s16 OUT.wav"
        )
    return data


def list_wavs(directory):
    return sorted(glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True))


def predict(model, data):
    """Streaming probabilities, one per model invocation (stride * STEP_MS)."""
    return np.asarray(model.predict_clip(data, step_ms=STEP_MS), dtype=np.float32)


def window_mean(probs, window):
    """Mean over the last `window` probabilities at each step (ESPHome's
    sliding window). Index i covers probs[i:i+window]; shorter-than-window
    inputs yield nothing, exactly like the device right after a reset."""
    if len(probs) < window:
        return np.zeros(0, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(probs, kernel, mode="valid")


def detections(means, cutoff, refractory_frames):
    """Indices where the windowed mean crosses `cutoff`, collapsing anything
    within `refractory_frames` of the previous detection into one event - a
    single loud TV "Миранда"-like burst stays above the cutoff for several
    consecutive frames but only wakes the device once."""
    hits = np.flatnonzero(means >= cutoff)
    events = []
    last = -refractory_frames - 1
    for idx in hits:
        if idx - last > refractory_frames:
            events.append(int(idx))
            last = idx
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="streaming quantized .tflite")
    parser.add_argument("--negatives", required=True, help="dir with long recordings that contain NO wake word")
    parser.add_argument("--positives", help="dir with short clips that each contain the wake word once")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97])
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 8, 10])
    parser.add_argument(
        "--refractory-s", type=float, default=2.0,
        help="detections closer than this are counted once (approximates the device ignoring "
             "triggers while it is already listening)",
    )
    parser.add_argument("--report-cutoff", type=float, help="operating point to list/dump false accepts for")
    parser.add_argument("--report-window", type=int, default=5)
    parser.add_argument("--dump-dir", help="write a WAV snippet around each false accept at the report point")
    parser.add_argument("--dump-seconds", type=float, default=3.0)
    args = parser.parse_args()

    model = Model(args.model)
    frame_s = model.stride * STEP_MS / 1000.0  # seconds per model invocation
    refractory_frames = int(round(args.refractory_s / frame_s))

    # --- negatives: FA/h ------------------------------------------------------
    neg_files = list_wavs(args.negatives)
    if not neg_files:
        raise SystemExit(f"no *.wav under {args.negatives}")
    neg = []  # (path, audio, probs)
    total_s = 0.0
    for path in neg_files:
        data = load_wav(path)
        total_s += len(data) / SAMPLE_RATE
        neg.append((path, data, predict(model, data)))
    hours = total_s / 3600.0
    print(f"negatives: {len(neg)} file(s), {hours:.2f} h; model step {frame_s * 1000:.0f} ms")

    # --- positives: recall ----------------------------------------------------
    pos = []
    if args.positives:
        pad = np.zeros(int(POSITIVE_PAD_S * SAMPLE_RATE), dtype=np.int16)
        for path in list_wavs(args.positives):
            data = load_wav(path)
            pos.append((path, predict(model, np.concatenate([pad, data, pad]))))
        print(f"positives: {len(pos)} clip(s)")

    # --- grid -----------------------------------------------------------------
    header = "cutoff | " + " | ".join(f"win={w:<2} FA/h   recall" for w in args.windows)
    print()
    print(header)
    print("-" * len(header))
    for cutoff in args.cutoffs:
        cells = []
        for w in args.windows:
            fa = sum(len(detections(window_mean(p, w), cutoff, refractory_frames)) for _, _, p in neg)
            if pos:
                hit = sum(1 for _, p in pos if len(detections(window_mean(p, w), cutoff, refractory_frames)))
                recall = f"{hit / len(pos):6.1%}"
            else:
                recall = "     -"
            cells.append(f"{fa / hours:12.2f} {recall}")
        print(f"{cutoff:6.2f} | " + " | ".join(cells))

    # --- per-event report / hard-negative dump --------------------------------
    if args.report_cutoff is None:
        return
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    print(f"\nfalse accepts at cutoff={args.report_cutoff} window={args.report_window}:")
    half = int(args.dump_seconds * SAMPLE_RATE / 2)
    for path, data, probs in neg:
        means = window_mean(probs, args.report_window)
        for idx in detections(means, args.report_cutoff, refractory_frames):
            # End of the window that fired, in seconds from the start of the file.
            t = (idx + args.report_window) * frame_s
            print(f"  {os.path.relpath(path, args.negatives)} @ {int(t // 60):02d}:{t % 60:05.2f}  mean={means[idx]:.3f}")
            if args.dump_dir:
                center = int(t * SAMPLE_RATE)
                snippet = data[max(0, center - half): center + half]
                base = os.path.splitext(os.path.basename(path))[0]
                wavfile.write(os.path.join(args.dump_dir, f"{base}_{t:08.2f}.wav"), SAMPLE_RATE, snippet)


if __name__ == "__main__":
    main()
