"""Sanity-checks the trained streaming model on the 5 held-out real recordings."""
import glob
import os

import mww_patches  # noqa: F401
import numpy as np
from scipy.io import wavfile

from microwakeword.inference import Model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model_path = "trained_models/miranda/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
model = Model(model_path)

for path in sorted(glob.glob("holdout_real_fresh/*.wav")):
    sr, data = wavfile.read(path)
    assert sr == 16000, f"unexpected sample rate {sr} for {path}"
    probs = model.predict_clip(data.astype(np.float32), step_ms=10)
    max_prob = max(probs) if probs else 0.0
    print(f"{os.path.basename(path)}: max_probability={max_prob:.4f} n_frames={len(probs)}")
