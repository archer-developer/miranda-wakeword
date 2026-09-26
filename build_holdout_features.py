"""Builds spectrogram features for holdout_real_fresh the same way training
positives were built, so evaluation is apples-to-apples with the trained model."""
import os

import mww_patches  # noqa: F401
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

os.chdir(os.path.dirname(os.path.abspath(__file__)))

clips = Clips(
    input_directory="holdout_real_fresh",
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=None,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "Gain": 1.0,
    },
    impulse_paths=[],
    background_paths=[],
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)

output_dir = "holdout_real_features"
os.makedirs(output_dir, exist_ok=True)
out_dir = os.path.join(output_dir, "testing")
os.makedirs(out_dir, exist_ok=True)

spectrograms = SpectrogramGeneration(
    clips=clips,
    augmenter=augmenter,
    slide_frames=1,
    step_ms=10,
)

RaggedMmap.from_generator(
    out_dir=os.path.join(out_dir, "wakeword_mmap"),
    sample_generator=spectrograms.spectrogram_generator(split=None, repeat=1),
    batch_size=100,
    verbose=True,
)
print("DONE")
