"""Builds negative spectrogram features (RaggedMmap) from our own recordings:
Russian TV, household conversation, music played at home, false-accept
snippets dumped by eval_false_accepts.py, adversarial TTS phrases, ...

Why: the stock `negative_datasets/*` from microWakeWord are generic and
mostly non-Russian; they don't contain the sounds that actually trigger
the device at home, and they were never recorded through our mic/room. With
only ~15 real positives recorded at home, the model can learn "sounds like
our room/mic + speech" == wake word. Negatives from the same room/device
break that shortcut.

Layout written (what microwakeword.data.MmapFeatureGenerator expects):
    <output_dir>/training/negative_mmap              - sampled during training
    <output_dir>/validation_ambient/negative_mmap    - only with --eval-fraction > 0
    <output_dir>/testing_ambient/negative_mmap       - only with --eval-fraction > 0
The *_ambient sets are merged into the FA/h metric that picks the best
checkpoint during training, so holding out some home audio there makes
checkpoint selection reflect our real environment, not just the English
dinner_party_eval set. The split is per *file*, so cut long recordings into
pieces of a few minutes first if you only have one or two of them, e.g.:
    ffmpeg -i tv.wav -f segment -segment_time 300 -c copy tv_%03d.wav

Usage:
    python build_negative_features.py negative_recordings/ negative_features_home [--eval-fraction 0.2]
    python build_negative_features.py adversarial_samples/ negative_features_adversarial

Then add the output dir to training_parameters.yaml's `features` with
`truth: false`, `truncation_strategy: random` (see the commented entries
there).
"""
import argparse
import os

import mww_patches  # noqa: F401  (patches pymicro_features before microwakeword uses it)
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def write(out_dir, generator):
    os.makedirs(out_dir, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=os.path.join(out_dir, "negative_mmap"),
        sample_generator=generator,
        batch_size=50,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--pattern", default="**/*.wav", help="glob relative to input_dir (any format `datasets.Audio` decodes)")
    parser.add_argument(
        "--split-seconds", type=float, default=10.0,
        help="training spectrograms are cut into non-overlapping pieces of this length "
             "(training then crops a random clip_duration_ms window out of each)",
    )
    parser.add_argument(
        "--eval-fraction", type=float, default=0.0,
        help="fraction of files held out into validation_ambient/testing_ambient (half each)",
    )
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # No augmentation on purpose: these are already real recordings through
    # the target acoustics; adding RIR/background on top would move them away
    # from what the device actually hears.
    clips = Clips(
        input_directory=args.input_dir,
        file_pattern=args.pattern,
        remove_silence=False,
        random_split_seed=args.seed if args.eval_fraction > 0 else None,
        split_count=args.eval_fraction / 2 if args.eval_fraction > 0 else 0.1,
    )

    train_split = "train" if args.eval_fraction > 0 else None
    split_gen = SpectrogramGeneration(clips=clips, step_ms=10, split_spectrogram_duration_s=args.split_seconds)
    write(os.path.join(args.output_dir, "training"), split_gen.spectrogram_generator(split=train_split))
    print("done: training")

    if args.eval_fraction > 0:
        # Ambient sets keep whole-file spectrograms; training's FA/h evaluation
        # splits them itself (truncation_strategy="split").
        whole_gen = SpectrogramGeneration(clips=clips, step_ms=10)
        write(os.path.join(args.output_dir, "validation_ambient"), whole_gen.spectrogram_generator(split="validation"))
        print("done: validation_ambient")
        write(os.path.join(args.output_dir, "testing_ambient"), whole_gen.spectrogram_generator(split="test"))
        print("done: testing_ambient")

    print("DONE")


if __name__ == "__main__":
    main()
