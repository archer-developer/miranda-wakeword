"""Downloads background/RIR audio for positive-sample augmentation.
Mirrors microWakeWord's basic_training_notebook cell 4.
"""
import itertools as it
import os
import subprocess
from pathlib import Path

import datasets
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def write_decoder_wav(path, decoder):
    samples = decoder.get_all_samples()
    data = samples.data.numpy()
    if data.ndim == 2:
        data = data[0]
    scipy.io.wavfile.write(path, samples.sample_rate, (data * 32767).astype(np.int16))


# MIT environmental impulse responses (RIR)
output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    for i, row in enumerate(tqdm(rir_dataset, desc="mit_rirs")):
        write_decoder_wav(os.path.join(output_dir, f"rir_{i}.wav"), row["audio"])

# AudioSet balanced-train split (streamed; the old per-shard .tar layout was
# replaced with Parquet shards upstream, so pull N examples via streaming
# instead of downloading one whole .tar file).
output_dir = "./audioset_16k"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    audioset_dataset = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True
    )
    for i, row in enumerate(tqdm(it.islice(audioset_dataset, 4000), desc="audioset_16k", total=4000)):
        write_decoder_wav(os.path.join(output_dir, f"audioset_{i}.wav"), row["audio"])

# Free Music Archive (extra-small subset)
output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    fname = "fma_xs.zip"
    link = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/" + fname
    out_dir = f"fma/{fname}"
    subprocess.run(["wget", "-q", "-O", out_dir, link], check=True)
    subprocess.run(["unzip", "-q", fname], cwd=output_dir, check=True)

    output_dir = "./fma_16k"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fma_dataset = datasets.Dataset.from_dict(
        {"audio": [str(i) for i in Path("fma/fma_small").glob("**/*.mp3")]}
    )
    fma_dataset = fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for i, row in enumerate(tqdm(fma_dataset, desc="fma_16k")):
        write_decoder_wav(os.path.join(output_dir, f"fma_{i}.wav"), row["audio"])

print("DONE")
