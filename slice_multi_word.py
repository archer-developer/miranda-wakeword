"""Slices recordings containing multiple "Миранда" repetitions (separated by
pauses) into individual clips, using WebRTC VAD to find speech segments."""
import argparse
import glob
import os

import numpy as np
import webrtcvad
from scipy.io import wavfile

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono


def frames(data):
    n = len(data) // FRAME_BYTES
    for i in range(n):
        yield data[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]


def find_segments(data, sample_rate, vad, min_silence_ms, min_segment_ms):
    flags = [vad.is_speech(f, sample_rate) for f in frames(data)]

    min_silence_frames = min_silence_ms // FRAME_MS
    min_segment_frames = min_segment_ms // FRAME_MS

    segments = []
    seg_start = None
    silence_run = 0
    for i, is_speech in enumerate(flags):
        if is_speech:
            if seg_start is None:
                seg_start = i
            silence_run = 0
        else:
            if seg_start is not None:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    seg_end = i - silence_run + 1
                    if seg_end - seg_start >= min_segment_frames:
                        segments.append((seg_start, seg_end))
                    seg_start = None
                    silence_run = 0
    if seg_start is not None:
        seg_end = len(flags)
        if seg_end - seg_start >= min_segment_frames:
            segments.append((seg_start, seg_end))
    return segments


def slice_file(path, output_dir, vad, min_silence_ms, min_segment_ms, pad_ms, name_prefix=None):
    sr, data_arr = wavfile.read(path)
    assert sr == SAMPLE_RATE, f"{path}: unexpected sample rate {sr}"
    raw = data_arr.astype("int16").tobytes()

    segments = find_segments(raw, sr, vad, min_silence_ms, min_segment_ms)
    base = name_prefix or os.path.splitext(os.path.basename(path))[0]
    print(f"{base}: {len(segments)} segment(s)")

    pad_bytes = int(SAMPLE_RATE * pad_ms / 1000) * 2
    written = []
    for idx, (s, e) in enumerate(segments):
        start_byte = max(0, s * FRAME_BYTES - pad_bytes)
        end_byte = min(len(raw), e * FRAME_BYTES + pad_bytes)
        clip = raw[start_byte:end_byte]
        out_path = os.path.join(output_dir, f"{base}_seg{idx:02d}.wav")
        wavfile.write(out_path, SAMPLE_RATE, np.frombuffer(clip, dtype="int16"))
        dur = len(clip) / 2 / SAMPLE_RATE
        print(f"  -> {out_path} ({dur:.2f}s)")
        written.append(out_path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default="multi_word_raw")
    parser.add_argument("output_dir", nargs="?", default="multi_word_sliced")
    parser.add_argument("--min-silence-ms", type=int, default=400)
    parser.add_argument("--min-segment-ms", type=int, default=200)
    parser.add_argument("--pad-ms", type=int, default=150)
    parser.add_argument("--vad-aggressiveness", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vad = webrtcvad.Vad(args.vad_aggressiveness)

    total = 0
    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.wav"))):
        written = slice_file(
            path, args.output_dir, vad,
            args.min_silence_ms, args.min_segment_ms, args.pad_ms,
        )
        total += len(written)

    print(f"DONE: {total} clips written to {args.output_dir}/")


if __name__ == "__main__":
    main()
