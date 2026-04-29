#!/usr/bin/env python3
"""
Subsample a slam_data_saver output directory.

Reads timestamps.txt, selects frames by stride / max count, and writes a new
directory with renumbered files (000000, 000001, ...).

Hard-links files when source and destination are on the same filesystem;
falls back to copy otherwise.

Usage:
  python3 slam_subsample.py --input ./slam_output --output ./slam_sub --stride 5
  python3 slam_subsample.py --input ./slam_output --output ./slam_sub --max-frames 30
  python3 slam_subsample.py --input ./slam_output --output ./slam_sub --stride 3 --max-frames 20
"""

import argparse
import os
import shutil

import numpy as np


_FRAME_SUBDIRS = {
    "rgb":   ".png",
    "depth": ".npy",
    "poses": ".npy",
    "kp2d":  ".npy",
    "kp3d":  ".npy",
}
_SHARED_FILES = ["intrinsics.npy", "map_points_latest.npy"]


def _link_or_copy(src: str, dst: str):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_timestamps(input_dir: str):
    ts_path = os.path.join(input_dir, "timestamps.txt")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"timestamps.txt not found in {input_dir}")
    rows = []
    with open(ts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tag, ts = line.split()
                rows.append((tag, float(ts)))
    return rows


def subsample(input_dir: str, output_dir: str, stride: int, max_frames: int):
    rows = _load_timestamps(input_dir)
    selected = rows[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]

    print(f"Total frames: {len(rows)}  →  selected: {len(selected)} "
          f"(stride={stride}, max={max_frames or 'none'})")

    os.makedirs(output_dir, exist_ok=True)
    present_subdirs = [s for s in _FRAME_SUBDIRS if
                       os.path.isdir(os.path.join(input_dir, s))]
    for subdir in present_subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    ts_out = open(os.path.join(output_dir, "timestamps.txt"), "w")

    for new_idx, (tag, ts) in enumerate(selected):
        new_tag = f"{new_idx:06d}"
        for subdir, ext in _FRAME_SUBDIRS.items():
            if subdir not in present_subdirs:
                continue
            src = os.path.join(input_dir, subdir, f"{tag}{ext}")
            dst = os.path.join(output_dir, subdir, f"{new_tag}{ext}")
            if os.path.exists(src):
                _link_or_copy(src, dst)
        ts_out.write(f"{new_tag}  {ts:.9f}\n")

    ts_out.close()

    for fname in _SHARED_FILES:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"Done → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",      required=True, help="slam_data_saver output dir")
    parser.add_argument("--output",     required=True, help="Destination dir")
    parser.add_argument("--stride",     type=int, default=1, help="Keep every N-th frame (default 1)")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap at N frames (0 = no cap)")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")

    subsample(
        input_dir=os.path.abspath(args.input),
        output_dir=os.path.abspath(args.output),
        stride=args.stride,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
