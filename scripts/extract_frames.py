#!/usr/bin/env python3
"""Extract sampled frames from a raw video into data/interim/."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.data.ingestion import extract_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a traffic camera video.")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/frames"),
        help="Output directory for extracted frames",
    )
    parser.add_argument("--every-n", type=int, default=30, help="Sample every Nth frame")
    args = parser.parse_args()
    saved = extract_frames(args.video, args.output, every_n=args.every_n)
    print(f"Saved {len(saved)} frames to {args.output}")


if __name__ == "__main__":
    main()
