#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, shutil, sys

DEFAULT_SRC = os.path.join("checkpoints", "np_trainer_best.npz")


def main():
    ap = argparse.ArgumentParser(description="Export the latest train_numpy checkpoint to a destination path.")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Source NPZ checkpoint (default: checkpoints/np_trainer_best.npz)")
    ap.add_argument("--out", required=True, help="Destination NPZ path")
    args = ap.parse_args()
    if not os.path.exists(args.src):
        print(f"Source checkpoint not found: {args.src}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy2(args.src, args.out)
    print(f"Exported: {args.out}")


if __name__ == "__main__":
    main()

