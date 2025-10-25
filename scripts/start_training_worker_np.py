#!/usr/bin/env python3
"""Compatibility wrapper for the legacy NumPy training worker entrypoint."""

from snn.trainers.train_numpy import main


if __name__ == "__main__":
    main()
