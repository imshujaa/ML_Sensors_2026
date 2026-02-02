#!/usr/bin/env python3
"""Generate sample data for ML Sensor Person Detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_sensor.data import create_sample_dataset

if __name__ == "__main__":
    print("Generating sample dataset...")
    create_sample_dataset(output_dir="data/sample", num_samples=200)
    print("Sample dataset generation complete!")
