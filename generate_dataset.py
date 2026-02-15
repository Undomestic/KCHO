"""
Standalone script to generate synthetic patient datasets and export as JSON.
This script uses the same ODE models as the main KCTHO project but does not
perform any analysis – it only outputs raw time series data.
"""

import json
import numpy as np
import argparse
import time
import os
import sys

try:
    from data_generation import generate_data_for_condition, get_default_params
    import config
except ImportError as e:
    print("Error: This script must be run in the same directory as data_generation.py and config.py")
    print(f"Import error: {e}")
    sys.exit(1)

def generate_dataset(condition, n_samples, output_file, seed=None):
    """
    Generate a dataset of synthetic patient trajectories.

    Parameters:
        condition: str, medical condition (e.g., 'sepsis', 'cardiac arrest', 'hemorrhage')
        n_samples: int, number of patients to generate
        output_file: str, path to output JSON file
        seed: int, optional base seed for reproducibility (if None, uses system time)
    """
    dataset = []
    base_seed = seed if seed is not None else int(time.time())

    for i in range(n_samples):
        patient_seed = base_seed + i * 1000
        raw_data = generate_data_for_condition(condition, seed=patient_seed, vary=True)


        patient_data = {
            'patient_id': i,
            'condition': condition,
            'seed_used': patient_seed,
            'time': raw_data['time'].tolist(),
            'IL6': raw_data['IL6'].tolist(),
            'HRV': raw_data['HRV'].tolist(),
            'MAP': raw_data['MAP'].tolist(),
            'Lactate': raw_data['Lactate'].tolist(),
            'NET': raw_data['NET'].tolist(),
            'ScvO2': raw_data['ScvO2'].tolist()
        }
        dataset.append(patient_data)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {n_samples} samples for condition '{condition}'.")
    print(f"Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic patient dataset as JSON.')
    parser.add_argument('--condition', type=str, default='sepsis',
                        help='Medical condition (sepsis, cardiac arrest, hemorrhage)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of patient trajectories to generate')
    parser.add_argument('--output', type=str, default='dataset.json',
                        help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base random seed (optional)')
    args = parser.parse_args()

    generate_dataset(args.condition, args.n_samples, args.output, args.seed)

if __name__ == '__main__':
    main()