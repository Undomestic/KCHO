"""
Loads and parses the provided CSV datasets (smolify.csv, smolify 2.csv, smolify 3.csv).
Each row contains a system prompt, a user input string with vitals, and an assistant response.
We extract the vitals (MAP, Lactate, HRV, ScvO2) and the diagnosis/warning.
Provides a function to find the closest example to a given set of vitals.
"""

import csv
import re
import json
import numpy as np
import os

def parse_vitals(vitals_str):
    """
    Extract MAP, Lactate, HRV, ScvO2 from a string like:
    "MAP: 72 mmHg, Lactate: 1.8 mmol/L, HRV: 65 ms, ScvO2: 70%"
    Returns a dict with keys: map, lactate, hrv, scvo2 (as floats).
    If a value is missing, returns None for that key.
    """
    vitals = {}
    match = re.search(r'MAP:\s*(\d+(?:\.\d+)?)', vitals_str, re.IGNORECASE)
    vitals['map'] = float(match.group(1)) if match else None
    match = re.search(r'Lactate:\s*(\d+(?:\.\d+)?)', vitals_str, re.IGNORECASE)
    vitals['lactate'] = float(match.group(1)) if match else None
    match = re.search(r'HRV:\s*(\d+(?:\.\d+)?)', vitals_str, re.IGNORECASE)
    vitals['hrv'] = float(match.group(1)) if match else None
    match = re.search(r'ScvO2:\s*(\d+(?:\.\d+)?)', vitals_str, re.IGNORECASE)
    vitals['scvo2'] = float(match.group(1)) if match else None
    return vitals

def parse_assistant_response(assistant_str):
    """
    The assistant column may contain plain text or a JSON string.
    We try to extract diagnosis and warning.
    Returns a dict with 'diagnosis' and 'warning'.
    """
    try:
        data = json.loads(assistant_str)
        diagnosis = data.get('diagnosis', '')
        warning = data.get('warning', '')
        return {'diagnosis': diagnosis, 'warning': warning}
    except:
        diag_match = re.search(r'Diagnosis:\s*(.*?)(?:\n|$)', assistant_str, re.IGNORECASE)
        inter_match = re.search(r'Intervention:\s*(.*?)(?:\n|$)', assistant_str, re.IGNORECASE)
        diagnosis = diag_match.group(1).strip() if diag_match else ''
        warning = inter_match.group(1).strip() if inter_match else ''
        return {'diagnosis': diagnosis, 'warning': warning}

def load_examples(csv_files):
    """
    Load all examples from the given CSV files.
    Returns a list of dicts with keys: vitals (dict), diagnosis, warning.
    """
    examples = []
    for file in csv_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping.")
            continue
        print(f"Loading examples from {file}...")
        with open(file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # start at 2 for header
                if 'user' not in row or 'assistant' not in row:
                    print(f"  Skipping row {row_num}: missing 'user' or 'assistant' column")
                    continue
                vitals_str = row['user']
                assistant_str = row['assistant']
                vitals = parse_vitals(vitals_str)
                if None in vitals.values():
                    missing = [k for k, v in vitals.items() if v is None]
                    print(f"  Skipping row {row_num}: missing vitals {missing}")
                    continue
                resp = parse_assistant_response(assistant_str)
                examples.append({
                    'vitals': vitals,
                    'diagnosis': resp['diagnosis'],
                    'warning': resp['warning']
                })
    print(f"Loaded {len(examples)} examples total.")
    return examples

def normalize_vitals(vitals, means, stds):
    """Normalize vitals using given mean and std."""
    norm = {}
    for key in vitals:
        if stds[key] > 0:
            norm[key] = (vitals[key] - means[key]) / stds[key]
        else:
            norm[key] = 0.0
    return norm

def find_closest_example(current_vitals, examples, n=1):
    """
    Find the closest example(s) to the current vitals using Euclidean distance
    on normalized vitals. Returns list of (distance, example) sorted.
    """
    if not examples:
        return []
    all_vals = {key: [] for key in ['map', 'lactate', 'hrv', 'scvo2']}
    for ex in examples:
        for key in all_vals:
            all_vals[key].append(ex['vitals'][key])
    means = {key: np.mean(all_vals[key]) for key in all_vals}
    stds = {key: np.std(all_vals[key]) for key in all_vals}
    curr_norm = normalize_vitals(current_vitals, means, stds)
    distances = []
    for ex in examples:
        ex_norm = normalize_vitals(ex['vitals'], means, stds)
        dist = np.sqrt(sum((curr_norm[k] - ex_norm[k])**2 for k in curr_norm))
        distances.append((dist, ex))
    distances.sort(key=lambda x: x[0])
    return distances[:n]

DEFAULT_FILES = ['smolify.csv', 'smolify 2.csv', 'smolify 3.csv']
_examples_cache = None

def get_examples(files=None):
    """Load examples once and cache."""
    global _examples_cache
    if _examples_cache is None:
        if files is None:
            files = DEFAULT_FILES
        _examples_cache = load_examples(files)
    return _examples_cache