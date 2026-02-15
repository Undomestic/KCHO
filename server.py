from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

import config
from data_generation import generate_data_for_condition, smooth_all_variables, get_default_params
from topology import compute_persistence, betti_numbers_at_scale, persistent_entropy
from kan import fit_kan_edge_bspline, most_complex_variable
from math_utils import (
    lyapunov_spectrum, correlation_dimension, approximate_entropy,
    ricci_curvature, box_counting_dimension, compute_health_score
)
from clinical import (
    interpret_lyapunov, generate_diagnosis, enhanced_report, generate_easy_explanations
)
from treatment_simulator import run_simulation
from example_loader import get_examples, find_closest_example

app = Flask(__name__)
CORS(app)

def sanitize(obj):
    """Convert numpy types to standard python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj

@app.route('/api/run', methods=['POST'])
def run_analysis():
    data = request.json
    condition = data.get('condition', 'sepsis')
    seed = data.get('seed', None)
    if seed == 0: seed = None
    use_gemini = data.get('use_gemini', False)
    
    # Update config
    config.USE_GEMINI = use_gemini

    # Generate
    raw_data = generate_data_for_condition(condition, seed=seed, vary=True)
    smoothed_data = smooth_all_variables(raw_data)
    
    # Analysis
    point_cloud = np.column_stack([smoothed_data[var] for var in config.PHYSIO_VARS])
    diagrams = compute_persistence(point_cloud)
    betti = betti_numbers_at_scale(diagrams, scale=config.BETTI_SCALE_THRESHOLD)
    entropy_h1 = persistent_entropy(diagrams[1]) if len(diagrams) > 1 else 0.0
    
    lyap_spec = lyapunov_spectrum(smoothed_data['MAP'],
                                emb_dim=config.LYAPUNOV_EMBEDDING_DIM,
                                matrix_dim=config.LYAPUNOV_MATRIX_DIM)
    lyap_max = lyap_spec[0]
    
    corr_dim = correlation_dimension(point_cloud)
    apen = approximate_entropy(smoothed_data['MAP'])
    box_dim = box_counting_dimension(point_cloud)
    
    curvatures = ricci_curvature(point_cloud)
    mean_curv = np.mean(curvatures)
    
    var_pairs = [(v, 'MAP') for v in config.PHYSIO_VARS if v != 'MAP']
    most_var, max_comp, all_comps = most_complex_variable(smoothed_data, var_pairs, method='curvature')
    
    health_score = compute_health_score(
        betti, lyap_max, entropy_h1, corr_dim, apen, box_dim, mean_curv, max_comp
    )
    
    diagnosis = generate_diagnosis(betti, lyap_max, most_var, mean_curv)
    
    results = {
        'betti': betti,
        'entropy_h1': entropy_h1,
        'lyap_max': lyap_max,
        'corr_dim': corr_dim,
        'apen': apen,
        'box_dim': box_dim,
        'mean_curv': mean_curv,
        'most_complex_var': most_var,
        'max_complex': max_comp,
        'all_comps': all_comps,
        'diagnosis': diagnosis,
        'lyap_interp': interpret_lyapunov(lyap_max),
        'health_score': health_score
    }
    
    # KAN Data for plotting
    x_kan = smoothed_data[most_var]
    y_kan = smoothed_data['MAP']
    spline = fit_kan_edge_bspline(x_kan, y_kan)
    x_grid = np.linspace(min(x_kan), max(x_kan), config.KAN_N_GRID)
    y_spline = spline(x_grid)

    # Report
    report_text = enhanced_report(results, condition)
    
    # Gemini
    gemini_text = ""
    if use_gemini:
        last_vitals = {
            'map': smoothed_data['MAP'][-1],
            'lactate': smoothed_data['Lactate'][-1],
            'hrv': smoothed_data['HRV'][-1],
            'scvo2': smoothed_data['ScvO2'][-1]
        }
        examples = get_examples()
        closest_examples = find_closest_example(last_vitals, examples, n=3) if examples else None
        metrics_dict_gemini = {
            **results,
            'hrv_value': np.mean(smoothed_data['HRV']),
            'map_value': np.mean(smoothed_data['MAP']),
            'il6_value': np.mean(smoothed_data['IL6']),
        }
        gemini_text = generate_easy_explanations(results, condition, metrics_dict_gemini, closest_examples)

    response = {
        'raw_data': raw_data,
        'smoothed_data': smoothed_data,
        'results': results,
        'curvatures': curvatures,
        'kan_data': {
            'x': x_kan,
            'y': y_kan,
            'x_grid': x_grid,
            'y_spline': y_spline,
            'xlabel': most_var,
            'ylabel': 'MAP'
        },
        'report': report_text,
        'gemini_explanation': gemini_text,
        'physio_vars': config.PHYSIO_VARS
    }
    
    return jsonify(sanitize(response))

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    condition = data.get('condition')
    treatment = data.get('treatment')
    seed = data.get('seed', None)
    if seed == 0: seed = None
    
    base_params = get_default_params(condition, vary=True)
    treatment_results = run_simulation(condition, base_params, treatment, seed=seed)
    
    return jsonify(sanitize(treatment_results))

if __name__ == '__main__':
    print("Starting KCTHO Backend Server on http://localhost:5000")
    app.run(debug=True, port=5000)