"""
Treatment simulation module.
Allows user to choose an intervention and see its effect on the physiological manifold.
"""

import numpy as np
from copy import deepcopy
import config
from data_generation import generate_data_for_condition, smooth_all_variables, get_default_params
from topology import (
    compute_persistence, betti_numbers_at_scale, persistent_entropy,
    dominant_homology
)
from kan import most_complex_variable
from math_utils import (
    lyapunov_spectrum, correlation_dimension, approximate_entropy,
    ricci_curvature, mean_curvature_trend, box_counting_dimension
)
from clinical import (
    interpret_lyapunov, interpret_betti, generate_diagnosis,
    generate_recommendations, generate_explanation
)
from clinical_interpretation import compare_to_reference, interpret_metric
from gemini_integration import generate_treatment_insight, generate_patient_friendly_summary

TREATMENT_EFFECTS = {
    'antibiotics': {
        'il6_rise_rate': 0.3,
        'net_rise_rate': 0.4,
        'map_drop_rate': 0.6,
        'lactate_rise_rate': 0.5,
        'hrv_collapse_rate': 0.7,
    },
    'oxygen': {
        'map_drop_rate': 0.5,
        'scvo2_dip_rate': -0.3,   # negative factor means increase (reducing dip)
        'lactate_rise_rate': 0.4,
        'hrv_collapse_rate': 0.8,
    },
    'fluids': {
        'map_drop_rate': 0.4,
        'hrv_collapse_rate': 0.6,
        'lactate_rise_rate': 0.5,
        'scvo2_dip_rate': -0.2,
    },
    'anti_inflammatories': {
        'il6_rise_rate': 0.2,
        'net_rise_rate': 0.3,
        'map_drop_rate': 0.7,
        'lactate_rise_rate': 0.6,
    }
}

def apply_treatment(base_params, treatment):
    """
    Modify a copy of the base ODE parameters according to the chosen treatment.
    Returns a new parameter dictionary.
    """
    if treatment not in TREATMENT_EFFECTS:
        raise ValueError(f"Unknown treatment: {treatment}")
    effects = TREATMENT_EFFECTS[treatment]
    new_params = deepcopy(base_params)
    for key, factor in effects.items():
        if key in new_params:
            if isinstance(factor, float):
                new_params[key] = new_params[key] * factor
            elif isinstance(factor, tuple) and len(factor) == 2:
                pass
    return new_params

def run_simulation(condition, base_params, treatment, seed=None):
    """
    Run the full analysis for a given treatment.
    Returns a dictionary of results.
    """
    treated_params = apply_treatment(base_params, treatment)

    raw_data = generate_data_for_condition(condition, params=treated_params, seed=seed, vary=True)
    smoothed_data = smooth_all_variables(raw_data)

    hrv_mean = np.mean(smoothed_data['HRV'])
    map_mean = np.mean(smoothed_data['MAP'])
    il6_mean = np.mean(smoothed_data['IL6'])

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

    var_pairs = [(v, 'MAP') for v in config.PHYSIO_VARS if v != 'MAP']
    most_var, max_comp, all_comps = most_complex_variable(smoothed_data, var_pairs, method='curvature')

    curvatures = ricci_curvature(point_cloud)
    mean_curv = np.mean(curvatures)
    curv_trend = mean_curvature_trend(curvatures)

    dom_hom = dominant_homology(betti)

    lyap_interp = interpret_lyapunov(lyap_max)
    betti_interp = interpret_betti(betti)
    diagnosis = generate_diagnosis(betti, lyap_max, most_var, mean_curv)

    return {
        'condition': condition,
        'treatment': treatment,
        'betti': betti,
        'entropy_h1': entropy_h1,
        'lyap_max': lyap_max,
        'lyap_interp': lyap_interp,
        'corr_dim': corr_dim,
        'apen': apen,
        'box_dim': box_dim,
        'most_complex_var': most_var,
        'max_complex': max_comp,
        'mean_curv': mean_curv,
        'curv_trend': curv_trend,
        'dom_hom': dom_hom,
        'diagnosis': diagnosis,
        'betti_interp': betti_interp,
        'hrv_mean': hrv_mean,
        'map_mean': map_mean,
        'il6_mean': il6_mean
    }

def print_comparison(initial_results, treatment_results):
    """Print a side-by-side comparison of metrics before and after treatment."""
    print("\n" + "="*70)
    print(f"TREATMENT SIMULATION: {treatment_results['treatment'].replace('_', ' ').title()}")
    print("="*70)
    print(f"{'Metric':<25} {'Before':>15} {'After':>15} {'Change':>15}")
    print("-"*70)

    b0_b, b1_b, b2_b = initial_results['betti']
    b0_a, b1_a, b2_a = treatment_results['betti']
    print(f"{'β0 (components)':<25} {b0_b:>15} {b0_a:>15} {b0_a - b0_b:>+15}")
    print(f"{'β1 (loops)':<25} {b1_b:>15} {b1_a:>15} {b1_a - b1_b:>+15}")
    print(f"{'β2 (voids)':<25} {b2_b:>15} {b2_a:>15} {b2_a - b2_b:>+15}")

    print(f"{'Persistent entropy H1':<25} {initial_results['entropy_h1']:>15.3f} {treatment_results['entropy_h1']:>15.3f} {treatment_results['entropy_h1'] - initial_results['entropy_h1']:>+15.3f}")

    print(f"{'Lyapunov λ':<25} {initial_results['lyap_max']:>15.3f} {treatment_results['lyap_max']:>15.3f} {treatment_results['lyap_max'] - initial_results['lyap_max']:>+15.3f} ({treatment_results['lyap_interp']})")

    print(f"{'Correlation dim':<25} {initial_results['corr_dim']:>15.3f} {treatment_results['corr_dim']:>15.3f} {treatment_results['corr_dim'] - initial_results['corr_dim']:>+15.3f}")

    print(f"{'Approx entropy':<25} {initial_results['apen']:>15.3f} {treatment_results['apen']:>15.3f} {treatment_results['apen'] - initial_results['apen']:>+15.3f}")

    print(f"{'Box dimension':<25} {initial_results['box_dim']:>15.3f} {treatment_results['box_dim']:>15.3f} {treatment_results['box_dim'] - initial_results['box_dim']:>+15.3f}")

    print(f"{'Most complex var':<25} {initial_results['most_complex_var']:>15} {treatment_results['most_complex_var']:>15}")

    print(f"{'Mean curvature':<25} {initial_results['mean_curv']:>15.3f} {treatment_results['mean_curv']:>15.3f} {treatment_results['mean_curv'] - initial_results['mean_curv']:>+15.3f}")

    print("-"*70)
    print(f"Diagnosis before: {initial_results['diagnosis']}")
    print(f"Diagnosis after : {treatment_results['diagnosis']}")
    print("="*70)

    print("\nPost-Treatment Interpretation:")
    beta0_a, beta1_a, beta2_a = treatment_results['betti']
    print(interpret_metric('beta0', beta0_a))
    print(interpret_metric('beta1', beta1_a))
    print(interpret_metric('beta2', beta2_a))
    print(interpret_metric('lyap_max', treatment_results['lyap_max']))

    metrics_post = {
        'beta0': beta0_a,
        'beta1': beta1_a,
        'beta2': beta2_a,
        'lyap_max': treatment_results['lyap_max'],
        'entropy_h1': treatment_results['entropy_h1'],
        'corr_dim': treatment_results['corr_dim'],
        'apen': treatment_results['apen'],
        'box_dim': treatment_results['box_dim'],
        'mean_curv': treatment_results['mean_curv'],
        'max_complex': treatment_results['max_complex']
    }
    healthy_devs = compare_to_reference(metrics_post, 'healthy')
    if not healthy_devs:
        print("\n✅ All metrics are now within healthy range!")
    else:
        print("\n❌ Still outside healthy range:")
        for d in healthy_devs:
            print(f"  {d}")

    if config.USE_GEMINI:
        print(f"DEBUG: USE_GEMINI = {config.USE_GEMINI}")  # diagnostic
        insight = generate_treatment_insight(initial_results, treatment_results)
        print("\n[GEMINI TREATMENT INSIGHT]\n", insight)

        metrics_dict = {
            'hrv_value': treatment_results['hrv_mean'],
            'map_value': treatment_results['map_mean'],
            'il6_value': treatment_results['il6_mean'],
            'betti': treatment_results['betti'],
            'health_score': 50  # placeholder
        }
        pf_treatment = generate_patient_friendly_summary(
            treatment_results['condition'],
            metrics_dict,
            treatment_results['diagnosis']
        )
        print("\n[PATIENT-FRIENDLY TREATMENT UPDATE]\n", pf_treatment)

def treatment_menu():
    """Display treatment options and get user choice."""
    print("\n--- What-If Treatment Simulator ---")
    print("Select an intervention to simulate:")
    print("1. Antibiotics")
    print("2. Oxygen boost")
    print("3. Fluids")
    print("4. Anti-inflammatories")
    print("5. Exit simulator")
    choice = input("Enter choice (1-5): ").strip()
    treatments = {
        '1': 'antibiotics',
        '2': 'oxygen',
        '3': 'fluids',
        '4': 'anti_inflammatories'
    }
    if choice in treatments:
        return treatments[choice]
    elif choice == '5':
        return None
    else:
        print("Invalid choice. Please enter 1-5.")
        return treatment_menu()