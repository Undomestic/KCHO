"""
Main entry point: user input, data generation, analysis, report, treatment simulation,
enhanced clinical interpretation with Gemini, and comparison with dataset examples.
"""

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import config
from data_generation import generate_data_for_condition, smooth_all_variables, get_default_params
from topology import (
    compute_persistence, betti_numbers_at_scale, persistent_entropy,
    persistence_landscape, dominant_homology
)
from kan import fit_kan_edge_bspline, kolmogorov_complexity_proxy, most_complex_variable
from math_utils import (
    lyapunov_spectrum, correlation_dimension, approximate_entropy,
    ricci_curvature, mean_curvature_trend, box_counting_dimension
)
from clinical import (
    interpret_lyapunov, interpret_betti, generate_diagnosis,
    generate_recommendations, generate_explanation,
    enhanced_report, generate_easy_explanations
)
from utils import (
    plot_raw_vs_smoothed, plot_persistence, plot_kan_edge,
    plot_3d_point_cloud, plot_curvature_over_time, print_report
)
from treatment_simulator import run_simulation, print_comparison, treatment_menu
from example_loader import get_examples, find_closest_example

def main():
    print("="*60)
    print("KOLMOGOROV-CHEBYSHEV TOPOLOGICAL HEALTH ORACLE (KCTHO)")
    print("="*60)
    print("\nEnter a medical condition or scenario (e.g., 'sepsis', 'cardiac arrest', 'hemorrhage'):")
    condition = input("> ").strip()
    if not condition:
        condition = "sepsis"
        print(f"No input, using default: {condition}")

    print("\n[PHASE 1] Generating synthetic physiological data...")
    raw_data = generate_data_for_condition(condition, seed=None, vary=True)
    smoothed_data = smooth_all_variables(raw_data)

    plot_raw_vs_smoothed(raw_data, smoothed_data)

    print("[PHASE 2] Running topological and dynamical analyses...")

    point_cloud = np.column_stack([smoothed_data[var] for var in config.PHYSIO_VARS])

    diagrams = compute_persistence(point_cloud)
    plot_persistence(diagrams)
    betti = betti_numbers_at_scale(diagrams, scale=config.BETTI_SCALE_THRESHOLD)
    beta0, beta1, beta2 = betti
    print(f"Betti numbers: β0={beta0}, β1={beta1}, β2={beta2}")

    if len(diagrams) > 1:
        entropy_h1 = persistent_entropy(diagrams[1])
    else:
        entropy_h1 = 0.0
    print(f"Persistent entropy (H1): {entropy_h1:.3f}")

    lyap_spec = lyapunov_spectrum(smoothed_data['MAP'],
                                   emb_dim=config.LYAPUNOV_EMBEDDING_DIM,
                                   matrix_dim=config.LYAPUNOV_MATRIX_DIM)
    lyap_max = lyap_spec[0]
    print(f"Max Lyapunov exponent: {lyap_max:.3f}")

    corr_dim = correlation_dimension(point_cloud)
    print(f"Correlation dimension: {corr_dim:.3f}")

    apen = approximate_entropy(smoothed_data['MAP'])
    print(f"Approximate entropy (MAP): {apen:.3f}")

    box_dim = box_counting_dimension(point_cloud)
    print(f"Box-counting dimension: {box_dim:.3f}")

    var_pairs = [(v, 'MAP') for v in config.PHYSIO_VARS if v != 'MAP']
    most_var, max_comp, all_comps = most_complex_variable(smoothed_data, var_pairs, method='curvature')
    print(f"Most complex variable (input to MAP): {most_var} with complexity {max_comp:.3f}")
    print("All complexities:", all_comps)

    curvatures = ricci_curvature(point_cloud)
    mean_curv = np.mean(curvatures)
    curv_trend = mean_curvature_trend(curvatures)
    print(f"Mean Ricci curvature: {mean_curv:.3f}, trend: {curv_trend:.3f}")

    plot_curvature_over_time(curvatures)

    plot_3d_point_cloud(point_cloud, color_by=curvatures)

    x = smoothed_data[most_var]
    y = smoothed_data['MAP']
    spline = fit_kan_edge_bspline(x, y)
    x_grid = np.linspace(min(x), max(x), config.KAN_N_GRID)
    y_spline = spline(x_grid)
    plot_kan_edge(x, y, spline, x_grid, y_spline, most_var, 'MAP')

    print("[PHASE 3] Generating clinical report...")

    dom_hom = dominant_homology(betti)
    lyap_interp = interpret_lyapunov(lyap_max)
    diagnosis = generate_diagnosis(betti, lyap_max, most_var, mean_curv)
    recommendations = generate_recommendations(condition, betti, lyap_max)
    explanation = generate_explanation(betti, lyap_max, most_var, mean_curv)

    report = config.REPORT_TEMPLATE.format(
        condition=condition.capitalize(),
        betti=f"({beta0},{beta1},{beta2})",
        entropy=entropy_h1,
        dominant_homology=dom_hom,
        lyap=lyap_max,
        lyap_interp=lyap_interp,
        corr_dim=corr_dim,
        approx_entropy=apen,
        most_complex_var=most_var,
        max_complex=max_comp,
        edge_shape="nonlinear" if max_comp > 0.5 else "near-linear",
        mean_curv=mean_curv,
        curv_trend="increasing" if curv_trend > 0 else "decreasing",
        diagnosis=diagnosis,
        recommendations=recommendations,
        explanation=explanation
    )

    print_report(report)

    initial_results = {
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
        'betti_interp': interpret_betti(betti)
    }

    print("\n" + enhanced_report(initial_results, condition))  # examples will be added later if available

    print("\n" + "="*60)
    print("COMPARISON WITH KNOWN CLINICAL EXAMPLES")
    print("="*60)
    last_vitals = {
        'map': smoothed_data['MAP'][-1],
        'lactate': smoothed_data['Lactate'][-1],
        'hrv': smoothed_data['HRV'][-1],
        'scvo2': smoothed_data['ScvO2'][-1]
    }
    print("Current vitals (last time point):")
    print(f"  MAP: {last_vitals['map']:.1f} mmHg")
    print(f"  Lactate: {last_vitals['lactate']:.2f} mmol/L")
    print(f"  HRV: {last_vitals['hrv']:.1f} ms")
    print(f"  ScvO2: {last_vitals['scvo2']:.1f}%")

    examples = get_examples()  # loads from smolify.csv, smolify 2.csv, smolify 3.csv
    closest_examples = None
    if examples:
        closest = find_closest_example(last_vitals, examples, n=3)
        if closest:
            closest_examples = closest
            print("\nClosest matching examples from dataset:")
            for i, (dist, ex) in enumerate(closest, 1):
                print(f"\n{i}. Distance: {dist:.3f}")
                print(f"   Vitals: MAP={ex['vitals']['map']}, Lactate={ex['vitals']['lactate']}, "
                      f"HRV={ex['vitals']['hrv']}, ScvO2={ex['vitals']['scvo2']}")
                print(f"   Diagnosis: {ex['diagnosis']}")
                print(f"   Warning: {ex['warning']}")
        else:
            print("No close examples found (distance calculation issue).")
    else:
        print("No examples loaded. Please ensure the following files are in the current directory:")
        for f in ['smolify.csv', 'smolify 2.csv', 'smolify 3.csv']:
            print(f"  - {f}")
    print("="*60)

    if config.USE_GEMINI:
        metrics_dict_gemini = {
            'betti': betti,
            'lyap_max': lyap_max,
            'lyap_interp': lyap_interp,
            'entropy_h1': entropy_h1,
            'corr_dim': corr_dim,
            'apen': apen,
            'box_dim': box_dim,
            'mean_curv': mean_curv,
            'most_complex_var': most_var,
            'max_complex': max_comp,
            'hrv_value': np.mean(smoothed_data['HRV']),
            'map_value': np.mean(smoothed_data['MAP']),
            'il6_value': np.mean(smoothed_data['IL6']),
        }
        easy_explanations = generate_easy_explanations(
            initial_results, condition, metrics_dict_gemini,
            closest_examples=closest_examples
        )
        print(easy_explanations)
    else:
        print("\n[Gemini API not configured – easy explanations skipped]")

    while True:
        treatment = treatment_menu()
        if treatment is None:
            break

        base_params = get_default_params(condition, vary=True)

        treatment_results = run_simulation(condition, base_params, treatment, seed=None)

        print_comparison(initial_results, treatment_results)

        cont = input("\nSimulate another treatment? (y/n): ").strip().lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    main()