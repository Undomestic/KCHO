"""
Clinical Interpretation Engine.
Maps topological and dynamical metrics to clinical meanings,
and provides reference ranges for validation.
"""

import numpy as np
import config

REFERENCE_RANGES = {
    'healthy': {
        'beta0': (1, 1),           # Single connected component
        'beta1': (1, 3),           # Some loops (e.g., cardiorespiratory coupling)
        'beta2': (0, 0),           # No voids
        'lyap_max': (-0.5, 0.1),   # Slightly negative or near-zero
        'entropy_h1': (0.5, 2.0),  # Moderate complexity
        'corr_dim': (2.0, 4.0),    # Typical dimension of physiological attractor
        'apen': (0.1, 0.5),        # Normal regularity
        'box_dim': (1.5, 3.0),
        'mean_curv': (-0.1, 0.1),  # Near zero curvature
        'max_complex': (0.1, 0.5)  # Low to moderate complexity
    },
    'sepsis': {
        'beta0': (2, 5),           # Fragmentation
        'beta1': (0, 1),           # Loss of loops
        'beta2': (1, 3),           # Microcirculatory voids
        'lyap_max': (0.3, 1.5),    # Positive, chaotic
        'entropy_h1': (0.0, 0.8),  # Reduced complexity
        'corr_dim': (1.0, 2.5),    # Lower dimension
        'apen': (0.0, 0.2),        # More regular (pathological)
        'box_dim': (1.0, 2.0),
        'mean_curv': (-0.5, 0.0),  # Negative curvature (expansion)
        'max_complex': (0.5, 2.0)  # High complexity (most variable)
    },
    'cardiac_arrest': {
        'beta0': (1, 2),
        'beta1': (0, 1),
        'beta2': (0, 1),
        'lyap_max': (0.5, 2.0),
        'entropy_h1': (0.0, 0.5),
        'corr_dim': (0.5, 1.5),
        'apen': (0.0, 0.1),
        'box_dim': (0.5, 1.5),
        'mean_curv': (-1.0, -0.2),
        'max_complex': (0.8, 2.5)
    },
    'hemorrhage': {
        'beta0': (1, 2),
        'beta1': (0, 2),
        'beta2': (0, 1),
        'lyap_max': (0.2, 0.8),
        'entropy_h1': (0.2, 1.0),
        'corr_dim': (1.5, 3.0),
        'apen': (0.1, 0.4),
        'box_dim': (1.5, 2.5),
        'mean_curv': (-0.3, 0.1),
        'max_complex': (0.3, 1.0)
    }
}

METRIC_MEANINGS = {
    'beta0': "Number of connected components – indicates whether the physiological system is unified (β0=1) or fragmented (β0>1). Fragmentation suggests organ dysconnectivity.",
    'beta1': "Number of one-dimensional loops – represents stable cycles like cardiorespiratory coupling. Loss of loops (β1=0) implies loss of homeostatic regulation.",
    'beta2': "Number of two-dimensional voids – indicates trapped states, e.g., microcirculatory shutdown where oxygen cannot be extracted despite adequate flow.",
    'lyap_max': "Maximal Lyapunov exponent – quantifies chaos. λ>0 means sensitive dependence on initial conditions (disease progression); λ<0 means stability (recovery).",
    'entropy_h1': "Persistent entropy of H1 features – measures complexity of loop structures. Lower entropy suggests loss of dynamical richness.",
    'corr_dim': "Correlation dimension – fractal dimension of the attractor. Lower values indicate simpler, possibly pathological dynamics.",
    'apen': "Approximate entropy – regularity of a time series. Lower values mean more predictable (pathological); higher values mean more complex (healthy).",
    'box_dim': "Box-counting dimension – another fractal measure. Deviations from healthy range indicate altered phase space filling.",
    'mean_curv': "Mean Ricci curvature – geometric measure of stability. Negative curvature implies expansion (unstable), positive curvature implies contraction (stable).",
    'max_complex': "Kolmogorov complexity proxy – identifies the variable carrying the most dynamical information. High complexity suggests that variable is a key driver."
}

def interpret_metric(metric_name, value):
    """
    Return a string interpreting the given metric value in clinical context.
    """
    meaning = METRIC_MEANINGS.get(metric_name, "No interpretation available.")
    healthy_range = REFERENCE_RANGES['healthy'].get(metric_name, (None, None))
    if healthy_range[0] is not None and healthy_range[1] is not None:
        if healthy_range[0] <= value <= healthy_range[1]:
            status = "within healthy range"
        else:
            status = "outside healthy range (abnormal)"
    else:
        status = ""
    return f"{meaning} Current value: {value:.3f} – {status}."

def interpret_betti_with_clinical(beta0, beta1, beta2):
    """Combine Betti number interpretations."""
    lines = []
    lines.append(interpret_metric('beta0', beta0))
    lines.append(interpret_metric('beta1', beta1))
    lines.append(interpret_metric('beta2', beta2))
    return "\n".join(lines)

def compare_to_reference(metrics_dict, condition='healthy'):
    """
    Compare a dictionary of metrics to a reference condition.
    Returns a list of deviations.
    """
    ref = REFERENCE_RANGES.get(condition, REFERENCE_RANGES['healthy'])
    deviations = []
    for key, value in metrics_dict.items():
        if key in ref:
            low, high = ref[key]
            if not (low <= value <= high):
                deviations.append(f"{key}: {value:.3f} outside {condition} range [{low}, {high}]")
    return deviations

def validation_summary(initial_results):
    """
    Generate a summary that compares the current patient to healthy and known disease states.
    """
    metrics = {
        'beta0': initial_results['betti'][0],
        'beta1': initial_results['betti'][1],
        'beta2': initial_results['betti'][2],
        'lyap_max': initial_results['lyap_max'],
        'entropy_h1': initial_results['entropy_h1'],
        'corr_dim': initial_results['corr_dim'],
        'apen': initial_results['apen'],
        'box_dim': initial_results['box_dim'],
        'mean_curv': initial_results['mean_curv'],
        'max_complex': initial_results['max_complex']
    }
    summary = []
    summary.append("=== VALIDATION: Comparison to Reference States ===")
    for state in ['healthy', 'sepsis', 'cardiac_arrest', 'hemorrhage']:
        devs = compare_to_reference(metrics, state)
        if not devs:
            summary.append(f"All metrics within {state} range.")
        else:
            summary.append(f"Deviations from {state}:")
            for d in devs:
                summary.append(f"  - {d}")
    return "\n".join(summary)