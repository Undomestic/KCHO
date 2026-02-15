"""
Clinical report generation and decision support.
Includes enhanced interpretations and validation summaries.
Uses Gemini for AI‑powered narratives with dataset examples.
"""

import config
from clinical_interpretation import (
    interpret_metric,
    interpret_betti_with_clinical,
    validation_summary,
    compare_to_reference,
    METRIC_MEANINGS
)
from gemini_integration import (
    generate_patient_friendly_summary,
    generate_metric_explanation,
    generate_synthetic_case_study,
    generate_health_score_interpretation,
    generate_clinical_summary
)
from math_utils import compute_health_score

def interpret_lyapunov(lyap):
    """Return a short textual interpretation of Lyapunov exponent."""
    if lyap > 0.5:
        return "highly chaotic (λ > 0.5)"
    elif lyap > 0:
        return "chaotic (λ > 0)"
    elif lyap < -0.1:
        return "stable (λ < 0)"
    else:
        return "near-zero (critical)"

def interpret_betti(betti):
    """Return a short textual interpretation of Betti numbers."""
    beta0, beta1, beta2 = betti
    lines = []
    if beta1 == 0:
        lines.append("Loss of homeostatic loop (β1=0)")
    if beta2 >= 1:
        lines.append(f"Microcirculatory void (β2={beta2})")
    if beta0 > 1:
        lines.append(f"Fragmentation (β0={beta0})")
    return "; ".join(lines) if lines else "Topologically intact"

def dominant_homology(betti):
    """Determine which homology dimension is most prominent based on Betti numbers."""
    if betti[2] > betti[1] and betti[2] > betti[0]:
        return "H2 (voids)"
    elif betti[1] > betti[0]:
        return "H1 (loops)"
    else:
        return "H0 (components)"

def generate_diagnosis(betti, lyap, most_complex_var, mean_curv):
    """Generate a concise diagnosis string from metrics."""
    parts = []
    if lyap > 0.3:
        parts.append("Chaotic instability")
    if betti[1] == 0:
        parts.append("Loop rupture")
    if betti[2] > 0:
        parts.append("Microcirculatory failure")
    if mean_curv < -0.1:
        parts.append("Negative curvature (expansion)")
    if not parts:
        return "Stable – no acute topological abnormality"
    return " + ".join(parts)

def generate_recommendations(condition, betti, lyap):
    """Generate treatment recommendations based on metrics and condition."""
    recs = []
    if betti[2] > 0:
        recs.append("Target microcirculatory void: fluid resuscitation + vasopressors")
    if lyap > 0.3:
        recs.append("Administer broad-spectrum antibiotics immediately to reduce chaos")
    if betti[1] == 0:
        recs.append("Consider inotropes to restore baroreflex loop")
    if 'sepsis' in condition.lower():
        recs.append("Source control: identify and drain infection")
    elif 'cardiac' in condition.lower():
        recs.append("Advanced cardiac life support (ACLS) protocol")
    elif 'hemorrhage' in condition.lower():
        recs.append("Control bleeding + massive transfusion protocol")
    if not recs:
        recs.append("Continue monitoring; reassess topology in 1 hour")
    return "\n• ".join(["", *recs])  # bullet list

def generate_explanation(betti, lyap, most_complex_var, mean_curv):
    """Generate a plain‑language explanation of the current state."""
    exp = f"The patient's physiology resides in a {interpret_lyapunov(lyap)} regime. "
    exp += f"Topological analysis reveals {interpret_betti(betti)}. "
    exp += f"The variable with highest Kolmogorov complexity is {most_complex_var}, indicating it carries the most dynamical information. "
    exp += f"Mean Ricci curvature is {mean_curv:.3f}, suggesting a {'contracting' if mean_curv > 0 else 'expanding'} phase space. "
    if lyap > 0 and betti[1] == 0:
        exp += "The combination of chaos and loop rupture implies imminent hemodynamic collapse if untreated."
    return exp

def enhanced_report(initial_results, condition, closest_examples=None):
    """
    Generate a comprehensive report with per‑metric interpretations and validation.
    Optionally includes Gemini-generated summary with dataset examples.
    """
    beta0, beta1, beta2 = initial_results['betti']
    lines = []
    lines.append("="*60)
    lines.append("ENHANCED CLINICAL REPORT WITH INTERPRETATIONS")
    lines.append("="*60)
    lines.append(f"Condition: {condition}")
    lines.append("")
    lines.append("-- Topological Metrics --")
    lines.append(interpret_betti_with_clinical(beta0, beta1, beta2))
    lines.append("")
    lines.append("-- Dynamical Metrics --")
    lines.append(interpret_metric('lyap_max', initial_results['lyap_max']))
    lines.append(interpret_metric('entropy_h1', initial_results['entropy_h1']))
    lines.append(interpret_metric('corr_dim', initial_results['corr_dim']))
    lines.append(interpret_metric('apen', initial_results['apen']))
    lines.append(interpret_metric('box_dim', initial_results['box_dim']))
    lines.append(interpret_metric('mean_curv', initial_results['mean_curv']))
    lines.append(interpret_metric('max_complex', initial_results['max_complex']))
    lines.append("")
    lines.append(validation_summary(initial_results))
    lines.append("="*60)

    if config.USE_GEMINI:
        metrics_dict = {
            'betti': initial_results['betti'],
            'lyap_max': initial_results['lyap_max'],
            'lyap_interp': initial_results['lyap_interp'],
            'entropy_h1': initial_results['entropy_h1'],
            'corr_dim': initial_results['corr_dim'],
            'apen': initial_results['apen'],
            'box_dim': initial_results['box_dim'],
            'mean_curv': initial_results['mean_curv'],
            'most_complex_var': initial_results['most_complex_var'],
            'max_complex': initial_results['max_complex']
        }
        betti_interp = interpret_betti_with_clinical(beta0, beta1, beta2)
        diagnosis = generate_diagnosis(
            initial_results['betti'],
            initial_results['lyap_max'],
            initial_results['most_complex_var'],
            initial_results['mean_curv']
        )
        gemini_text = generate_clinical_summary(
            condition, metrics_dict, betti_interp, diagnosis,
            closest_examples=closest_examples
        )
        lines.append("\n[GEMINI AI SUMMARY]\n" + gemini_text)
        lines.append("="*60)

    return "\n".join(lines)

def generate_easy_explanations(initial_results, condition, metrics_dict, closest_examples=None):
    """
    Generate a set of easy-to-understand explanations using Gemini.
    Incorporates dataset examples if provided.
    """
    if not config.USE_GEMINI:
        return "Gemini API not configured – easy explanations unavailable."

    lines = []
    lines.append("\n" + "="*60)
    lines.append("EASY EXPLANATIONS (Powered by Gemini)")
    lines.append("="*60)

    pf = generate_patient_friendly_summary(
        condition, metrics_dict, initial_results['diagnosis'],
        closest_examples=closest_examples
    )
    lines.append("\n[For the Patient/Family]\n" + pf)

    health_score = compute_health_score(
        initial_results['betti'],
        initial_results['lyap_max'],
        initial_results['entropy_h1'],
        initial_results['corr_dim'],
        initial_results['apen'],
        initial_results['box_dim'],
        initial_results['mean_curv'],
        initial_results['max_complex']
    )
    hs = generate_health_score_interpretation(
        health_score, metrics_dict,
        closest_examples=closest_examples
    )
    lines.append(f"\n[Health Score: {health_score:.1f}/100]\n" + hs)

    lines.append("\n[What Do These Numbers Mean?]")
    for metric in ['beta0', 'beta1', 'beta2', 'lyap_max']:
        if metric == 'lyap_max':
            value = initial_results['lyap_max']
            interp = "measures chaos"
        else:
            idx = {'beta0':0, 'beta1':1, 'beta2':2}[metric]
            value = initial_results['betti'][idx]
            interp = f"{metric} counts topological features"
        exp = generate_metric_explanation(metric, value, interp)
        lines.append(f"\n{metric.upper()} = {value:.2f}:\n{exp}")

    cs = generate_synthetic_case_study(
        condition, metrics_dict, initial_results['diagnosis'],
        closest_examples=closest_examples
    )
    lines.append("\n[Synthetic Case Study]\n" + cs)

    lines.append("="*60)
    return "\n".join(lines)