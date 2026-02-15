"""
Health score computation: combines multiple metrics into a single 0-100 score.
"""

import numpy as np

def compute_health_score(results):
    """
    Compute a health score from the analysis results dictionary.
    Returns a float between 0 (critical) and 100 (healthy).
    """
    score = 0.0
    weights = {}

    lyap = results.get('lyap_max', 0)
    lyap_norm = np.clip(lyap, -1, 1)
    lyap_score = 1 - (lyap_norm + 1) / 2  # when lyap=-1 -> 1, lyap=1 -> 0
    weights['lyap'] = 0.25
    score += weights['lyap'] * lyap_score

    betti = results.get('betti', (0,0,0))
    beta0 = betti[0]
    beta0_score = 1.0 / (1 + abs(beta0 - 1))  # 1 if beta0=1, else <1.
    beta1 = betti[1]
    beta1_score = 1.0 / (1 + abs(beta1 - 1))
    beta2 = betti[2]
    beta2_score = 1.0 / (1 + beta2)
    betti_score = (beta0_score + beta1_score + beta2_score) / 3
    weights['betti'] = 0.25
    score += weights['betti'] * betti_score

    entropy = results.get('entropy_h1', 0)
    entropy_norm = np.clip(entropy / 5, 0, 1)
    entropy_score = 1 - entropy_norm
    weights['entropy'] = 0.15
    score += weights['entropy'] * entropy_score

    curv = results.get('mean_curv', 0)
    curv_score = (curv + 1) / 2
    weights['curv'] = 0.15
    score += weights['curv'] * curv_score

    complexity = results.get('max_complex', 0)
    complexity_norm = np.clip(complexity / 10, 0, 1)
    complexity_score = 1 - complexity_norm
    weights['complexity'] = 0.1
    score += weights['complexity'] * complexity_score

    apen = results.get('apen', 0)
    apen_norm = np.clip(apen / 2, 0, 1)  # typical ApEn < 2
    apen_score = 1 - apen_norm
    weights['apen'] = 0.1
    score += weights['apen'] * apen_score

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1.0
    score = score / total_weight  # should already be weighted sum; but ensure normalization

    if not np.isfinite(score):
        score = 0.0

    clipped = np.clip(score * 100, 0, 100)
    clipped = float(np.nan_to_num(clipped, nan=0.0, posinf=100.0, neginf=0.0))
    health = int(clipped)
    return health

def get_health_status(score):
    """Return textual status based on health score."""
    if score >= 80:
        return "Stable"
    elif score >= 60:
        return "Mild instability"
    elif score >= 40:
        return "Moderate instability"
    elif score >= 20:
        return "Severe instability"
    else:
        return "Critical"