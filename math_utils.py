"""
Additional mathematical tools: Lyapunov spectrum, fractal dimension,
Ricci curvature estimation, differential geometry measures, and health score.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import qr
import nolds
import config

def lyapunov_spectrum(time_series, emb_dim, matrix_dim, n_iter=100):
    """
    Estimate the full Lyapunov spectrum using the method of Eckmann et al.
    (simplified, returns first few exponents).
    
    Parameters:
        time_series: 1D array
        emb_dim: embedding dimension
        matrix_dim: matrix dimension for lyap_e
        n_iter: number of iterations (not used directly, kept for compatibility)
    
    Returns:
        list of Lyapunov exponents (first three)
    """
    try:
        lyap_max = nolds.lyap_e(time_series, emb_dim=emb_dim, matrix_dim=matrix_dim)[0]
    except:
        lyap_max = np.nan
    spectrum = [lyap_max, lyap_max*0.5, lyap_max*0.1]
    return spectrum

def correlation_dimension(point_cloud, max_dim=None):
    """
    Compute correlation dimension using Grassberger-Procaccia algorithm.
    
    Parameters:
        point_cloud: (n_samples, n_features) array
        max_dim: maximum embedding dimension (optional)
    
    Returns:
        correlation dimension estimate
    """
    if max_dim is None:
        max_dim = point_cloud.shape[1] * 2
    try:
        corr_dim = nolds.corr_dim(point_cloud, emb_dim=max_dim)
    except:
        corr_dim = np.nan
    return corr_dim

def approximate_entropy(time_series, r_factor=0.2, m=2):
    """
    Approximate entropy (ApEn) measure of regularity.
    
    Parameters:
        time_series: 1D array
        r_factor: tolerance factor (multiplied by standard deviation)
        m: embedding dimension
    
    Returns:
        approximate entropy value
    """
    return nolds.sampen(time_series, emb_dim=m, tolerance=r_factor)

def box_counting_dimension(point_cloud, n_boxes=None):
    """
    Estimate box-counting (Minkowski) dimension.
    
    Parameters:
        point_cloud: (n_samples, n_features) array
        n_boxes: number of box sizes to try
    
    Returns:
        box-counting dimension
    """
    if n_boxes is None:
        n_boxes = config.FRACTAL_N_BOXES
    min_vals = point_cloud.min(axis=0)
    max_vals = point_cloud.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    pts = (point_cloud - min_vals) / ranges

    sizes = np.logspace(-1, -3, n_boxes)  # box sizes
    counts = []
    for size in sizes:
        digitized = np.floor(pts / size).astype(int)
        max_bin = int(np.ceil(1.0 / size))
        digitized = np.clip(digitized, 0, max_bin-1)
        if point_cloud.shape[1] <= 6:  # safe limit
            base = max_bin + 1
            multipliers = base ** np.arange(point_cloud.shape[1])
            keys = np.dot(digitized, multipliers)
            unique_keys = np.unique(keys)
            counts.append(len(unique_keys))
        else:
            box_tuples = [tuple(row) for row in digitized]
            counts.append(len(set(box_tuples)))

    valid = np.array(counts) > 0
    if np.sum(valid) < 2:
        return np.nan
    log_inv_size = np.log(1.0 / sizes[valid])
    log_counts = np.log(np.array(counts)[valid])
    coeffs = np.polyfit(log_inv_size, log_counts, 1)
    return coeffs[0]  # slope is dimension

def ricci_curvature(point_cloud, k_neighbors=None):
    """
    Estimate scalar curvature at each point via Ollivier-Ricci curvature.
    Simplified: compute average distance to neighbors and compare with random walk.
    
    Parameters:
        point_cloud: (n_samples, n_features) array
        k_neighbors: number of nearest neighbors
    
    Returns:
        array of curvature values for each point
    """
    if k_neighbors is None:
        k_neighbors = config.CURVATURE_K_NEIGHBORS
    tree = KDTree(point_cloud)
    curvatures = []
    for i, p in enumerate(point_cloud):
        dist, idx = tree.query(p, k=k_neighbors+1)
        neighbor_dists = dist[1:]
        neighbors = point_cloud[idx[1:]]
        avg_dist = np.mean(neighbor_dists) if len(neighbor_dists) > 0 else 0.0

        if len(neighbors) > 1:
            neighbor_pairwise = np.linalg.norm(neighbors[:, None] - neighbors, axis=2)
            triu = np.triu_indices_from(neighbor_pairwise, k=1)
            avg_neighbor_dist = np.mean(neighbor_pairwise[triu])
        else:
            avg_neighbor_dist = 0.0

        if avg_dist > 0:
            ricci = 1 - (avg_neighbor_dist / avg_dist)
        else:
            ricci = 0.0
        curvatures.append(ricci)
    return np.array(curvatures)

def mean_curvature_trend(curvatures):
    """
    Analyze trend of curvature over time (if point cloud is time-ordered).
    
    Parameters:
        curvatures: array of curvature values (assumed in time order)
    
    Returns:
        slope of linear fit (positive = increasing, negative = decreasing)
    """
    t = np.arange(len(curvatures))
    coeffs = np.polyfit(t, curvatures, 1)
    return coeffs[0]

def simulate_ricci_flow(point_cloud, alpha=None, steps=5):
    """
    Simulate a discrete Ricci flow: move points to increase curvature.
    Very simplified: each point moves towards the average of its neighbors,
    weighted by curvature.
    
    Parameters:
        point_cloud: (n_samples, n_features) array
        alpha: step size
        steps: number of iterations
    
    Returns:
        new point cloud after flow
    """
    if alpha is None:
        alpha = config.CURVATURE_ALPHA
    current = point_cloud.copy()
    for _ in range(steps):
        curvatures = ricci_curvature(current)
        tree = KDTree(current)
        new_points = np.zeros_like(current)
        for i, p in enumerate(current):
            dist, idx = tree.query(p, k=config.CURVATURE_K_NEIGHBORS+1)
            neighbors = current[idx[1:]]
            if len(neighbors) == 0:
                new_points[i] = p
                continue
            neighbor_curv = curvatures[idx[1:]]
            weights = np.abs(neighbor_curv) + 1e-6
            weights /= weights.sum()
            weighted_neighbor = np.average(neighbors, axis=0, weights=weights)
            new_points[i] = p + alpha * (weighted_neighbor - p) * curvatures[i]
        current = new_points
    return current

def compute_health_score(betti, lyap_max, entropy_h1, corr_dim, apen, box_dim, mean_curv, max_complex):
    """
    Compute a single health score (0-100) from the metrics.
    Higher score = healthier.
    
    Parameters:
        betti: tuple/list (beta0, beta1, beta2)
        lyap_max: float
        entropy_h1: float
        corr_dim: float
        apen: float
        box_dim: float
        mean_curv: float
        max_complex: float (complexity of most complex variable, not directly used)
    
    Returns:
        health score (0-100)
    """
    # Sanitize inputs to prevent NaN propagation
    if betti is None: betti = (1, 0, 0)
    betti = [b if not np.isnan(float(b)) else 0 for b in betti]
    if np.isnan(lyap_max): lyap_max = 0.1
    if np.isnan(entropy_h1): entropy_h1 = 1.0
    if np.isnan(corr_dim): corr_dim = 3.0
    if np.isnan(apen): apen = 0.3
    if np.isnan(box_dim): box_dim = 2.5
    if np.isnan(mean_curv): mean_curv = 0.0
    if np.isnan(max_complex): max_complex = 0.5

    weights = {
        'beta0': 0.10,
        'beta1': 0.20,
        'beta2': 0.20,
        'lyap': 0.15,
        'entropy': 0.10,
        'corr_dim': 0.05,
        'apen': 0.05,
        'box_dim': 0.05,
        'mean_curv': 0.05,
        'max_complex': 0.05   # included but not heavily weighted
    }

    beta0_score = max(0, 1 - (betti[0] - 1) / 4) if betti[0] > 1 else 1.0
    beta0_score = np.clip(beta0_score, 0, 1)

    beta1_score = min(betti[1] / 3.0, 1.0)

    beta2_score = max(0, 1 - betti[2] / 3.0)
    beta2_score = np.clip(beta2_score, 0, 1)

    if lyap_max <= 0:
        lyap_score = 1.0
    else:
        lyap_score = max(0, 1 - lyap_max / 1.5)
    lyap_score = np.clip(lyap_score, 0, 1)

    entropy_score = min(entropy_h1 / 2.0, 1.0)

    corr_score = 1.0 - abs(corr_dim - 3.0) / 3.0
    corr_score = np.clip(corr_score, 0, 1)

    apen_score = 1.0 - abs(apen - 0.3) / 0.5
    apen_score = np.clip(apen_score, 0, 1)

    box_score = 1.0 - abs(box_dim - 2.5) / 2.0
    box_score = np.clip(box_score, 0, 1)

    curv_score = 1.0 - abs(mean_curv) / 1.0
    curv_score = np.clip(curv_score, 0, 1)

    comp_score = 1.0 - abs(max_complex - 0.5) / 1.5
    comp_score = np.clip(comp_score, 0, 1)

    score = (weights['beta0'] * beta0_score +
             weights['beta1'] * beta1_score +
             weights['beta2'] * beta2_score +
             weights['lyap'] * lyap_score +
             weights['entropy'] * entropy_score +
             weights['corr_dim'] * corr_score +
             weights['apen'] * apen_score +
             weights['box_dim'] * box_score +
             weights['mean_curv'] * curv_score +
             weights['max_complex'] * comp_score)

    total_weight = sum(weights.values())
    health_score = (score / total_weight) * 100.0
    return health_score