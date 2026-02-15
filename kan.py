"""
Kolmogorov-Arnold Network simulation: learnable spline-based edge functions,
Kolmogorov complexity via LZ and spectral measures.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, BSpline
from scipy.fft import fft
from scipy.stats import entropy
import config

def fit_kan_edge_bspline(x, y, n_knots=None, degree=3):
    """
    Fit a B-spline to represent a KAN edge activation.
    Returns the spline object and knot points.
    """
    if n_knots is None:
        n_knots = config.KAN_N_KNOTS
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    uniq_x, inv = np.unique(x, return_inverse=True)
    if uniq_x.size != x.size:
        y_mean = np.zeros(uniq_x.size, dtype=float)
        counts = np.zeros(uniq_x.size, dtype=int)
        for i, idx in enumerate(inv):
            y_mean[idx] += y[i]
            counts[idx] += 1
        y_mean /= counts
        x = uniq_x
        y = y_mean

    if x.size <= degree or (n_knots is not None and n_knots <= 0):
        from scipy.interpolate import UnivariateSpline
        return UnivariateSpline(x, y, k=degree, s=0)

    try:
        from scipy.interpolate import LSQUnivariateSpline
        knots = np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]
        knots = np.clip(knots, x.min() + 1e-12, x.max() - 1e-12)
        spline = LSQUnivariateSpline(x, y, knots, k=degree)
        return spline
    except Exception:
        from scipy.interpolate import UnivariateSpline
        return UnivariateSpline(x, y, k=degree, s=0)

def kan_edge_curvature(spline, x_grid):
    """Compute mean absolute curvature of the spline as complexity measure."""
    deriv1 = spline.derivative(1)(x_grid)
    deriv2 = spline.derivative(2)(x_grid)
    curvature = np.abs(deriv2) / (1 + deriv1**2)**1.5
    return np.mean(curvature)

def lz_complexity(sequence):
    """
    Lempel-Ziv complexity for a discrete sequence.
    Quantifies the rate of new patterns.
    """
    seq = np.array(sequence)
    if seq.dtype != bool:
        med = np.median(seq)
        seq = (seq > med).astype(int)
    seq_str = ''.join(map(str, seq))
    n = len(seq_str)
    c = 1
    l = 1
    i = 0
    k = 1
    while True:
        if seq_str[i+k] not in seq_str[i:i+k]:
            c += 1
            i += k
            k = 1
        else:
            k += 1
        if i + k >= n:
            break
    return c

def spectral_entropy(signal, n_bins=50):
    """Compute entropy of power spectrum."""
    fft_vals = np.abs(fft(signal))
    power = fft_vals**2
    power = power[:len(power)//2]
    power = power / np.sum(power)
    hist, _ = np.histogram(power, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=np.e)

def kolmogorov_complexity_proxy(spline, x_grid, method='curvature'):
    """
    Unified complexity measure for a KAN edge.
    Options: 'curvature', 'lz', 'spectral'.
    """
    if method == 'curvature':
        return kan_edge_curvature(spline, x_grid)
    elif method == 'lz':
        y_grid = spline(x_grid)
        return lz_complexity(y_grid)
    elif method == 'spectral':
        y_grid = spline(x_grid)
        return spectral_entropy(y_grid)
    else:
        raise ValueError(f"Unknown method: {method}")

def most_complex_variable(smoothed_data, var_pairs, method='curvature'):
    """
    For a list of (input_var, output_var) pairs, compute complexity
    and return the input variable with highest complexity.
    """
    complexities = {}
    for in_var, out_var in var_pairs:
        x = smoothed_data[in_var]
        y = smoothed_data[out_var]
        spline = fit_kan_edge_bspline(x, y)
        x_grid = np.linspace(min(x), max(x), config.KAN_N_GRID)
        comp = kolmogorov_complexity_proxy(spline, x_grid, method)
        complexities[in_var] = comp
    most = max(complexities, key=complexities.get)
    return most, complexities[most], complexities