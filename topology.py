"""
Topological data analysis: persistent homology, Betti numbers,
persistent entropy, persistence landscapes, and Wasserstein distances.
"""

import numpy as np
import ripser
import persim
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
 
import config

def compute_persistence(point_cloud, max_dim=None):
    """Compute persistent homology diagrams using Ripser."""
    if max_dim is None:
        max_dim = config.TDA_MAX_HOMOLOGY_DIM
    diagrams = ripser.ripser(point_cloud, maxdim=max_dim)['dgms']
    return diagrams

def betti_numbers_at_scale(diagrams, scale=None):
    """Compute Betti numbers at a given filtration scale."""
    if scale is None:
        scale = config.BETTI_SCALE_THRESHOLD
    betti = []
    for dgm in diagrams:
        if len(dgm) == 0:
            betti.append(0)
        else:
            count = np.sum((dgm[:, 0] <= scale) & (dgm[:, 1] > scale))
            betti.append(count)
    while len(betti) < 3:
        betti.append(0)
    return betti[:3]

def persistent_entropy(diagram, bin_width=None):
    """
    Compute persistent entropy of a persistence diagram.
    Based on the distribution of persistence lifetimes.
    """
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    if bin_width is None:
        bin_width = config.ENTROPY_BIN_WIDTH
    bins = np.arange(0, np.max(lifetimes) + bin_width, bin_width)
    hist, _ = np.histogram(lifetimes, bins=bins, density=True)
    hist = hist[hist > 0]
    return entropy(hist, base=np.e)

def persistence_landscape(diagram, hom_dim=1, n_points=100):
    """
    Compute persistence landscape for a given homology dimension.
    Simplified: returns the landscape function at sampled points.
    """
    if len(diagram) == 0:
        return np.zeros(n_points)
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    lambdas = np.linspace(0, np.max(deaths), n_points)
    landscape = np.zeros(n_points)
    for i, lam in enumerate(lambdas):
        contrib = np.maximum(0, np.minimum(deaths - lam, lam - births))
        landscape[i] = np.max(contrib) if len(contrib) > 0 else 0
    return landscape

def wasserstein_distance(dgm1, dgm2, p=2):
    """Compute p-Wasserstein distance between two persistence diagrams."""
    return persim.wasserstein(dgm1, dgm2, matching=False)

def bottleneck_distance(dgm1, dgm2):
    """Bottleneck distance."""
    return persim.bottleneck(dgm1, dgm2)

def persistent_landscapes_distance(land1, land2):
    """L2 distance between persistence landscapes."""
    return np.sqrt(np.sum((land1 - land2)**2))

def dominant_homology(betti):
    """Determine which homology dimension is most prominent based on Betti numbers."""
    if betti[2] > betti[1] and betti[2] > betti[0]:
        return "H2 (voids)"
    elif betti[1] > betti[0]:
        return "H1 (loops)"
    else:
        return "H0 (components)"