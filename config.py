"""
Global configuration parameters for the KCTHO pipeline.
Includes Gemini API key (set your own).
"""

import os

N_TIME_POINTS = 500          # number of time samples
TIME_SPAN = 10.0              # hours

PHYSIO_VARS = ['IL6', 'HRV', 'MAP', 'Lactate', 'NET', 'ScvO2']

PARAM_VARIATION = 0.2  # ±20% variation

CHEBYSHEV_DEGREE = 12

TDA_MAX_HOMOLOGY_DIM = 2
TDA_FILTRATION_MAX = 50.0
TDA_FILTRATION_STEP = 0.5
BETTI_SCALE_THRESHOLD = 20.0   # for simple Betti count

ENTROPY_BIN_WIDTH = 0.5

LYAPUNOV_EMBEDDING_DIM = 10
LYAPUNOV_MATRIX_DIM = 2
LYAPUNOV_TOLERANCE = 0.1

KAN_SPLINE_SMOOTHNESS = 1.0   # smoothing factor for UnivariateSpline
KAN_N_GRID = 200
KAN_N_KNOTS = 10               # for B-spline KAN

FRACTAL_N_BOXES = 50

CURVATURE_K_NEIGHBORS = 10
CURVATURE_ALPHA = 0.1          # regularisation for Ricci flow simulation

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_DEFAULT_KEY = ""
USE_GEMINI = (GEMINI_API_KEY != _DEFAULT_KEY) and (len(GEMINI_API_KEY) > 0)

REPORT_TEMPLATE = """
=========================================================
         KOLMOGOROV-CHEBYSHEV TOPOLOGICAL HEALTH ORACLE
=========================================================
Patient Condition: {condition}

[TOPOLOGICAL STATE]
• Betti numbers (β0,β1,β2): {betti}
• Persistent entropy: {entropy:.3f}
• Dominant homology group: {dominant_homology}

[DYNAMICAL STABILITY]
• Maximal Lyapunov exponent λ = {lyap:.3f}  ({lyap_interp})
• Correlation dimension: {corr_dim:.3f}
• Approximate entropy: {approx_entropy:.3f}

[KOLMOGOROV-ARNOLD NETWORK]
• Most complex variable: {most_complex_var} (complexity = {max_complex:.3f})
• Edge activation shape: {edge_shape}

[RICCI CURVATURE]
• Mean scalar curvature: {mean_curv:.3f}
• Curvature trend: {curv_trend}

[DIAGNOSIS]
{diagnosis}

[RECOMMENDATIONS]
{recommendations}

[EXPLANATION]
{explanation}
=========================================================
"""