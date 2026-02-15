"""
Dynamic synthetic data generation based on user‑input scenario.
Uses stochastic differential equations with randomised parameters.
"""

import numpy as np
from scipy.integrate import odeint
import config
import time

def _sepsis_ode(y, t, params):
    """Simplified ODE system for sepsis progression."""
    il6, hrv, map_, lac, net, scvo2 = y
    d_il6 = params['il6_rise_rate'] * il6 * (t > 2) - 0.1 * il6 * (t < 2)
    d_hrv = -params['hrv_collapse_rate'] * hrv * (t > 3) + 0.5 * (80 - hrv) * (t < 3)
    d_map = -params['map_drop_rate'] * (t > 4) * map_ + 0.2 * (100 - map_) * (t < 4)
    d_lac = params['lactate_rise_rate'] * (t > 3) * (1 + 0.5 * il6) - 0.1 * lac
    d_net = params['net_rise_rate'] * (t > 5) * (1 + 0.3 * il6) - 0.05 * net
    d_scvo2 = -params['scvo2_dip_rate'] * (t > 2) * (scvo2 - 30) + 0.1 * (75 - scvo2) * (t < 2)
    return [d_il6, d_hrv, d_map, d_lac, d_net, d_scvo2]

def _cardiac_arrest_ode(y, t, params):
    """ODE for cardiac arrest – rapid collapse."""
    il6, hrv, map_, lac, net, scvo2 = y
    d_il6 = 0.1 * (t > 4)  # delayed inflammation
    d_hrv = -params['hrv_plunge_rate'] * hrv * (t > 1)
    d_map = -params['map_crash_rate'] * map_ * (t > 2)
    d_lac = params['lactate_spike_rate'] * (t > 2) * (1 - np.exp(-(t-2)))
    d_net = 0.2 * (t > 3)
    d_scvo2 = -params['scvo2_drop_rate'] * (t > 2)
    return [d_il6, d_hrv, d_map, d_lac, d_net, d_scvo2]

def _hemorrhage_ode(y, t, params):
    """ODE for hemorrhage – linear MAP drop, compensatory tachycardia."""
    il6, hrv, map_, lac, net, scvo2 = y
    d_il6 = 0.1 * (t > 3)
    d_hrv = params['hrv_compensatory'] * (1 - hrv/100) * (t < 4) - 0.3 * hrv * (t > 4)
    d_map = -params['map_linear_drop'] * (t > 1) + 0.2 * (80 - map_) * (t < 1)
    d_lac = params['lactate_rise'] * (t > 2)
    d_net = 0.1 * (t > 5)
    d_scvo2 = -params['scvo2_plummet'] * (t > 1)
    return [d_il6, d_hrv, d_map, d_lac, d_net, d_scvo2]

def get_default_params(condition, vary=True):
    """
    Return the default ODE parameters for a given condition,
    optionally with random variation to generate diverse outputs.
    """
    cond_lower = condition.lower()
    if 'sepsis' in cond_lower:
        params = {
            'il6_rise_rate': 0.8,
            'hrv_collapse_rate': 0.5,
            'map_drop_rate': 0.3,
            'lactate_rise_rate': 0.4,
            'net_rise_rate': 0.6,
            'scvo2_dip_rate': 0.4
        }
    elif 'cardiac' in cond_lower or 'arrest' in cond_lower:
        params = {
            'hrv_plunge_rate': 0.9,
            'map_crash_rate': 0.7,
            'lactate_spike_rate': 1.0,
            'scvo2_drop_rate': 0.5
        }
    elif 'hemorrhage' in cond_lower or 'bleed' in cond_lower:
        params = {
            'map_linear_drop': 0.6,
            'hrv_compensatory': 0.4,
            'lactate_rise': 0.3,
            'scvo2_plummet': 0.5
        }
    else:
        params = {
            'il6_rise_rate': 0.8,
            'hrv_collapse_rate': 0.5,
            'map_drop_rate': 0.3,
            'lactate_rise_rate': 0.4,
            'net_rise_rate': 0.6,
            'scvo2_dip_rate': 0.4
        }

    if vary:
        for key in params:
            factor = 1.0 + np.random.uniform(-config.PARAM_VARIATION, config.PARAM_VARIATION)
            params[key] *= factor
    return params

def generate_data_for_condition(condition, t_points=None, seed=None, params=None, vary=True):
    """
    Generate synthetic time series for a given medical condition.
    
    Parameters:
        condition: string describing the scenario
        t_points: int, number of time points
        seed: int, random seed (if None, uses system time)
        params: dict, optional custom parameters to override defaults
        vary: bool, whether to apply random variation to default parameters
    
    Returns:
        dict with keys 'time' and each variable name.
    """
    if t_points is None:
        t_points = config.N_TIME_POINTS
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time() * 1000) % 2**32)
    t = np.linspace(0, config.TIME_SPAN, t_points)

    y0 = [10.0, 80.0, 95.0, 1.2, 5.0, 75.0]
    y0 = [v * (1 + 0.05 * np.random.randn()) for v in y0]  # 5% variation

    cond_lower = condition.lower()
    if 'sepsis' in cond_lower:
        default_params = get_default_params(condition, vary=vary)
        ode = _sepsis_ode
    elif 'cardiac' in cond_lower or 'arrest' in cond_lower:
        default_params = get_default_params(condition, vary=vary)
        ode = _cardiac_arrest_ode
    elif 'hemorrhage' in cond_lower or 'bleed' in cond_lower:
        default_params = get_default_params(condition, vary=vary)
        ode = _hemorrhage_ode
    else:
        default_params = get_default_params('sepsis', vary=vary)
        ode = _sepsis_ode

    if params is not None:
        for key in params:
            if key in default_params:
                default_params[key] = params[key]
            else:
                print(f"Warning: Parameter '{key}' not used in this ODE.")

    sol = odeint(ode, y0, t, args=(default_params,))
    noise = 0.05 * np.random.randn(*sol.shape)
    sol_noisy = sol + noise * sol

    data = {
        'time': t,
        'IL6': sol_noisy[:, 0],
        'HRV': np.clip(sol_noisy[:, 1], 0, 150),
        'MAP': np.clip(sol_noisy[:, 2], 20, 140),
        'Lactate': np.clip(sol_noisy[:, 3], 0.5, 20),
        'NET': np.clip(sol_noisy[:, 4], 0, 50),
        'ScvO2': np.clip(sol_noisy[:, 5], 20, 95)
    }
    return data

def chebyshev_smooth(y, t, degree=None):
    """Smooth a 1D signal using Chebyshev polynomial approximation."""
    from numpy.polynomial import Chebyshev
    if degree is None:
        degree = config.CHEBYSHEV_DEGREE
    c = Chebyshev.fit(t, y, degree)
    return c(t)

def smooth_all_variables(raw_data, degree=None):
    """Apply Chebyshev smoothing to all physiological variables."""
    smoothed = {'time': raw_data['time']}
    for var in config.PHYSIO_VARS:
        smoothed[var] = chebyshev_smooth(raw_data[var], raw_data['time'], degree)
    return smoothed