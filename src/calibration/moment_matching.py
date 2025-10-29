from dataclasses import replace
import numpy as np

PARAMS_TO_FIT = [
    ("xi", None),
    ("delta", None),
    ("xi_mu_G0", None),
    ("xi_mu_B0", None),
    ("xi_sigma_G", None),   # mirror to B inside mapper
    ("sigma_epsilon_G", None),  # mirror to B inside mapper
    ("a_G", None),          # mirror to B inside mapper
]

BOUNDS = {
    "xi": (1e-5, 1e-2),
    "delta": (0.001, 0.05),
    "xi_mult_G": (0.3, 2.0),
    "xi_mult_B": (0.3, 2.0),
    "xi_mu_G0": (1.0, 6.0),
    "xi_mu_B0": (1.0, 6.0),
    "xi_sigma_G": (0.1, 2.0),
    "sigma_epsilon_G": (1e-4, 0.2),
    "a_G": (0.2, 5.0),
}

def pack_theta(cfg):
    names = [k for k,_ in PARAMS_TO_FIT]
    theta = []
    for name in names:
        theta.append(getattr(cfg, name))
    return np.array(theta), names

def apply_theta(cfg, theta, names):
    # Write fitted params, enforce symmetries
    d = {n: v for n, v in zip(names, theta)}
    cfg2 = replace(cfg)
    for n, v in d.items():
        setattr(cfg2, n, float(v))
    # Symmetry constraints:
    if "a_G" in d:
        cfg2.a_B = cfg2.a_G
    if "xi_sigma_G" in d:
        cfg2.xi_sigma_B = cfg2.xi_sigma_G
    if "sigma_epsilon_G" in d:
        cfg2.sigma_epsilon_B = cfg2.sigma_epsilon_G
    return cfg2

def bounds_array(names):
    return np.array([BOUNDS[n] for n in names], dtype=float)
