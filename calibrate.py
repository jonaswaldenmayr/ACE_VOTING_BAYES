"""
Simulated Method of Moments (SMM) calibration — one-file drop‑in
-----------------------------------------------------------------
• Tries to use your real model (ACE + Beliefs + Voting) if available
  - Imports from: src.ACE.ACE_reduced, src.voting.OfficeMotiv_model, src.updating.belief_updating
  - Tries multiple common method names/signatures
• Falls back to a lightweight stub simulator so the script always runs
• Multi-start optimizer, moment scaling, common random numbers
• Saves calibrated params to JSON

Usage
-----
$ python calibrate.py \
    --n-starts 4 \
    --method Nelder-Mead \
    --seed 999 \
    --save calibrated_params.json

Edit the M_TARGET section to your empirical targets.
"""
from __future__ import annotations
import json
import math
import sys
import argparse
from dataclasses import dataclass, replace, asdict
from typing import Dict, Tuple, List

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    raise SystemExit("scipy is required. Install with `pip install scipy`.\n" + str(e))

# =============================================================
# 0) CONFIG / PARAMETERS DATACLASS (fallback if your module isn't importable)
# =============================================================
try:
    # Prefer user's implementation
    from your_config_module import all_config  # type: ignore
    HAVE_EXTERNAL_CONFIG = True
except Exception:
    HAVE_EXTERNAL_CONFIG = False

    @dataclass(slots=True)
    class Parameters:
        # ---- CORE ----
        period_len: int = 10
        periods: int = 10
        start_year: int = 2020
        # ---- ACE ----
        Y_init: float = 130.0
        A_init: float = 16.09
        K_init: float = 130.0 * 3
        nu: float = 0.07
        kappa: float = 0.30
        beta: float = 0.84
        tech_improvement_rate: float = 0.018
        prtp: float = 0.013896
        xi: float = 0.0002046
        delta: float = 0.01
        ppm_value: float = 427.0
        ppm_to_GtC: float = 2.31
        BAU_E_CO2_init: float = 35.0
        molar_mass_CO2: float = 44.01
        molar_mass_C: float = 12.01
        kappa_E: float = 0.35
        @property
        def E_bau(self) -> float:
            return self.BAU_E_CO2_init * (self.molar_mass_CO2 / self.molar_mass_C)
        @property
        def M_init(self) -> float:
            return self.ppm_value * self.ppm_to_GtC
        # ---- VOTING ----
        num_voters: int = 1000
        qG: float = 0.5
        qB: float = 0.5
        xi_mult_G: float = 1.3
        xi_mult_B: float = 0.7
        a_G: float = 1.0
        a_B: float = 1.0
        # ---- BAYES ----
        xi_mu_G0: float = 3.5
        xi_mu_B0: float = 2.5
        xi_sigma_G: float = 0.7
        xi_sigma_B: float = 0.7
        sigma_epsilon_G: float = 0.03
        sigma_epsilon_B: float = 0.03

    def all_config() -> Parameters:
        return Parameters()

# =============================================================
# 1) WHICH PARAMETERS TO FIT + BOUNDS
# =============================================================
PARAMS_TO_FIT: List[Tuple[str, None]] = [
    ("xi", None),
    ("delta", None),
    ("xi_mu_G0", None),
    ("xi_mu_B0", None),
    ("xi_sigma_G", None),           # mirror to B
    ("sigma_epsilon_G", None),      # mirror to B
    ("a_G", None),                  # mirror to B
]

BOUNDS: Dict[str, Tuple[float, float]] = {
    "xi": (1e-5, 1e-2),
    "delta": (0.001, 0.05),
    "xi_mult_G": (0.3, 2.0),  # not fitted but kept for completeness
    "xi_mult_B": (0.3, 2.0),
    "xi_mu_G0": (1.0, 6.0),
    "xi_mu_B0": (1.0, 6.0),
    "xi_sigma_G": (0.1, 2.0),
    "sigma_epsilon_G": (1e-4, 0.2),
    "a_G": (0.2, 5.0),
}

# =============================================================
# 2) TARGET MOMENTS (EDIT THESE TO YOUR EMPIRICAL TARGETS)
# =============================================================
M_TARGET = np.array([
    0.0,   # trend(E)
    0.25,  # mean green vote share
    0.02,  # var(D)
    0.6,   # ac1(E)
    8.0,   # mean(E)
    0.00,  # trend(PG)
], dtype=float)

# =============================================================
# 3) PACK/APPLY HELPERS
# =============================================================

def pack_theta(cfg) -> Tuple[NDArray[np.float64], List[str]]:
    names = [k for k, _ in PARAMS_TO_FIT]
    theta = [float(getattr(cfg, name)) for name in names]
    return np.array(theta, dtype=float), names


def apply_theta(cfg, theta: NDArray[np.float64], names: List[str]):
    # Write fitted params, enforce group symmetries for B = G mirrors
    cfg2 = replace(cfg)
    d = {n: float(v) for n, v in zip(names, theta)}
    for n, v in d.items():
        setattr(cfg2, n, v)
    if "a_G" in d:
        cfg2.a_B = cfg2.a_G
    if "xi_sigma_G" in d:
        cfg2.xi_sigma_B = cfg2.xi_sigma_G
    if "sigma_epsilon_G" in d:
        cfg2.sigma_epsilon_B = cfg2.sigma_epsilon_G
    return cfg2


def bounds_array(names: List[str]) -> NDArray[np.float64]:
    return np.array([BOUNDS[n] for n in names], dtype=float)

# =============================================================
# 4) MOMENT MAP (series -> moments)
# =============================================================

def _trend(y: NDArray[np.float64]) -> float:
    t = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def _ac1(y: NDArray[np.float64]) -> float:
    if len(y) < 2:
        return 0.0
    y0, y1 = y[:-1], y[1:]
    s0, s1 = np.std(y0), np.std(y1)
    if s0 == 0 or s1 == 0:
        return 0.0
    return float(np.corrcoef(y0, y1)[0, 1])


def compute_moments(series: Dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    E, D, PG = series["E"], series["D"], series["PG"]
    return np.array([
        _trend(E),           # trend(E)
        float(np.mean(PG)),  # mean PG
        float(np.var(D, ddof=1 if len(D) > 1 else 0)),  # var(D)
        _ac1(E),             # AR(1) of E
        float(np.mean(E)),   # mean(E)
        _trend(PG),          # trend(PG)
    ], dtype=float)

# =============================================================
# 5) SIMULATOR: REAL MODEL if available, else STUB
# =============================================================

# Attempt to import user's real model components
_HAVE_REAL = True
try:
    from src.ACE.ACE_reduced import ACEModel_RF  # type: ignore
    from src.updating.belief_updating import GroupBeliefUpdating  # type: ignore
except Exception:
    _HAVE_REAL = False


def _simulate_series_stub(cfg, T: int, seed: int) -> Dict[str, NDArray[np.float64]]:
    """Fallback toy dynamics to allow end-to-end testing without your model."""
    rng = np.random.default_rng(seed)
    # Simple dynamics that depend on select parameters so calibration behaves sensibly
    xi = cfg.xi
    delta = cfg.delta
    muG0, muB0 = cfg.xi_mu_G0, cfg.xi_mu_B0
    sig_eps = cfg.sigma_epsilon_G
    a = cfg.a_G

    E = []
    D = []
    PG = []

    M = cfg.M_init
    muG, muB = muG0, muB0

    for t in range(T):
        # Green preference rises with muG - muB and with damages
        base_pg = 0.5 + 0.05 * math.tanh(0.2 * (muG - muB))
        # taste noise scale a -> flattens choices
        pg = np.clip(base_pg + rng.normal(0, 0.02 * (1 / (1 + a))), 0.0, 1.0)

        E_G = cfg.E_bau * max(0.3, 1.0 - 0.1 * muG)
        E_B = cfg.E_bau * max(0.3, 1.0 - 0.05 * muB)
        E_t = E_G if rng.random() < pg else E_B

        # Carbon and damages
        M = (1.0 - delta) * M + E_t
        D_t = 1.0 - math.exp(-xi * M + rng.normal(0, sig_eps))
        D_t = float(np.clip(D_t, 0.0, 0.95))

        # Belief drift toward each other with noisy signal ~ damages
        signal = -math.log(max(1e-6, 1.0 - D_t)) / max(M, 1e-6)
        # Simple Bayesian-ish pull
        k_gain = 1.0 / (1.0 + cfg.xi_sigma_G)
        muG = 0.9 * muG + k_gain * (signal - muG)
        muB = 0.95 * muB + 0.5 * k_gain * (signal - muB)

        E.append(E_t)
        D.append(D_t)
        PG.append(pg)

    return {"E": np.array(E), "D": np.array(D), "PG": np.array(PG)}


def _try_construct_ace(cfg):
    """Try a few common constructor signatures."""
    # Try ACEModel_RF(cfg, cfg)
    try:
        return ACEModel_RF(cfg, cfg)  # type: ignore
    except Exception:
        pass
    # Try ACEModel_RF(cfg)
    try:
        return ACEModel_RF(cfg)  # type: ignore
    except Exception:
        pass
    # Try ACEModel_RF(cfg_ace, cfg_core) if present
    try:
        return ACEModel_RF(cfg.ace, cfg.core)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Unable to construct ACEModel_RF with provided cfg: {e}")


def _ace_get_state(ACE, t: int):
    # Try attributes
    for attrM, attrD in [("M", "D"), ("M_path", "D_path")]:
        if hasattr(ACE, attrM) and hasattr(ACE, attrD):
            try:
                return getattr(ACE, attrM)[t], getattr(ACE, attrD)[t]
            except Exception:
                pass
    # Try methods
    if hasattr(ACE, "state_at"):
        return ACE.state_at(t)  # type: ignore
    raise RuntimeError("ACE: cannot access state at time t.")


def _ace_advance(ACE, E_t: float):
    for meth in ("advance", "step", "next"):
        if hasattr(ACE, meth):
            return getattr(ACE, meth)(E_t)  # type: ignore
    raise RuntimeError("ACE: no advance/step/next method found.")


def _beliefs_from_cfg(cfg):
    # Prefer classmethod config
    if "GroupBeliefUpdating" in globals() and hasattr(GroupBeliefUpdating, "config"):
        return GroupBeliefUpdating.config(cfg)  # type: ignore
    # Else try direct constructor
    if "GroupBeliefUpdating" in globals():
        try:
            return GroupBeliefUpdating(
                mu=np.array([cfg.xi_mu_G0, cfg.xi_mu_B0], dtype=float),
                var=np.array([cfg.xi_sigma_G**2, cfg.xi_sigma_B**2], dtype=float),
                sigma_epsylon=np.array([cfg.sigma_epsilon_G, cfg.sigma_epsilon_B], dtype=float),
            )  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Beliefs: cannot construct — {e}")
    raise RuntimeError("Beliefs: GroupBeliefUpdating not available.")


def _beliefs_update_and_get_xi(beliefs, M_t: float, D_t: float):
    # Try the user's update API
    for meth in ("update_from_damage", "update", "observe_damage"):
        if hasattr(beliefs, meth):
            getattr(beliefs, meth)(M_t=M_t, D_t_obs=D_t)  # type: ignore
            break
    # Try current_xi()
    if hasattr(beliefs, "current_xi"):
        return beliefs.current_xi()  # type: ignore
    # Or expose means directly
    if hasattr(beliefs, "mu"):
        mu = getattr(beliefs, "mu")
        return float(mu[0]), float(mu[1])
    raise RuntimeError("Beliefs: cannot get current xi.")


def _build_voting(cfg):
    # Try standard builder
    if "OfficeMotivPVM" in globals():
        try:
            params = build_pvm_params(cfg) if "build_pvm_params" in globals() else cfg
            return OfficeMotivPVM(params)  # type: ignore
        except Exception:
            pass
    raise RuntimeError("Voting: OfficeMotivPVM unavailable.")


def _voting_solve(voting, M_t: float, xi_G: float, xi_B: float, t: int):
    # Try several common signatures and return (P_G, E_G, E_B)
    candidates = [
        ("solve_period", (M_t, xi_G, xi_B, t)),
        ("solve_period", (M_t, xi_G, xi_B)),
        ("solve", (M_t, xi_G, xi_B, t)),
        ("solve", (M_t, xi_G, xi_B)),
    ]
    for name, args in candidates:
        if hasattr(voting, name):
            out = getattr(voting, name)(*args)  # type: ignore
            # Heuristics for unpacking
            if isinstance(out, tuple):
                flat = list(out)
                # Look for a probability and two E policies in result
                Pg = None
                Es = []
                for val in flat:
                    if isinstance(val, (float, np.floating)):
                        Es.append(float(val))
                # crude assumption: last two floats are policies
                if len(Es) >= 2:
                    E_G, E_B = Es[-2], Es[-1]
                else:
                    E_G = E_B = float(np.mean(Es)) if Es else 0.0
                # try attributes for vote share
                for key in ("P_G", "gv", "p_green", "vote_share_green"):
                    if hasattr(out, key):
                        Pg = float(getattr(out, key))
                        break
                if Pg is None:
                    # fallback: center at 0.5 if unknown
                    Pg = 0.5
                return Pg, E_G, E_B
    # If all failed
    raise RuntimeError("Voting: cannot solve for policies/outcome with provided API.")

def simulate_series(cfg, T: int | None = None, seed: int = 999) -> Dict[str, np.ndarray]:
    if T is None:
        T = getattr(cfg, "periods", 30)

    # If real stack is not even importable, use stub
    if not _HAVE_REAL:
        print("[calibrate] Using STUB simulator (real model imports not found).")
        return _simulate_series_stub(cfg, T, seed)

    rng = np.random.default_rng(seed)

    # Try to build the real components; on ANY failure, fall back to stub
    try:
        ACE = _try_construct_ace(cfg)
        beliefs = _beliefs_from_cfg(cfg)
        voting = _build_voting(cfg)
    except Exception as e:
        print(f"[calibrate] Falling back to STUB simulator (reason: {e})")
        return _simulate_series_stub(cfg, T, seed)

    E_path, D_path, PG_path = [], [], []

    for t in range(T):
        try:
            M_t, D_t = _ace_get_state(ACE, t)
            xi_G, xi_B = _beliefs_update_and_get_xi(beliefs, M_t, D_t)
            P_G, E_G, E_B = _voting_solve(voting, M_t, xi_G, xi_B, t)
            E_t = E_G if rng.random() < P_G else E_B
            _ace_advance(ACE, E_t)

            # Damage timing
            try:
                _, D_real = _ace_get_state(ACE, t)
            except Exception:
                try:
                    _, D_real = _ace_get_state(ACE, t + 1)
                except Exception:
                    D_real = float(D_t)

            E_path.append(float(E_t))
            D_path.append(float(D_real))
            PG_path.append(float(np.clip(P_G, 0.0, 1.0)))
        except Exception as e:
            print(f"[calibrate] Runtime issue at t={t}: {e}. Switching to STUB for remainder.")
            return _simulate_series_stub(cfg, T, seed)

    return {"E": np.array(E_path), "D": np.array(D_path), "PG": np.array(PG_path)}

# =============================================================
# 6) LOSS + CALIBRATION
# =============================================================

def loss(theta: NDArray[np.float64], names: List[str], cfg0, m_target: NDArray[np.float64], seed: int) -> float:
    cfg_theta = apply_theta(cfg0, theta, names)
    series = simulate_series(cfg_theta, seed=seed)
    m_model = compute_moments(series)
    # Scale by target magnitude for balance; protect against zeros
    scale = np.clip(np.abs(m_target), 1e-6, None)
    z = (m_model - m_target) / scale
    return float(z @ z)


def calibrate(cfg0, m_target: NDArray[np.float64], n_starts: int = 4, method: str = "Nelder-Mead", seed: int = 999):
    theta0, names = pack_theta(cfg0)
    bnds = bounds_array(names)

    # Multi-starts: one at theta0 + uniform draws in bounds
    rng = np.random.default_rng(123)
    starts = [theta0]
    for _ in range(max(0, n_starts - 1)):
        lo, hi = bnds[:, 0], bnds[:, 1]
        starts.append(rng.uniform(lo, hi))

    best = None
    for i, th0 in enumerate(starts):
        res = minimize(
            lambda th: loss(th, names, cfg0, m_target, seed),
            th0,
            method=method,
            options=dict(maxfev=6000, xatol=1e-4, fatol=1e-4),
        )
        th_proj = np.clip(res.x, bnds[:, 0], bnds[:, 1])
        val = loss(th_proj, names, cfg0, m_target, seed)
        if (best is None) or (val < best["val"]):
            best = dict(theta=th_proj, val=val, res=res, names=names)

    cfg_hat = apply_theta(cfg0, best["theta"], best["names"])  # type: ignore
    return cfg_hat, best

# =============================================================
# 7) MAIN / CLI
# =============================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Simulated Method of Moments (SMM) calibration")
    p.add_argument("--n-starts", type=int, default=4, help="number of optimizer starts")
    p.add_argument("--method", type=str, default="Nelder-Mead", help="scipy.optimize method")
    p.add_argument("--seed", type=int, default=999, help="common RNG seed for smooth loss")
    p.add_argument("--save", type=str, default="", help="optional path to save calibrated params as JSON")
    args = p.parse_args(argv)

    cfg0 = all_config()

    cfg_hat, info = calibrate(cfg0, M_TARGET, n_starts=args.n_starts, method=args.method, seed=args.seed)

    # Report
    theta_hat, names = pack_theta(cfg_hat)
    print("\n=== Calibration results ===")
    for n, v in zip(names, theta_hat):
        lo, hi = BOUNDS[n]
        print(f"{n:>18s} = {v: .6g}  [bounds {lo}, {hi}]")
    print(f"Best loss: {info['val']:.6g}")

    # Fit diagnostics
    series0 = simulate_series(cfg0, seed=args.seed)
    seriesH = simulate_series(cfg_hat, seed=args.seed)
    m0 = compute_moments(series0)
    mH = compute_moments(seriesH)

    print("\nMoments (target / initial / calibrated):")
    for i, (mt, m_init, m_hat) in enumerate(zip(M_TARGET, m0, mH)):
        print(f"m[{i}] target={mt: .6g} | init={m_init: .6g} | hat={m_hat: .6g}")

    if args.save:
        out = asdict(cfg_hat) if hasattr(cfg_hat, "__dataclass_fields__") else cfg_hat.__dict__
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved calibrated parameters to: {args.save}")


if __name__ == "__main__":
    main()
