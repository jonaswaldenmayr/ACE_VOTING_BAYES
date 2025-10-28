from __future__ import annotations
from dataclasses import dataclass
import numpy as np



@dataclass(slots=True)
class GroupBeliefUpdating:
    mu: np.ndarray            # shape (2,) : posterior means [mu_G, mu_B]
    var: np.ndarray           # shape (2,) : posterior variances [var_G, var_B]  (σ_{ξ,t}^2)
    sigma_epsilon: np.ndarray # shape (2,) : damage shock std devs [σ_ε,G, σ_ε,B]




    @classmethod
    def config(cls, updating_cfg):

        mu_G0 = getattr(updating_cfg, "xi_mu_G0")
        mu_B0 = getattr(updating_cfg, "xi_mu_B0")

        # Priors given as std devs in config; store as variances here
        sigma_G0 = updating_cfg.xi_sigma_G
        sigma_B0 = updating_cfg.xi_sigma_B
        var0 = np.array([sigma_G0**2, sigma_B0**2], dtype=float)

        # damage shock / observation noise -> std devs 
        se_G = getattr(updating_cfg, "sigma_epsilon_G", 0.03)
        se_B = getattr(updating_cfg, "sigma_epsilon_B", 0.03)
        sigma_eps = np.array([se_G, se_B], dtype=float)

        return cls(
            mu=np.array([mu_G0, mu_B0], dtype=float),
            var=var0,
            sigma_epsilon=sigma_eps,
        )

        

    def update(self, M_t: float, D_t: float):
        """
        Bayesian update with y_t = log(1 - D_t) = xi * M_t + eps, eps ~ N(0, sigma_eps^2).
        """

        y_t = float(np.log(1.0 - D_t))
        M2 = M_t * M_t

        # Vectorized per group
        prior_mu = self.mu
        prior_var = self.var
        prior_prec = 1.0 / prior_var
        obs_prec = M2 / (self.sigma_epsilon ** 2)

        post_var = 1.0 / (prior_prec + obs_prec)
        post_mu = post_var * (prior_prec * prior_mu + (M_t * y_t) / (self.sigma_epsilon ** 2))

        self.mu = post_mu
        self.var = post_var


    def current_xi(self) -> tuple[float, float]:
        """Return current posterior means (xî_G, xî_B)."""
        return float(self.mu[0]), float(self.mu[1])

