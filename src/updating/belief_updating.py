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

        mu_H0 = getattr(updating_cfg, "xi_mu_H0")
        mu_L0 = getattr(updating_cfg, "xi_mu_L0")

        # Priors given as std devs in config; store as variances here
        sigma_H0 = updating_cfg.xi_sigma_H
        sigma_L0 = updating_cfg.xi_sigma_L
        var0 = np.array([sigma_H0**2, sigma_L0**2], dtype=float)

        # damage shock / observation noise -> std devs 
        se_H = getattr(updating_cfg, "sigma_epsilon_H", 0.03)
        se_L = getattr(updating_cfg, "sigma_epsilon_L", 0.03)
        sigma_eps = np.array([se_H, se_L], dtype=float)

        return cls(
            mu=np.array([mu_H0, mu_L0], dtype=float),
            var=var0,
            sigma_epsilon=sigma_eps,
        )

    def update_new(self, M_t: float, D_t: float, M_pre: float):
        # print("")
        XI_mu_H = self.mu[0]
        XI_mu_L = self.mu[1]
        # print(f"mu_H_prior: {XI_mu_H:.6f}")
        # print(f"mu_L_prior: {XI_mu_L:.6f}")
        # print("")
        XI_var_H = self.var[0]
        XI_var_L = self.var[1]
        # print(f"var_H_prior: {XI_var_H:.6f}")
        # print(f"var_L_prior: {XI_var_L:.6f}")
        # print("")
        shock_sigma_H = self.sigma_epsilon[0]
        shock_sigma_L = self.sigma_epsilon[1]
        # print(f"shock_sigma_H: {shock_sigma_H:.6f}")
        # print(f"shock_sigma_L: {shock_sigma_L:.6f}")
        # print("")
        # print(f"M_t:",M_t)        
        # print(f"D_t:",D_t)
        # print("")

        M2 = M_t * M_t
        damage_frac = (float(-np.log(1.0 - D_t))/M_t)

        mu_numerator_H = M2 * XI_var_H * damage_frac + (shock_sigma_H**2) * XI_mu_H
        mu_denominator_H = M2 * XI_var_H + (shock_sigma_H**2)
        XI_mu_H_POST = mu_numerator_H / mu_denominator_H
        # print(f"XI_mu_H_POST: {XI_mu_H_POST:.10f}")

        # alpha = 0.5
        # M = (M_t ** ( alpha))
        var_numerator_H = XI_var_H * (shock_sigma_H**2) 
        var_denominator_H = (1) * XI_var_H + (shock_sigma_H**2)
        Xi_var_H_POST = var_numerator_H / var_denominator_H
        # print(f"XI_var_H_POST: {Xi_var_H_POST:.10f}")



        mu_numerator_L = M2 * XI_var_L * damage_frac + (shock_sigma_L**2) * XI_mu_L
        mu_denominator_L = M2 * XI_var_L + (shock_sigma_L**2)
        XI_mu_L_POST = mu_numerator_L / mu_denominator_L
        # print(f"XI_mu_L_POST: {XI_mu_L_POST:.10f}")

        # alpha = 0.5
        # M = (M_t ** ( alpha))
        var_numerator_L = XI_var_L * (shock_sigma_L**2) 
        var_denominator_L = (M2) * XI_var_L + (shock_sigma_L**2)
        Xi_var_L_POST = var_numerator_L / var_denominator_L
        # print(f"XI_var_L_POST: {Xi_var_L_POST:.10f}")


        self.mu[0] = XI_mu_H_POST
        self.mu[1] = XI_mu_L_POST
        self.var[0] = Xi_var_H_POST
        self.var[1] = Xi_var_L_POST






        #input("Press Enter to continue...")

        

    def update(self, M_t: float, D_t: float, M_pre: float):
        """
        Bayesian update with y_t = log(1 - D_t) = xi * M_t + eps, eps ~ N(0, sigma_eps^2).
        """

        y_t = float(np.log(1.0 - D_t))
        M_t = (M_t / M_pre)
        M2 = M_t * M_t

        # Vectorized per group
        prior_mu = self.mu
        prior_var = self.var
        prior_prec = 1.0 / prior_var
        obs_prec = M2 / (self.sigma_epsilon ** 2)

        post_var = 1.0 / (prior_prec + obs_prec)
        post_mu = post_var * (prior_prec * prior_mu + ((M_t/M_pre) * y_t) / (self.sigma_epsilon ** 2))

        self.mu = post_mu
        self.var = post_var



    def update01(self, M_t: float, D_t: float, M_pre: float):
        # 1) ACE-consistent signal
        z_t = np.log(1.0 - D_t)      # = ξ_true * M_t_real + noise
        M_real = M_t

        # 2) "observation" of ξ itself (what Jensen & Traeger call the climate-sensitivity observation)
        xi_obs = z_t / M_real         # ≈ ξ_true + noise/M

        # 3) heteroskedastic noise scaled by M_t / M_init
        scale = M_real / M_pre
        obs_var  = (self.sigma_epsilon * scale) ** 2    # shape (2,)
        obs_prec = 1.0 / obs_var

        # 4) prior
        prior_mu  = self.mu
        prior_var = self.var
        prior_prec = 1.0 / prior_var

        # 5) posterior variance
        post_var = 1.0 / (prior_prec + obs_prec)

        # 6) posterior mean – **use xi_obs here, not z_t**
        post_mu = post_var * (prior_prec * prior_mu + obs_prec * xi_obs)

        self.mu  = post_mu
        self.var = post_var



    def current_xi(self) -> tuple[float, float]:
        """Return current posterior means (xî_H, xî_L)."""
        return float(self.mu[0]), float(self.mu[1])

