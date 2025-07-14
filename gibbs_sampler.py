# gibbs_sampler.py
import numpy as np
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from priors import NormalWishartPrior
from scipy.stats import wishart, multivariate_normal as mvn

from logger import get_logger
logger = get_logger(__name__, log_file="results/engage.log")


class EngageGibbs:
    """
    Gibbs sampler for the ENGAGE model (expert-finding).
    """

    def __init__(
        self,
        E: csr_matrix,            # shape (n_users, n_tags)
        G: csr_matrix,            # shape (n_users, n_users)
        alpha: float = 5.0,       # precision of E part
        beta: float = 1.0,        # precision of G part
        K: int = 15,
        hyper_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.E, self.G = E.tocsr(), G.tocsr()
        self.alpha, self.beta, self.K = alpha, beta, K
        self.rng = rng or np.random.default_rng()

        n_users, n_tags = E.shape
        prior_kwargs = hyper_kwargs or {}
        self.prior_L = NormalWishartPrior(K, **prior_kwargs, rng=self.rng)
        self.prior_H = NormalWishartPrior(K, **prior_kwargs, rng=self.rng)

        # --- initial samples --------------------------------------------------
        self.L, self.mu_L, self.Lambda_L = self.prior_L.sample_latent_matrix(n_users)
        self.H, self.mu_H, self.Lambda_H = self.prior_H.sample_latent_matrix(n_tags)
        logger.info(f"Initialized EngageGibbs with L shape {self.L.shape}, H shape {self.H.shape}, E shape {E.shape}, G shape {G.shape}")

        # pre-compute sparse indices
        self.user_nonzero = [E[u].indices for u in range(n_users)]
        self.tag_nonzero  = [E[:, t].indices for t in range(n_tags)]
        self.G_rows       = [G[u].indices for u in range(n_users)]


    # ----- helpers for conditional draws ---------------------------------- #
    def _sample_theta(self, X, prior: NormalWishartPrior):
        """
        Sample (mu, Lambda) | X  according to Eq. (7).
        X : (K, c)
        """
        K, c = X.shape
        x_bar = X.mean(axis=1, keepdims=True)        # (K,1)
        S_x   = ((X - x_bar) @ (X - x_bar).T) / c    # (K,K)

        mu0, beta0, nu0, W0 = (
            prior.mu0,
            prior.beta0,
            prior.nu0,
            prior.W0,
        )

        beta_star = beta0 + c
        mu_star   = (beta0 * mu0 + c * x_bar.flatten()) / beta_star

        W_star_inv = (W0 / c) + c * S_x + (beta0 * c / beta_star) * np.outer(
            mu0 - x_bar.flatten(), mu0 - x_bar.flatten()
        )
        W_star = inv(W_star_inv)

        # Λ ~ Wishart(ν0+c, W_star)
        Lambda = wishart(df=nu0 + c, scale=W_star).rvs(random_state=self.rng)
        logger.info(f"Sampled Lambda from Wishart distribution (updatated precision matrix) for {X}")

        # μ | Λ ~ N(mu_star, (beta_star Λ)^-1)
        cov_mu = inv(beta_star * Lambda)
        mu     = self.rng.multivariate_normal(mu_star, cov_mu)
        logger.info(f"Sampled Mu from multivariate normal distribution (updatated mean vector) for {X}")

        return mu, Lambda

    # ------ One full Gibbs iteration ----------------------------------------- #
    def step(self):
        """One full Gibbs iteration."""
        logger.info("Starting a new Gibbs step...")
        # --- 1. update hyper-parameters Θ_L , Θ_H ----------------------------
        self.mu_L, self.Lambda_L = self._sample_theta(self.L, self.prior_L)
        self.mu_H, self.Lambda_H = self._sample_theta(self.H, self.prior_H)
        logger.info(f"Updated hyperparameters: mu_L, Lambda_L, mu_H, Lambda_H")

        # --- 2. update latent vectors L_u  (Eq. 5) ---------------------------
        K = self.K
        for u in range(self.L.shape[1]):
            tags = self.user_nonzero[u]
            if len(tags) == 0 and len(self.G_rows[u]) == 0:
                continue  # completely isolated ⇒ keep previous value

            # precision Λ*_{L,u}
            precision = self.Lambda_L.copy()
            if tags.size:
                H_sub = self.H[:, tags]               # (K, |tags|)
                precision += self.alpha * H_sub @ H_sub.T
            if self.G_rows[u].size:
                L_neighbors = self.L[:, self.G_rows[u]]
                precision += self.beta * L_neighbors @ L_neighbors.T

            # mean μ*_{L,u}
            rhs = self.Lambda_L @ self.mu_L
            if tags.size:
                w = self.E[u, tags].toarray().ravel()             # ← (|tags|,)
                rhs += self.alpha * (self.H[:, tags] @ w)
            if self.G_rows[u].size:
                g_w = self.G[u, self.G_rows[u]].toarray().ravel()  # ← (|nbrs|,)
                rhs += self.beta * (self.L[:, self.G_rows[u]] @ g_w)

            # jitter & symmetrize
            eps = 1e-8
            precision += eps * np.eye(K)
            precision = (precision + precision.T) / 2


            # -------------[FOR DEBUG]-------------
            # tol = 1e-4

            # if u%1000 == 0:
            #     print(u)

            # if np.allclose(precision, precision.T, atol=tol):
            #     pass
            # else:
            #     print(f'nok{u}')


            # eigvals = np.linalg.eigvalsh(precision)
            # min_ev = eigvals.min()
            # # print("min eigen val:", min_ev)
            # if min_ev >= 0:
            #     pass
            # else:
            #     print(f"NSD (min eigen = {min_ev})")

            # --------------------------------------------

            cov = inv(precision)
            
            # jitter & symmetrize
            cov = (cov + cov.T) / 2
            eig_min = np.linalg.eigvalsh(cov).min()
            if eig_min < eps:
                cov += (eps - eig_min) * np.eye(K)


            self.L[:, u] = self.rng.multivariate_normal(cov @ rhs, cov)


            if u % 1000 == 0:
                logger.info(f"Updated latent vector L[:, {u}] from {self.L.shape[1]} users.")

        # --- 3. update latent vectors H_t  (Eq. 6) ---------------------------
        for t in range(self.H.shape[1]):
            users = self.tag_nonzero[t]
            if users.size == 0:
                continue

            precision = self.Lambda_H.copy()
            L_sub = self.L[:, users]
            precision += self.alpha * L_sub @ L_sub.T

            e_w = self.E[users, t].toarray().ravel()              # ← (|users|,)
            rhs = self.Lambda_H @ self.mu_H + self.alpha * (L_sub @ e_w)

            cov = inv(precision)
            self.H[:, t] = self.rng.multivariate_normal(cov @ rhs, cov)

            if t % 500 == 0:
                logger.info(f"Updated latent vector H[:, {t}] from {self.H.shape[1]} tags.")

    # --------------------------------------------------------------------- #
    def run(self, n_iter=3000, burn_in=1000, thin=10):
        """
        Executes the Gibbs chain and returns a list of retained samples.
        """
        samples = []
        for h in range(1, n_iter + 1):
            logger.info(f"------------------- Running Gibbs iteration {h}/{n_iter} -------------------")
            self.step()
            logger.info(f"------------------- Completed Gibbs iteration {h} -------------------")
            if h > burn_in and (h - burn_in) % thin == 0:
                samples.append((self.L.copy(), self.H.copy()))  
        return samples
 