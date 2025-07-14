import numpy as np
from numpy.linalg import inv
from scipy.stats import wishart, multivariate_normal as mvn

from logger import get_logger
logger = get_logger(__name__, log_file="results/engage.log")

class NormalWishartPrior:
    """
        Gaussian–Wishart prior for ENGAGE latent factors.

        Parameters
        ----------
        K : int
            Latent dimensionality.
        mu0 : (K,) array_like, default 0
        beta0 : float, default 1.0
        nu0 : int,   default K+2   (must be > K-1)
        W0  : (K,K) array_like, default identity
        rng : np.random.Generator, optional
    """

    def __init__(self, K, mu0=None, beta0=1.0, nu0=None, W0=None, rng=None):
        logger.info(f"Initializing NormalWishartPrior starting...")
        self.k = K
        self.mu0 = np.zeros(K) if mu0 is None else np.asarray(mu0)
        self.beta0 = beta0
        self.nu0 = K + 2 if nu0 is None else nu0
        self.W0 = np.eye(K) if W0 is None else np.asarray(W0)
        self.rng = rng or np.random.default_rng()

        assert self.mu0.shape == (K,), "mu0 must be a vector of shape (K,)"
        assert self.W0.shape == (K, K), "W0 must be a square matrix of shape (K, K)"
        assert self.nu0 > K - 1, "nu0 must be greater than K-1"
        logger.info(f"Initializing NormalWishartPrior completed.")

    def sample_lambda(self):
        """Λ ~ Wishart(ν0, W0)  — returns precision matrix Λ"""
        logger.info(f"Sampling Lambda from Wishart distribution.")
        return wishart(df=self.nu0, scale=self.W0).rvs(random_state=self.rng)
    
    def sample_mu(self, Lambda):
        """μ | Λ ~ N(μ0, (β0 Λ)^-1)"""
        cov = inv(self.beta0 * Lambda)
        logger.info(f"Sampling Mu from multivariate normal distribution")
        return self.rng.multivariate_normal(self.mu0, cov)
    
    def sample_latent_matrix(self, n_cols):
        """
        Draw n_cols latent K-vectors all at once.

        Returns
        -------
        M : (K, n_cols) np.ndarray
        μ : (K,)        np.ndarray
        Λ : (K,K)       np.ndarray
        """
        Λ = self.sample_lambda()
        μ = self.sample_mu(Λ)
        cov = inv(Λ)
        M  = self.rng.multivariate_normal(μ, cov, size=n_cols).T  # shape (K, n)
        logger.info(f"Sampled latent matrix M with shape {M.shape}, mean vector mu with shape {μ.shape}, and precision matrix Lambda with shape {Λ.shape}")
        return M, μ, Λ
    

if __name__ == "__main__":
    
    # Example usage
    prior = NormalWishartPrior(K=3)
    M, mu, Lambda = prior.sample_latent_matrix(n_cols=10)
    print("Sampled latent matrix M:\n", M)
    print("Sampled mean vector mu:\n", mu)
    print("Sampled precision matrix Lambda:\n", Lambda)
    print('hyperparameters mu0:', prior.mu0)
    print('hyperparameters beta0:', prior.beta0)
    print('hyperparameters nu0:', prior.nu0)
    print('hyperparameters W0:\n', prior.W0)
