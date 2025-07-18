import pickle
import numpy as np

from logger import get_logger
logger = get_logger(__name__, log_file="results/engage.log")

def build_E_new(path: str = "results/engage_samples.pkl", E: np.ndarray = None):
    """
    This function computes the average engagement matrix E_hat
    from the Gibbs sampling results and fills in the missing values
    in the original engagement matrix E.
    """

    # --------- Load the data from the pickle file ---------
    with open(path, "rb") as f:
        samples = pickle.load(f)          # list[(L, H)]
        logger.info(f"Loaded  samples from {path}")

    # --------- Compute the average engagement matrix E_hat ---------
    logger.info("Computing the average engagement matrix E_hat")
    S = len(samples)
    K, n_users = samples[0][0].shape   # L.shape = (K, n_users)
    n_tags     = samples[0][1].shape[1]  # H.shape = (K, n_tags)
    logger.info(f"Number of n_users: {n_users}, n_tags: {n_tags}")

    E_hat = np.zeros((n_users, n_tags), dtype=np.float64)
    logger.info(f"Shape of E_hat is {E_hat.shape}")

    for s, (L, H) in enumerate(samples, 1):
        E_hat += (L.T @ H - E_hat) / s


    E_hat /= S
    logger.info(f"Computed E_hat with shape {E_hat.shape}")


    E_filled = E.copy()
    mask = (E == 0)
    E_filled[mask] = E_hat[mask]

    E_new = E_filled

    np.save("results/E_new.npy", E_new)
    logger.info(f"Saved the filled engagement matrix E_new with shape {E_new.shape} to results/E_new.npy")

    return E_new
