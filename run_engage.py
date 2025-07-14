import os
import pickle

from EandG import build_E_G
from logger import get_logger
from gibbs_sampler import EngageGibbs
from data_preprosess import load_and_clean

if __name__ == "__main__":

    logger = get_logger(__name__, log_file="results/engage.log")
    logger.info("Starting the Engage Gibbs sampling process")
    
    try:
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        logger.info("Data and results directories are ready")

        # Load and clean the data
        logger.info("Loading and cleaning data started")
        df = load_and_clean("data/sample_data.csv")
        logger.info("Data loaded and cleaned successfully")
        
        # Build E and G matrices
        logger.info("Building E and G matrices started")
        E, G, user_ids, tag_ids = build_E_G(df)
        logger.info("E and G matrices built successfully")
        
        # Initialize the EngageGibbs sampler
        logger.info("Initializing EngageGibbs sampler started")
        sampler = EngageGibbs(E, G, alpha=5.0, beta=1.0, K=15)
        logger.info("EngageGibbs sampler initialized")

        # Run the Gibbs sampling
        logger.info("Running Gibbs sampling started")
        samples = sampler.run(n_iter=3000, burn_in=1000, thin=10)
        logger.info("Gibbs sampling completed successfully")
        
        # Save the samples
        logger.info("Saving samples to results/engage_samples.pkl")
        with open("results/engage_samples.pkl", "wb") as f:
            pickle.dump(samples, f)
            logger.info("Samples saved to results/engage_samples.pkl")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")