import pickle, numpy as np
from scipy.sparse import load_npz
from data_preprosess import load_and_clean
from EandG import build_E_G
from gibbs_sampler import EngageGibbs

df = load_and_clean("data/sample_data.csv")

E, G, user_ids, tag_ids = build_E_G(df)

sampler = EngageGibbs(E, G, alpha=5.0, beta=1.0, K=15)

samples = sampler.run(n_iter=3000, burn_in=1000, thin=10)

with open("results/engage_samples.pkl", "wb") as f:
    pickle.dump(samples, f)