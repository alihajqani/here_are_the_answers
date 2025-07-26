
# ENGAGE: Time-evolving Tag-based Expertise Model

This repository contains a Python implementation of the ENGAGE model for time-sensitive expert recommendation in community Q&A platforms.

## Project Structure

├── check_dataset.py

├── data_preprosess.py

├── EandG.py

├── E_hat_E_filled.py

├── gibbs_sampler.py

├── logger.py

├── priors.py

├── requirements.txt

├── run_engage.py

└── test_engage.py


- **check_dataset.py**  
  Filter raw StackExchange XML dump into a focused dataset

- **data_preprosess.py**  
  Load, clean, timestamp and split posts
  ```
  Note: In data preprosess we change score 0 to 1 and timestamp base on year.(for E)
  ```

- **EandG.py**  
  Build engagement (E) and graph (G) matrices
  ```
  Note: lambda_decay=0.2
  ```

- **E_hat_E_filled.py**  
  Compute estimated engagement matrix and fill missing values

- **gibbs_sampler.py**  
  Core Gibbs sampler for latent factors
  ```
  Note:
  - We use jitter & symmetrize while create precision matrix with eps=1e-5
  - alpha=5, beta=1, K=15
  ```

- **logger.py**  
  Console and rotating-file logger configuration

- **priors.py**  
  Gaussian–Wishart prior implementation
  ```
  Note: beta0 = 1, nu0=0
  ```

- **run_engage.py**  
  End-to-end training: sample parameters and save results

- **test_engage.py**  
  Evaluate model with Precision@k, MRR, nDCG, etc.

- **requirements.txt**  
  Python dependencies

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/alihajqani/here_are_the_answers.git
   cd here_are_the_answers

2. Create a virtual environment and install dependencies

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Data Preparation

1. Download the raw XML dump to `data/train_data.xml`.
2. Filter for active users and produce `data/train_filtered.xml`:

   ```bash
   python check_dataset.py
   ```

## Training

Run the full pipeline:

```bash
python run_engage.py
```

This will:

* Build and save the E and G matrices
* Run the Gibbs sampler (default 3000 iterations, 1000 burn-in)
* Save posterior samples and filled engagement matrix

## Evaluation

Evaluate on held-out data:

```bash
python test_engage.py --test data/test_data.xml --top_k 5
```

Results (Precision\@5, Recall\@5, MRR, nDCG) are printed to console and saved in `results/engage_test.log`.

