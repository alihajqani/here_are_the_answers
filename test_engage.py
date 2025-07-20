#!/usr/bin/env python
"""
Test pipeline for the ENGAGE expert‑finding model.

Prerequisites (all produced by the training pipeline)::
    results/E_new.npy          – filled engagement matrix (users × tags)
    results/user_ids.csv       – one user‑id per line, order corresponds to axis‑0 of E_new
    results/tag_ids.csv        – one tag per line,   order corresponds to axis‑1 of E_new

Usage
-----
$ python test_engage.py --test data/test_dataset.xml --top_k 5

The script will
1.  Clean the XML/CSV test dump exactly as in training (via *load_and_clean*).
2.  Rank the top‑k candidate answerers for every **question** post.
3.  Compare the ranking with the actual answerers in the test set and
    report Precision@k, Recall@k, MRR@k and nDCG@k.

Author:  (c) 2025 – ENGAGE replication code
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set, Sequence

from data_preprosess import load_and_clean
from logger import get_logger

# --------------------------------------------------------------------------- #
logger = get_logger(__name__, log_file="results/engage_test.log")

# ------------------------------ helper utils ------------------------------ #

def _dcg(rel: Sequence[int]) -> float:
    """Discounted cumulative gain – no log2(1) shift required (i = 0 ⇒ log2(2))."""
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def precision_at_k(pred: List[int], truth: Set[int], k: int) -> float:
    return len(set(pred[:k]) & truth) / k


def recall_at_k(pred: List[int], truth: Set[int], k: int) -> float:
    return len(set(pred[:k]) & truth) / len(truth) if truth else 0.0


def mrr_at_k(pred: List[int], truth: Set[int], k: int) -> float:
    for rank, u in enumerate(pred[:k], start=1):
        if u in truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred: List[int], truth: Set[int], k: int) -> float:
    gains = [1 if u in truth else 0 for u in pred[:k]]
    dcg_val = _dcg(gains)
    ideal_gains = sorted(gains, reverse=True)
    idcg_val = _dcg(ideal_gains)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# --------------------------- ranking procedure ---------------------------- #

def rank_users(tags: List[str],
               E_new: np.ndarray,
               tag_index: pd.Index,
               user_index: pd.Index,
               k: int = 5) -> List[int]:
    """Return *user‑ids* of top‑k candidate experts for a given tag list."""
    tag_ids = [tag_index.get_loc(t) for t in tags if t in tag_index]
    if not tag_ids:
        logger.debug("No known tags in question – returning empty list")
        return []

    # Mean engagement score over the question tags  → shape (n_users,)
    scores = E_new[:, tag_ids].mean(axis=1)
    top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    # sort these *k* indices exactly by score desc.
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return user_index[top_idx].tolist()


# ----------------------------- evaluation --------------------------------- #

def evaluate(questions: pd.DataFrame,
             answers: pd.DataFrame,
             E_new: np.ndarray,
             tag_index: pd.Index,
             user_index: pd.Index,
             k: int = 5):
    metrics = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "ndcg": [],
    }

    # build mapping: qId → {answererIds}
    gt_dict: dict[int, Set[int]] = (answers.groupby("ParentId")["OwnerUserId"]
                                    .apply(set)
                                    .to_dict())

    for _, q in questions.iterrows():
        qid = int(q["Id"])
        true_ans = gt_dict.get(qid, set())
        if not true_ans:
            continue  # no ground truth – skip

        pred = rank_users(q["Tags"], E_new, tag_index, user_index, k)
        if not pred:
            continue  # unknown tags – skip

        metrics["precision"].append(precision_at_k(pred, true_ans, k))
        metrics["recall"].append(recall_at_k(pred, true_ans, k))
        metrics["mrr"].append(mrr_at_k(pred, true_ans, k))
        metrics["ndcg"].append(ndcg_at_k(pred, true_ans, k))

    # aggregate
    return {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}


# ------------------------------- main ------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate ENGAGE on a test set")
    # parser.add_argument("--test", required=True,
    #                     help="Path to StackExchange test dump (.xml or .csv)")
    # parser.add_argument("--top_k", type=int, default=5,
    #                     help="k for Precision@k, etc. [default: 5]")
    args = parser.parse_args()
    args.test = "data/test_data.xml"
    args.top_k = 5
    # 1. data ----------------------------------------------------------------
    # logger.info("Loading and cleaning test data…")
    df_test = load_and_clean(args.test)

    # split into Q/A
    questions = df_test[df_test["PostTypeId"] == 1].copy()
    answers   = df_test[df_test["PostTypeId"] == 2].copy()
    logger.info(f"Parsed {len(questions):,} questions and {len(answers):,} answers from test set.")

    # 2. artefacts -----------------------------------------------------------
    logger.info("Loading E_new and ID mappings…")
    E_new = np.load("results/E_new.npy")
    user_index = pd.Index(pd.read_csv("results/user_ids.csv", header=None)[0], name="UserId")
    tag_index  = pd.Index(pd.read_csv("results/tag_ids.csv",  header=None)[0], name="Tag")

    assert E_new.shape == (len(user_index), len(tag_index)), "Shape mismatch between E_new and indices"

    # 3. evaluation ----------------------------------------------------------
    logger.info("Scoring questions and computing metrics…")
    report = evaluate(questions, answers, E_new, tag_index, user_index, k=args.top_k)

    # 4. print / log ---------------------------------------------------------
    logger.info("\n" + "\n".join(f"{m.upper()}@{args.top_k}: {v:.4f}" for m, v in report.items()))
    print("=" * 60)
    print(f"ENGAGE evaluation on {Path(args.test).name} (k={args.top_k})")
    for m, v in report.items():
        print(f"{m.upper():8s}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


