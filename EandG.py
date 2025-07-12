import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from data_preprosess import load_and_clean


def build_E_G(df: pd.DataFrame, lambda_decay=0.2, now_ts=None):
    if not os.path.exists('results'):
        os.makedirs('results')

    if now_ts is None:
        now_ts = df['Timestamp'].max()

    questions = df[df.PostTypeId == 1][['Id', 'OwnerUserId', 'Tags']].rename(
        columns={'OwnerUserId':"AskerId"})
    answers = df[df.PostTypeId == 2][['OwnerUserId', 'ParentId', 'Score', 'Timestamp']].rename(
        columns={'OwnerUserId':'AnswererId'})
    
    ans = answers.merge(questions, left_on='ParentId', right_on='Id', how='inner')

    age_years = (now_ts - ans['Timestamp']) / (365*24*60*60)
    decay = np.exp(-lambda_decay * age_years)

    plt.figure(figsize=(8, 5))
    plt.plot(age_years, decay, marker='o', linestyle='-')
    plt.xlabel('age (year)')
    plt.ylabel('decay')
    plt.title('Decay vs Age Years')
    plt.savefig('results/decay_vs_age_years.png')
    plt.grid(True)
    plt.tight_layout()
    plt.close()

    ans['w_num'] = ans['Score'] * decay
    ans['w_den'] = decay

    ans = ans.explode("Tags").dropna(subset=["Tags"])

    grp = (ans.groupby(["AnswererId", "Tags"])
            .agg(num=("w_num", "sum"), den=("w_den", "sum"))
            .reset_index())
    grp["E_val"] = grp["num"] / grp["den"]

    user_ids = pd.Index(pd.concat([ans['AnswererId'], ans['AskerId']])
                        .unique(), name='UserId')
    tag_ids = pd.Index(grp['Tags'].unique(), name='Tag')


    rows = user_ids.get_indexer(grp["AnswererId"].values)
    cols = tag_ids.get_indexer(grp["Tags"].values)

    E = csr_matrix((grp["E_val"], (rows, cols)),
                   shape=(len(user_ids), len(tag_ids)))
    

    u2i = {user_id: idx for idx, user_id in enumerate(user_ids)}
    edges = ans[["AnswererId", "AskerId"]].drop_duplicates()
    rows_g = edges["AnswererId"].map(u2i)
    cols_g = edges["AskerId"].map(u2i)
    G = csr_matrix((np.ones(len(edges), dtype=np.int8),
                    (rows_g, cols_g)),
                   shape=(len(user_ids), len(user_ids)))

    
    ans.to_csv('results/ans.csv', index=False)
    grp.to_csv('results/grp.csv', index=False)

    dense = E.toarray()
    e_df = pd.DataFrame(dense)
    e_df.to_csv('results/E.csv', index=False)

    dense = G.toarray()
    g_df = pd.DataFrame(dense)
    g_df.to_csv('results/G.csv', index=False)



    return E, G, user_ids, tag_ids




if __name__ == "__main__":
    df = load_and_clean("data/sample_data.csv") 
    build_E_G(df=df)
