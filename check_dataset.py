import pandas as pd

posts_df = pd.read_xml('data/train_data.xml')
answers = posts_df[posts_df['PostTypeId'] == 2]
answer_counts = answers.groupby('OwnerUserId').size()

prolific_users = answer_counts[answer_counts >= 5].index
num_prolific = len(prolific_users)
print(f"Users with ≥ 5 answers: {num_prolific}")

total_answer_users = answers['OwnerUserId'].nunique()
print(f"Total distinct answerers: {total_answer_users}")

question_ids = (
    answers
    .loc[answers['OwnerUserId'].isin(prolific_users), 'ParentId']
    .dropna()
    .unique()
)

filtered = posts_df[
    posts_df['Id'].isin(question_ids)       # the question posts
    | posts_df['ParentId'].isin(question_ids)  # any answers to them
]

print(f"Filtered dataset size: {len(filtered)} rows")

filtered.to_xml('data/train_filtered.xml', index=False)
