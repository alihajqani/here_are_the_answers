import pandas as pd
from pathlib import Path

def load_and_clean(path: str | Path,
                   cols_keep=("Id", "PostTypeId", "CreationDate", "Score",
                              "OwnerUserId", "Tags", "ParentId")) -> pd.DataFrame:
    
    df = pd.read_csv(path, usecols=cols_keep)

    df = df[df["PostTypeId"].isin([1, 2])].copy()

    df["CreationDate"] = pd.to_datetime(df["CreationDate"], utc=True)
    df["Timestamp"] = df["CreationDate"].view("int64") // 10 ** 9
    df.drop(columns="CreationDate", inplace=True)

    df["OwnerUserId"] = df["OwnerUserId"].astype("Int64")
    df["ParentId"] = df["ParentId"].dropna().astype("Int64")

    df["Tags"] = (df["Tags"]
                  .fillna("")
                  .str.strip("|")
                  .str.split("|")
                  .apply(lambda lst: lst if lst != [""] else None))
    

    return df


if __name__ == "__main__":
    dataset_path = "sample_data.csv"
    df = load_and_clean(dataset_path)


    print(df.head())
    print(df.info())
    print(df.dtypes)
    print(f"Total rows after cleaning: {len(df):,}")