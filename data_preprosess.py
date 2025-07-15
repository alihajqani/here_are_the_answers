# data_preprosess.py

import pandas as pd
from pathlib import Path

from logger import get_logger
logger = get_logger(__name__, log_file="results/engage.log")

def load_and_clean(xml_path: str | Path,
                   cols_keep=("Id", "PostTypeId", "CreationDate", "Score",
                              "OwnerUserId", "Tags", "ParentId")) -> pd.DataFrame:
    
    xml_path = Path(xml_path)
    df_xml = pd.read_xml(xml_path)
    csv_path = xml_path.with_suffix(".csv")
    df_xml.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path, usecols=cols_keep)

    df = df[df["PostTypeId"].isin([1, 2])].copy()
    df = df[df["OwnerUserId"].notna()].copy()

    #chage score=0 to score=1
    df.loc[df['Score'] == 0, 'Score'] = 1

    df["CreationDate"] = pd.to_datetime(df["CreationDate"], format="mixed")
    df["Timestamp"] = df["CreationDate"].astype("int64") // 10 ** 9
    df.drop(columns="CreationDate", inplace=True)

    df["OwnerUserId"] = df["OwnerUserId"].astype("Int64")
    df["ParentId"] = df["ParentId"].dropna().astype("Int64")

    df["Tags"] = (df["Tags"]
                  .fillna("")
                  .str.strip("|")
                  .str.split("|")
                  .apply(lambda lst: lst if lst != [""] else None))
    
    logger.info(f"Data loaded and cleaned: {len(df):,} rows and {len(df.columns)} columns")

    clean_csv_path = xml_path.with_name(f"{xml_path.stem}_clean.csv")
    df.to_csv(clean_csv_path, index=False)
    logger.info(f"{xml_path.stem}_clean.csv save in {clean_csv_path}")

    return df



if __name__ == "__main__":
    dataset_path = "data/train_data.xml"
    df = load_and_clean(dataset_path)


    print(df.head())
    print(df.info())
    print(df.dtypes)
    print(f"Total rows after cleaning: {len(df):,}")