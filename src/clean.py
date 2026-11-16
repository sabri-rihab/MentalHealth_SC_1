import pandas as pd
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, DROP_COLS


# __load data_______________________________________________________________
def load_raw_data(path= RAW_DATA_PATH)->pd.DataFrame:
    return pd.read_csv(path)


# __drop&fix data_______________________________________________________________
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    condition = df["Screen_Time_Hours"] < df["Social_Media_Hours"]
    df.loc[condition, "Screen_Time_Hours"] = df.loc[condition, "Social_Media_Hours"]

    cols_to_drop = [clm for clm in DROP_COLS if clm in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df

     
# __save clean data_______________________________________________________________
def save_data(df: pd.DataFrame, path=CLEANED_DATA_PATH)->None:
    df.to_csv(path, index=False)

