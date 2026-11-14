import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# _________________________________________________________________
def load_raw_data():
    return pd.read_csv('../data/cleaned_ds.csv')




# _________________________________________________________________
def drop_columns(df):
    df.drop(columns=['Age'], inplace=True)
    df.drop(columns=['Student_ID'], inplace=True)
    return 'unusefull column are deleted!'



     
# _________________________________________________________________
def fix_screen_time(df):
    df = df[(df['Screen_Time_Hours'] < df['Social_Media_Hours'])]
    return df





# _________________________________________________________________
def impute_missing_values(df):
    col_num = df.select_dtypes(exclude=['object', 'string']).columns
    col_catg = df.select_dtypes(include=['object', 'string']).columns

    encoder =  OrdinalEncoder()
    scaler = StandardScaler()

    df[col_catg] = encoder.fit_transform(df[col_catg])
    df[col_num] = scaler.fit_transform(df[col_num])

    imputer = IterativeImputer(random_state=42, max_iter=10)

    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    df_imputed[col_num] = scaler.inverse_transform(df_imputed[col_num])
    df_imputed['Year_of_Study'] = df_imputed['Year_of_Study'].round().astype('int64')
    df_imputed[col_catg] = df_imputed[col_catg].round().astype(int)
    df_imputed[col_catg] = encoder.inverse_transform(df_imputed[col_catg])

    df = df_imputed
    return df





# _________________________________________________________________
def save_cleaned_data(df):
    df.to_csv('../data/cleaned_ds.csv', index=False)
    return 'data is saved!'