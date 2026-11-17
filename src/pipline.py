from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import NUMERICAL_COLS, ONE_HOT_COLS, ORDINAL_COLS, DROP_COLS
from sklearn.linear_model import LogisticRegression



# _________________________________________________________________
def build_preprocessor():
    num_cols = [c for c in NUMERICAL_COLS if c not in DROP_COLS]

    num_transformer = Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=42)),
        ('scaler', StandardScaler())
    ])

    one_hot_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    Physical_Activity_catg = ['Low', 'Moderate', 'High']
    Diet_Quality_catg = ['Poor', 'Average', 'Good']
    Part_Time_Job_catg = ['No', 'Yes']
    Previous_Mental_Health_History_catg = ['No', 'Yes']
    
    categories = [
        Physical_Activity_catg, 
        Diet_Quality_catg, 
        Part_Time_Job_catg, 
        Previous_Mental_Health_History_catg
        ]
    ordinal_cols = list(ORDINAL_COLS.keys())

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OrdinalEncoder(
            categories=categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])


    preprocessor  = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('one-hot', one_hot_transformer, ONE_HOT_COLS),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    return preprocessor 



def create_full_pipline(model=None)->Pipeline:
    if model == None:
        model = LogisticRegression(random_state=42, max_iter=1000)

    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline