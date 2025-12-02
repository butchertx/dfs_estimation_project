import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def save_model(model, filepath: str):
    """Saves the model to a file using pickle."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """Loads the model from a file using pickle."""
    import pickle
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

class DataFrameLookupModel:
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        if any(col not in df.columns for col in feature_cols + [target_col]):
            raise ValueError("DataFrame does not contain all specified feature or target columns.")
        
        self.df.set_index(feature_cols, inplace=True)

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        X = pd.DataFrame(X, columns=self.feature_cols)
        return self.df.loc[X.set_index(self.feature_cols).index, self.target_col].values


def get_preprocessor():
    cat_features = [
        'roster_slot_id', 'contest_type', 'position', 'double_up',
        'games_count', 'multientry'
    ]

    cat_pipeline = Pipeline([
        ('one_hot', OneHotEncoder(categories='auto', sparse_output=False, handle_unknown='ignore')),
    ])
    
    num_features = [
        'entry_count', 'payout', 'entries_max', 'entries_fee', 'max_entry_fee',
        'salary', 'projection',
        'projection_value_ratio', 'salary_mean', 'salary_max', 'salary_std',
        'projection_mean', 'projection_max', 'projection_std',
        'projection_value_ratio_mean', 'projection_value_ratio_max',
        'projection_value_ratio_std', 'salary_vor', 'projection_vor',
        'projection_value_ratio_vor', 'salary_vum', 'projection_vum',
        'projection_value_ratio_vum'
    ]

    num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])
    
    basic_preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_features),
        ('num', num_pipeline, num_features)
    ])
    
    return basic_preprocessor

PREPROC = get_preprocessor()
MODELS = {
    # 'DataFrameLookupModel': DataFrameLookupModel,
    'LinearRegression': Pipeline([
                            ('preprocessor', PREPROC),
                            ('lin_reg', LinearRegression())
                        ]),
    'Lasso': Pipeline([
                ('preprocessor', PREPROC),
                ('lasso', Lasso(alpha=0.01))
            ]),
    'Ridge': Pipeline([
                ('preprocessor', PREPROC),
                ('ridge', Ridge())
            ]),
    'RandomForest': Pipeline([
                            ('preprocessor', PREPROC),
                            ('forest', RandomForestRegressor(
                                min_samples_leaf=20,
                                min_samples_split=50,
                                n_jobs=-1
                            ))
                        ]),
    'SKLGradientBoosting': Pipeline([
                            ('preprocessor', PREPROC),
                            ('hist_gradient_boosting', HistGradientBoostingRegressor())
                        ]),
    'XGBoost': Pipeline([
                            ('preprocessor', PREPROC),
                            ('xgboost', XGBRegressor())
                        ]),
    'LightGBM': Pipeline([
                            ('preprocessor', PREPROC),
                            ('lightgbm', LGBMRegressor(
                                force_col_wise=True
                                ))
                        ]),
}