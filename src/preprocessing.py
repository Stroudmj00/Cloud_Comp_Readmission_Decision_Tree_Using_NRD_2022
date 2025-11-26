#preprocessing
### Generative AI Disclaimer see README

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import joblib
from pathlib import Path

def run_preprocessing():
    # Define paths relative to this script
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    PROCESSED_DIR = DATA_DIR / 'processed'
    
    print(f"Directory check:\n Base: {BASE_DIR}\n Data: {DATA_DIR}")
    
    # Create directory for artifacts
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    # Assumes parquet file is in the data folder
    input_file = DATA_DIR / 'nrd_preprocessed_updated.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f"Cannot find input file at {input_file}")
        
    df = pd.read_parquet(input_file)
    
    target, id_col = 'readmitted_30_days', 'nrd_visitlink'
    df[id_col] = df[id_col].astype(str)
    df[target] = df[target].astype(int)

    X = df.drop(columns=[target, id_col])
    y = df[target].to_numpy()
    groups = df[id_col].to_numpy()

    print(f"Positive rate: {y.mean().round(3)}")

    # Group-aware Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test  = y[test_idx]
    groups_train = groups[train_idx]

    # Downcast numerics
    num_cols = X_train.select_dtypes(include=['float64','float32','int64','int32']).columns
    X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, downcast='float')
    X_test[num_cols]  = X_test[num_cols].apply(pd.to_numeric, downcast='float')
    
    y_train = y_train.astype('int8')
    y_test  = y_test.astype('int8')

    # One-Hot Encoding
    cols_ohe = ['mdc','hcup_ed','i10_serviceline','pay1','pclass_orproc',
                'resident','hosp_ur_teach','h_contrl','los_group','aprdrg']

    print("Applying One-Hot Encoding...")
    X_train = pd.get_dummies(X_train, columns=cols_ohe, drop_first=False, dtype='uint8')
    X_test  = pd.get_dummies(X_test,  columns=cols_ohe, drop_first=False, dtype='uint8')

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Save Artifacts
    print(f"Saving processed data to {PROCESSED_DIR}...")
    joblib.dump(X_train, PROCESSED_DIR / 'X_train.pkl')
    joblib.dump(X_test, PROCESSED_DIR / 'X_test.pkl')
    joblib.dump(y_train, PROCESSED_DIR / 'y_train.pkl')
    joblib.dump(y_test, PROCESSED_DIR / 'y_test.pkl')
    joblib.dump(groups_train, PROCESSED_DIR / 'groups_train.pkl')
    print("Preprocessing complete.")

if __name__ == "__main__":
    run_preprocessing()