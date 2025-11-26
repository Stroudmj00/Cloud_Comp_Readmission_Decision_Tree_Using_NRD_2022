#model
### Generative AI Disclaimer see README

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import joblib
from pathlib import Path

def run_training():
    # Define paths
    BASE_DIR = Path(__file__).parent.parent
    PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
    RESULTS_DIR = BASE_DIR / 'results'
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data for training...")
    try:
        X_train = joblib.load(PROCESSED_DIR / 'X_train.pkl')
        y_train = joblib.load(PROCESSED_DIR / 'y_train.pkl')
        groups_train = joblib.load(PROCESSED_DIR / 'groups_train.pkl')
    except FileNotFoundError:
        print("Error: Processed data not found. Run preprocessing.py first.")
        return

    # Hyperparameters
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': randint(3, 8),
        'min_samples_leaf': randint(50, 201),
        'min_samples_split': randint(50, 201),
        'max_features': uniform(0.3, 0.4),
        'class_weight': [None, 'balanced']
    }

    # Search Configuration
    rs = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc', 
        cv=GroupKFold(n_splits=3), 
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Starting RandomizedSearchCV...")
    rs.fit(X_train, y_train, groups=groups_train)

    print(f"Best Score (ROC-AUC): {rs.best_score_:.3f}")
    
    # Save Best Model
    model_path = RESULTS_DIR / 'best_model.pkl'
    joblib.dump(rs.best_estimator_, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    run_training()