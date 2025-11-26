#evaluate
### Generative AI Disclaimer see README

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
import json
from pathlib import Path

def run_evaluation():
    # Define paths
    BASE_DIR = Path(__file__).parent.parent
    PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
    RESULTS_DIR = BASE_DIR / 'results'
    FIGURES_DIR = RESULTS_DIR / 'figures'
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and test data...")
    best_model = joblib.load(RESULTS_DIR / 'best_model.pkl')
    X_test = joblib.load(PROCESSED_DIR / 'X_test.pkl')
    y_test = joblib.load(PROCESSED_DIR / 'y_test.pkl')
    
    feature_names = X_test.columns

    # Predictions
    proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test ROC-AUC: {auc:.3f}")

    # Save Metrics to JSON
    metrics = {'roc_auc': auc}
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Feature Importance Plot
    imp = best_model.feature_importances_
    order = np.argsort(imp)[::-1][:20]

    plt.figure(figsize=(8,6), dpi=140)
    plt.barh(range(len(order)), imp[order][::-1])
    plt.yticks(range(len(order)), feature_names[order][::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    
    # Save Plot
    plt.savefig(FIGURES_DIR / 'feature_importance.png')
    print(f"Plot saved to {FIGURES_DIR / 'feature_importance.png'}")

if __name__ == "__main__":
    run_evaluation()