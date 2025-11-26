#Main
### Generative AI Disclaimer see README

import preprocessing
import model
import evaluate
import time

def main():
    print("=== Starting Readmission Prediction Pipeline ===")
    start_time = time.time()

    print("\n[Step 1] Preprocessing")
    preprocessing.run_preprocessing()

    print("\n[Step 2] Training Model")
    model.run_training()

    print("\n[Step 3] Evaluation")
    evaluate.run_evaluation()
    
    elapsed = time.time() - start_time
    print(f"\n=== Pipeline Complete in {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()