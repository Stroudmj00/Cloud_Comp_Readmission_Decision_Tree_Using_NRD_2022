# Cloud Readmission Prediction with NRD 2022

Author: Michael Stroud

## Summary

This project uses the 2022 Nationwide Readmissions Database (NRD) to build a cloud-hosted baseline model for 30-day hospital readmission prediction. The main value is not a high-performing final model; it is the reproducible cloud pipeline, memory-constrained feature engineering, and honest evaluation of what a simple decision tree can and cannot explain.

The work was run on the Nautilus cloud computing platform with Kubernetes resources, Dockerized dependencies, and a Python ML pipeline split into preprocessing, modeling, evaluation, and reporting modules.

## What This Demonstrates

- Moving a healthcare ML workload from local memory limits to a cloud/Kubernetes environment.
- Structuring a notebook-style model into reusable Python modules.
- Handling a large NRD-derived sample with one-hot encoded features and saved intermediate artifacts.
- Reporting a low baseline result without overstating model quality.
- Preparing a path for stronger follow-on models such as random forests or gradient-boosted methods.

## Data and Modeling Context

Hospital readmissions are operationally and financially important because excessive 30-day readmissions can affect patient welfare, capacity planning, and reimbursement incentives.

This project uses a one-million-encounter sample of patients aged 65 or older from NRD 2022. The pipeline trains and tunes a decision tree model to predict 30-day readmission risk and evaluates the model using ROC-AUC.

## Result

- **Model:** decision tree baseline with hyperparameter tuning
- **Test ROC-AUC:** 0.598
- **Most important predictors:** mortality risk grouping and length-of-stay grouping
- **Interpretation:** useful as a cloud-computing and baseline-modeling proof point, not as a production readmission-risk model

## Running the Model

Create the persistent volume claim:

```bash
kubectl apply -f kubernetes/pvc.yaml
```

Create the pod:

```bash
kubectl apply -f kubernetes/pod.yaml
```

Watch pod status:

```bash
kubectl get pods -w
```

Upload data to persistent storage:

```bash
kubectl cp <your-data> <your-pod>:/data/repo/data/
```

Open a shell in the pod:

```bash
kubectl exec -it pod-mjsrkq-train -- /bin/bash
cd /data/repo
pip install -r requirements.txt
python3 src/main.py
```

Copy results back locally:

```bash
kubectl cp pod-mjsrkq-train:/data/repo/results/ ./results/
```

## Performance Notes

A local NLP container hit an out-of-memory failure. A larger capstone container completed the run in roughly four minutes. The Nautilus cloud run completed in roughly two minutes, with runtime affected by writing processed data to disk between pipeline stages.

## Repository Structure

```text
kubernetes/
  pvc.yaml
  pod.yaml
src/
  preprocessing.py
  model.py
  evaluate.py
  main.py
results/
  metrics.json
  feature importance.png
docs/
  SETUP.md
  CLOUD_SETUP.md
Dockerfile
requirements.txt
README.md
```

## Generative AI Use

Generative AI helped port notebook code into modular Python files, improve PEP 8 compliance, use relative paths, and understand Docker configuration. The original notebook logic and project framing were written by Michael Stroud.

## Limitations

- ROC-AUC is low and should be treated as a baseline only.
- NRD-derived data handling requires care around publication and reproducibility.
- Decision trees are interpretable but limited for this task.
- The next modeling step should compare stronger ensemble methods under the same cloud pipeline.