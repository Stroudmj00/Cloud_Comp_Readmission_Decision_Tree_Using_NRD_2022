# Cloud Computing: Predicting Readmission using NRD dataset and Decision Tree Algorithm
Author: Michael Stroud

## Description
Hospital readmissions present a significant challenge to the global healthcare system, impacting both patient welfare and operational efficiency. In the United States, incentive structures such as the Hospital Readmissions Reduction Program (HRRP) financially penalize facilities with excessive 30-day readmission rates. A critical need is seen for predictive analytics to identify high-risk patients. This project leverages the Nautilus cloud computing platform to deploy a scalable machine learning pipeline capable of predicting 30-day readmissions in a one million encounter sample of patients aged 65 or greater using the 2022 Nationwide Readmissions Database (NRD). The pipeline implements a decision tree and hyperparameter tuning which maximize the Receiver Operating Characteristic Area Under the Curve (ROC-AUC).

Previous local attempts to analyze this dataset faced memory constraints due to many one-hot encoding variables and an impressive dataset size of 16.5 million encounters. Utilizing the cloud infrastructure allows these models to reach their full potential. 

The final model achieved a Test ROC-AUC of 0.598. Feature importance analysis revealed that risk of mortality (aprdrg_risk_mortality) and length of stay (los_group) were the most salient predictors of readmission. The current model establishes a low baseline, but more importantly proves that a successful use of cloud computing can overcome local resource bottlenecks. This paper paves the way for a random forest model to be used in my 2026 case study. 


## Running the Model
(1) Create PVC
kubectl apply -f kubernetes/pvc.yaml

(2) Create Pod
kubectl apply -f kubernetes/pod.yaml

(3) Ensure pods are running
kubectl get pods -w


(4) Upload your data to your persistant storage
kubectl cp {your data} {your pod}:/data/repo/data/

(5) Log into the pod
kubectl exec -it pod-mjsrkq-train -- /bin/bash

(6) inside the pod change directory to where the data is
cd /data/repo

(7) install requirements.txt
pip install -r requirements.txt

(8) Run main.py 
python3 src/main.py

(9) return results to your local machine. 
kubectl cp pod-mjsrkq-train:/data/repo/results/ ./results/

## Performance Comparison
Utilizing the NLP container this code does not run do to a OOM error. Utilizing the Capstone container, this code takes ~4 minutes to run. Utilizing cloud computing the code took 2 minutes to run. Part of why it took so long is that this code needs to write the processed data and save it to a new file to communicate between steps.

## Directory Structure:
data/

|_ README.md 

|_ Dockerfile 

|_ requirements.txt 

|_ kubernetes/

| |_ pvc.yaml 

| |_ pod.yaml 

|_ src/

| |_ preprocessing.py 

| |_ model.py

| |_ evaluate.py 

| |_ main.py 

|_ data/

| |_ README.md 

| |_ nrd_preprocessed_updated.parquet

|_ results/

| |_ metrics.json 

| |_ feature importance.png
|_ docs/

|_ SETUP.md

|_ CLOUD_SETUP.md


## Generative AI Disclaimer
Generative AI was used in the project to port by jupiter notebook python code (which was written by myself) into python code which is split into the for files as following:
    (1) preprocessing.py
    (2) model.py
    (3) evaluate.py
    (4) main.py
The AI's was instructed to (1) improve on my code, (2) become PEP8 compliant, and (3) use relative file paths. 

Generative AI was also used to understand and configure Docker.
