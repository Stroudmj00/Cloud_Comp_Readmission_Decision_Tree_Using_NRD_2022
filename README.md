# Cloud Computing: Predicting Readmission using NRD dataset and Decision Tree Algorithm
Author: Michael Stroud

## Description
This project focuses on cloud-based machine learning models with a goal to predict 30-day hospital readmissions using the Nationwide Readmissions Database (NRD). I am using a 1 million row sample for this project, but next semester we will create a job to run all 16.5 million encounters. Docker containerized this pipeline. The model is deployed on the Nautilus Kubernetes cluster. Our model utilizes a decision tree classifier with hyperparameter tuning. This model outputs the best performing decision tree, a feature importance graph, and a json file containing the most important metric, ROC-AUC.

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
