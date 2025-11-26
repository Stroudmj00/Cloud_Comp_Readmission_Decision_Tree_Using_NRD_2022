# Project Setup Guide

## Before we begin you need...
(1) Nautilus
(2) kubernetes
(3) Python
(4) Git

## Strucutre of repo
src folder contains python scripts, source code
kubernetes folder contains yaml for pod and pvc deployment 
data folder contains the storage for datasets

## Run these commands to access the cloud pod
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