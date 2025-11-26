# Cloud Setup
## Background
Documentation video:
https://www.youtube.com/watch?v=LKneRTXvVd8

By carefully examining our yaml files we can see how we configured our interaction with nautilus. 

We utilize this url to access the nautilus hypercluster
https://gp-engine.nrp-nautilus.io/

## PVC
I am selecting the k8 container. 
My persistent storage claim is named "mjsrkq-pv"
My storage class is "rook-cephfs"
I requested 50 GB

## Compute / pods
We are using one pod with 8 cores.
I am using 32 gb of ram because this is the max I could use without submitting a job.
We are not using a gpu. 

## Custom Docker Image
We utilize a custom docker image located here: stroudmj00/readmission-project:v1