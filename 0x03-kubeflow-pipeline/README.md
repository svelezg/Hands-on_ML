# 0x03 Kubeflow-pipeline


<img src="https://github.com/svelezg/Hands-on_ML/blob/main/9x99-images/kubeflow.jpg" width="350"/>

## Intro
Kubeflow goal is to make deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable. 
The spitit is to to provide a straightforward way to deploy best-of-breed open-source systems for ML to diverse infrastructures. 
Anywhere you are running Kubernetes, you should be able to run Kubeflow.

One of Kubeflow components are Pipelines. It is a platform for building and deploying portable and scalable 
end-to-end ML workflows, based on containers.

From the the Kubeflow Pipelines platform documentation the following goals:

* End-to-end orchestration: enabling and simplifying the orchestration of machine learning pipelines.

* Easy experimentation: making it easy for you to try numerous ideas and techniques and manage your various trials/experiments.

* Easy re-use: enabling you to re-use components and pipelines to quickly cobble together end-to-end solutions, without having to rebuild each time.


## Objective
Carry out a single step pipeline for training within Kubeflow.

## GCP Virtual machine
Kubeflow can run anywhere Kubernetes can run. That holds only if there are enough computational resources. For academic purposes, the suggested choice is to use GCP free credits to spin a Virtual machine big enough to work comfortably.

Suggested is:

e2-standard-8 
*    8 vCPUs 
*    32 GB memory
*    100Gb disk
*    debian-10-buster-v20201014	

After creating and starting the VM follow the next step to get to the training pipeline.

## Installation
Open the VM SSH connection

### Clone the repo
```
git glone
cd 0x03-kubeflow-pipeline
```

### wget installation
```
sudo apt-get install wget
```

### Docker installation
```
sudo apt update
sudo apt install --yes apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update
sudo apt install --yes docker-ce
```

adding user to the docker group
```
sudo usermod -aG docker $USER
logout 
```

Test docker
```
docker run busybox date
```

Auteticate user
```
gcloud auth configure-docker
```

### kubectl installation 
```
sudo apt-get update && sudo apt-get install -y apt-transport-https gnupg2 curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl=1.14.10-00
```

### minikube installation
```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
```


### kubernetes installation 
```
minikube start \
    --cpus 5 \
    --memory 10288 \
    --disk-size 20gb \
    --kubernetes-version 1.14.10
```
To restart minikube after installation run
```
minikube start
```


### kubernetes dashboard (optional)
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0/aio/deploy/recommended.yaml
```
**Get token**
```
kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | awk '/^deployment-controller-token-/{print $1}') | awk '$1=="token:"{print $2}'
```
**start proxy**
```bash
kubectl proxy -p 8010
```
**open dashboard**
[http://localhost:8010/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy](http://localhost:8010/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy)



### kubeflow installation
Run provided script

```
./install/kf-1.0.sh
```

Check pods deployment status
```
kubectl get pod -n kubeflow
```

Forward port to open dashboard
```
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```


### Kubeflow Pipelines SDK installation
```
pip3 install kfp --upgrade
```

## local test python
python main.py --epochs 2


## Deployment and run
We will deploy and later build the training pipeline

### pipeline compilation
Choose and compile a pipeline
```
cd pipeline
dsl-compile --py pipeline.py --output pipeline.tar.gz
kfp pipeline upload-version pipeline.yml -p c554c4c8-b672-4c0f-90d6-24e7132ee06c -v test-pipeline-$(date +%s)
```


# Build and push docker image
Build image (do not forget the point at the end)
```
sudo docker build -t pipeline .
```

Test locally
 ```
sudo docker run -i --name pipeline -p 8000:8000 pipeline
```

Open the container console
```
docker exec -it pipeline bash
```
Test with python
```
python training/main.py --epochs 20
```
Test with bash scrip
```
./training/main.sh
```

Tag the docker image. You can use either DockerHub or Google Container Registry (gcr). Remember to Enable the Container Registry API
```
docker tag pipeline gcr.io/{PROJECT NAME AND ID]/pipeline
```
Puch the image (may take some minutes)
```
docker push gcr.io/{PROJECT NAME AND ID}/pipeline
```
Look for the pushed image in the container registry and make it public.

# Upload and run the pipeline
On the left panel open pipeline. Give it a name and choose upload from file.
Select the tar file

Review the yaml file that kubeflow automatically creates.


##
Make a new experiment and within it a new run.
