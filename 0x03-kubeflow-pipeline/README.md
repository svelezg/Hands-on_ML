# 0x03 Kubeflow-pipeline

## Intro


## GCP Virtual machine
e2-standard-8 
    8 vCPUs 
    32 GB memory
    100Gb disk
    debian-10-buster-v20201014	


## Clone the repo

```
git glone
cd 0x03-kubeflow-pipeline
```

# wget installation
```
sudo apt-get install wget
```

## Docker installation
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

# minikube installation
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


## Kubeflow Pipelines SDK installation
```
pip3 install kfp --upgrade
```

## local test python
python main.py --epochs 2


### pipeline compilation
Choose and compile a pipeline
```
cd pipeline
dsl-compile --py pipeline.py --output pipeline.tar.gz
kfp pipeline upload-version pipeline.yml -p c554c4c8-b672-4c0f-90d6-24e7132ee06c -v test-pipeline-$(date +%s)
```


# build and push image
Build image
sudo docker build -t pipeline .

Test locally 
sudo docker run -i --name pipeline -p 8000:8000 pipeline


docker exec -it pipeline bash
Test with python
python training/main.py --epochs 2
Test with bash scrip
./training/main.sh


docker tag pipeline gcr.io/{PROJECT NAME AND ID]/pipeline

Enable the Container Registry API
docker push gcr.io/otherproject-294618/cubo
Make image public