set -e

KF_PATH="$HOME/.kubeflow"
rm -fr $KF_PATH
mkdir -p $KF_PATH
cd $KF_PATH
wget https://github.com/kubeflow/kfctl/releases/download/v1.0/kfctl_v1.0-0-g94c35cf_linux.tar.gz -O kfctl_linux.tar.gz	

tar -xvf kfctl_linux.tar.gz	

export PATH=$PATH:$KF_PATH
export KF_NAME=my-kubeflow
export BASE_DIR=$KF_PATH
export KF_DIR=${BASE_DIR}/${KF_NAME}
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_k8s_istio.v1.0.2.yaml" 

mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}