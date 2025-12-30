# Whisper Triton HyperPod EKS Deployment

This directory contains the successful deployment configuration for Whisper Triton on Amazon EKS HyperPod.

## Prerequisites

Before deployment, ensure the following components are installed and configured:

### 0. Image Build and Model Compilation (Required First)

**Important**: Before deploying to HyperPod EKS, you must first build the Docker image and compile the model. Please follow the instructions in the parent directory:

📖 **Reference**: [../README.md](../README.md)

Key steps:
1. Configure `../config.sh` with your parameters
2. Run `../prepare_and_deploy.sh` to:
   - Build and push Docker image to ECR
   - Compile model with TensorRT-LLM
   - Upload compiled model to S3

⚠️ **Note**: Models compiled with TensorRT-LLM can only be deployed on the same instance type used for compilation. For example:
- Model compiled on G5 instance → Deploy on G5 nodes
- Model compiled on G6E instance → Deploy on G6E nodes

### 1. AWS Load Balancer Controller
Check if AWS Load Balancer Controller is installed:
```bash
kubectl get deployment aws-load-balancer-controller -n kube-system
```

If not installed, run:
```bash
./create_lb_controller.sh -v <VPC_ID> -c <CLUSTER_NAME>
```

### 2. S3 CSI Driver
Check if S3 CSI Driver is installed:
```bash
kubectl get daemonset s3-csi-node -n kube-system
```

If not installed, run:
```bash
./create_s3_csi_driver.sh -c <CLUSTER_NAME> -r <REGION>
```

## Configuration

**IMPORTANT**: Before deployment, modify the `config.env` file with your specific values:

```bash
# AWS Configuration
AWS_REGION=us-east-1                    # Your AWS region
AWS_ACCOUNT_ID=596899493901             # Your AWS account ID

# S3 Configuration  
S3_BUCKET_NAME=triton-models-xq         # Your S3 bucket name
S3_SCRIPTS_PATH=deployment_codes        # Path for deployment scripts
S3_G6E_MODEL_PATH=test_turbo_g6e        # Path for G6E model files
S3_G5_MODEL_PATH=test_turbo             # Path for G5 model files

# ECR Configuration
ECR_REPOSITORY_NAME=sagemaker-endpoint/whisper-triton-byoc-g6e  # Your ECR repository
ECR_IMAGE_TAG=latest                    # Your image tag

# Instance Types
G6E_INSTANCE_TYPE=ml.g6e.2xlarge       # G6E instance type
G5_INSTANCE_TYPE=ml.g5.2xlarge         # G5 instance type

# Kubernetes Configuration
NAMESPACE=default                       # Kubernetes namespace
PV_NAME=pv-triton-models               # PersistentVolume name
PVC_NAME=triton-models                 # PersistentVolumeClaim name
STORAGE_SIZE=1200Gi                    # Storage size

# Service Configuration
SERVICE_NAME=whisper-triton-unified-nlb # Service name
TRITON_PORT=10086                      # Triton server port
API_PORT=8080                          # API server port
METRICS_PORT=10087                     # Metrics port
```

## Model Compilation and Upload
1. Compile models - different instance types require compilation on corresponding GPU machines
2. Upload models to S3
3. Bind the S3 path as PV in the cluster

## S3 Bucket PV/PVC Binding

### Create PersistentVolume and PersistentVolumeClaim for S3 bucket:

```bash
# Apply PV configuration
kubectl apply -f pv-triton-models.yaml

# Apply PVC configuration  
kubectl apply -f pvc-triton-models.yaml

# Verify binding
kubectl get pv,pvc
```

The configurations use AWS S3 CSI driver to mount `triton-models-xq` S3 bucket:
- **PV**: `pv-triton-models` (1200Gi, ReadWriteMany)
- **PVC**: `triton-models` (bound to PV)
- **Mount options**: `allow-delete`, `region us-east-1`

## Deployment

### PV Scripts introduction
- `whisper-triton-pv-scripts.yaml` - Scripts stored directly in S3/PV
- `upload_scripts_to_s3.sh` - Upload scripts to S3 bucket
- `deploy_pv_scripts.sh` - Deploy with PV-based scripts


### Steps:
```bash
# 1. Configure environment variables
cp config.env.example config.env  # Create from template if needeS
# Edit config.env with your values

# 2. Upload scripts to S3
source config.env && ./upload_scripts_to_s3.sh

# 3. Deploy
source config.env && ./deploy_pv_scripts.sh
```

## Configuration

- **G6E Instance**: `ml.g6e.2xlarge` with model path `test_turbo_g6e`
- **G5 Instance**: `ml.g5.2xlarge` with model path `test_turbo`
- **S3 Bucket**: `triton-models-xq` containing models and deployment scripts
- **Ports**: 
  - Triton Server: 10086
  - API Server: 8080

## S3 Bucket Structure (PV Scripts)

```
s3://triton-models-xq/
├── test_turbo_g6e/           # G6E model files
├── test_turbo/               # G5 model files
└── deployment_codes/         # Deployment scripts
    ├── run_server.py         # API server script
    └── whisper_api.py        # Whisper API implementation
```

## Service Endpoints

### Unified LoadBalancer (Single Entry Point)
- **Unified NLB**: `whisper-triton-unified-nlb`
  - Current: `k8s-default-whispert-72e1749839-7dc3f5766c221754.elb.us-east-2.amazonaws.com:8080`
  - IP: `18.219.99.188:8080`

## Testing

```bash
# Test unified endpoint
python3 test_unified_lb.py

# Manual test
curl http://18.219.99.188:8080/ping

# real audio test
curl -X POST http://k8s-default-whispert-eb4eb229b7-03296d4e2539c9bf.elb.us-east-2.amazonaws.com:8080/invocations -H "Content-Type: application/json" -d "{\"audio_data\": \"$(base64 -w 0 test.wav)\", \"whisper_prompt\": \"\"}" | jq .

# Python API test
python3 -c "
import requests
import base64

with open('audio.wav', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

url = 'http://k8s-default-whispert-eb4eb229b7-03296d4e2539c9bf.elb.us-east-2.amazonaws.com:8080/invocations'
payload = {'audio_data': audio_b64, 'whisper_prompt': ''}

response = requests.post(url, json=payload)
result = response.json()

print(f\"Status: {response.status_code}\")
print(f\"Result: {result}\")
```

## Advantages of PV Scripts

1. **No ConfigMap needed** - Scripts stored directly with models
2. **Simpler deployment** - Fewer Kubernetes resources
3. **Easier updates** - Just upload to S3, restart pods
4. **Version control** - S3 versioning for scripts
5. **Shared storage** - Scripts available to all pods

✅ **Status**: All deployment options fully operational!

## Configuration Files

All deployment files now use environment variables for easy customization:

- **config.env**: Main configuration file containing all deployment variables
- **pv-triton-models.yaml**: PersistentVolume configuration with variables `${S3_BUCKET_NAME}`, `${AWS_REGION}`, etc.
- **pvc-triton-models.yaml**: PersistentVolumeClaim configuration with variables `${PVC_NAME}`, `${NAMESPACE}`, etc.
- **whisper-triton-pv-scripts.yaml**: Main deployment with variables for images, ports, paths, etc.
- **upload_scripts_to_s3.sh**: Script using `${S3_BUCKET_NAME}` and `${S3_SCRIPTS_PATH}`
- **deploy_pv_scripts.sh**: Deployment script that loads config.env and uses `envsubst` for variable substitution

**Note**: All files contain the original hardcoded values as comments for reference.
