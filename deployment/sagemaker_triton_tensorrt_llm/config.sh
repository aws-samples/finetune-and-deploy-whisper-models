#!/bin/bash

ACCOUNT="Your AWS Account Id"
REGION="Your SageMaker Region"
# 项目配置
PROJECT_ROOT="/home/ec2-user/SageMaker/finetune-and-deploy-whisper-models"

#注意 s3 路径最后加上 /
# Defualt to your sagemaker defualt bucket, you could modify this path to the bucket you created, e.g. s3://Your-S3-bucket/test_whisper_turbo/
S3_PATH="s3://sagemaker-${REGION}-${ACCOUNT}/whisper-triton-models/"


# Docker 配置
## 注意：如果修了docker image 的值，也需要同步修改 deploy_and_test_preprocessed.ipynb 文件里的镜像地址
## 默认的 image url 会是: ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${DOCKER_IMAGE}
DOCKER_IMAGE="sagemaker-endpoint/whisper-triton-byoc:latest"

# 模型配置
# MODEL_NAME="base"
# MODEL_NAME="tiny"
MODEL_NAME="large-v3-turbo"
# MODEL_NAME="large-v3"
N_MELS=128 # 128 dim for large-v3/large-v3-turbo, 80 dim for large-v2/base/tiny...

HUGGING_FACE_MODEL_ID="openai/whisper-large-v3-turbo"
# HUGGING_FACE_MODEL_ID="openai/whisper-large-v3"
# HUGGING_FACE_MODEL_ID="openai/whisper-tiny"
# HUGGING_FACE_MODEL_ID="openai/whisper-base"

# OUTPUT_MODEL_PATH="$PROJECT_ROOT/sagemaker_triton/assets/base.pt"
# OUTPUT_MODEL_PATH="$PROJECT_ROOT/sagemaker_triton/assets/tiny.pt"
OUTPUT_MODEL_PATH="$PROJECT_ROOT/sagemaker_triton/assets/large-v3-turbo.pt"
# OUTPUT_MODEL_PATH="$PROJECT_ROOT/sagemaker_triton/assets/large-v3.pt"

# OPENAI_WHISPER_DOWNLOAD_URL="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
# OPENAI_WHISPER_DOWNLOAD_URL="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
# OPENAI_WHISPER_DOWNLOAD_URL="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
OPENAI_WHISPER_DOWNLOAD_URL="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"

# LoRA 配置
USE_LORA=false  # 设置为 true 如果需要合并 LoRA 模型
LORA_PATH="/home/ec2-user/SageMaker/lora_model/checkpoint-135"  # 仅在 USE_LORA 为 true 时使用，修改为实际的 whisper lora checkpoint 路径

# 其他配置
CONDA_ENV="pytorch_p310"
