#!/bin/bash

set -e

# 加载配置
CONFIG_FILE=${1:-"./config.sh"}
echo $CONFIG_FILE
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

source "$CONFIG_FILE"

# 函数：检查上一个命令的执行状态
check_status() {
    if [ $? -ne 0 ]; then
        echo "错误：$* 失败"
        exit 1
    else
        echo "$* 成功完成"
    fi
}

echo "当前的项目路径" $PROJECT_ROOT
# 步骤1: 构建并推送Docker镜像
echo "开始构建并推送Docker镜像..."
cd "$PROJECT_ROOT/sagemaker_triton" && ./build_and_push.sh "$DOCKER_IMAGE"
check_status "Docker镜像构建和推送"

# 步骤2: 准备模型
echo "开始准备模型..."
source activate $CONDA_ENV || { echo "错误：无法激活 $CONDA_ENV 环境"; exit 1; }
# pip install openai-whisper peft transformers
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
pip install --upgrade pip
pip install openai-whisper peft transformers 
check_status "依赖项安装"

mkdir -p "$PROJECT_ROOT/sagemaker_triton/assets"

if [ "${USE_LORA,,}" = "true" ]; then
    echo "合并 LoRA 模型..."
    python merge_lora.py --model-id "$HUGGING_FACE_MODEL_ID" --lora-path "$LORA_PATH" --export-to "$OUTPUT_MODEL_PATH" --openai_model_name "$MODEL_NAME"
    check_status "LoRA 模型合并"
else
    if [ -f "$OUTPUT_MODEL_PATH" ]; then
        echo "模型文件 $OUTPUT_MODEL_PATH 已存在，删除原文件"
        rm -rf $OUTPUT_MODEL_PATH 
    fi
    echo "下载原始 Whisper 模型..."
    wget --directory-prefix=assets $OPENAI_WHISPER_DOWNLOAD_URL
    
fi

# 步骤3: 编译模型
echo "开始编译模型..."
echo $PROJECT_ROOT
echo $DOCKER_IMAGE
docker run --rm -it --net host --shm-size=2g --gpus all \
  -v "$PROJECT_ROOT/sagemaker_triton/:/workspace/" \
  $DOCKER_IMAGE bash -c "cd /workspace && bash export_model.sh $MODEL_NAME"
check_status "模型编译"

# 步骤4: 上传编译后的模型到S3
# 检查是否提供了参数
if [ -z "${N_MELS}" ]; then
    echo "请提供 n_mels 的新值"
    exit 1
fi
wget -nc --directory-prefix=$PROJECT_ROOT/sagemaker_triton/model_repo_whisper_trtllm/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
# 使用 sed 命令修改 config.pbtxt 文件
sed -i '/key: "n_mels"/,/string_value:/ s/string_value:"[0-9]*"/string_value:"'${N_MELS}'"/' "$PROJECT_ROOT/sagemaker_triton/model_repo_whisper_trtllm/whisper/config.pbtxt"
head -n20 $PROJECT_ROOT/sagemaker_triton/model_repo_whisper_trtllm/whisper/config.pbtxt
echo "n_mels 的值已更新为 $N_MELS"

echo "开始上传模型到S3..."
aws s3 sync "$PROJECT_ROOT/sagemaker_triton/model_repo_whisper_trtllm/" "$S3_PATH" --exclude "*/__pycache__/*"
check_status "模型上传到S3"

# 步骤5: 修改模型部署脚本
echo "修改模型部署脚本..."
echo "模型部署s3 path：" $S3_PATH
# sed -i "s|s3://<Your S3 path>|$S3_PATH|g" "$PROJECT_ROOT/sagemaker_triton/model_data/start_triton_and_client.sh"
sed -i "s|S3_PATH=.*|S3_PATH=\"$S3_PATH\"|" $PROJECT_ROOT/sagemaker_triton/model_data/deploy_config.sh
cat $PROJECT_ROOT/sagemaker_triton/model_data/deploy_config.sh
check_status "部署脚本修改"

echo "所有准备工作已完成。请继续在Jupyter notebook中执行部署步骤。"
