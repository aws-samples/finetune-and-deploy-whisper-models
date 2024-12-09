#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Read S3 path from config file
source deploy_config.sh
if [ -z "$S3_PATH" ]; then
    echo "Error: S3_PATH not set in config.sh"
    exit 1
fi

# download model from s3
python3 download_model_from_s3.py --source_s3_url $S3_PATH --local_dir_path "model_repo_whisper_trtllm/" --working_dir "/workspace" || { echo "S3 sync failed"; exit 1; }

model_repo_path=/workspace/model_repo_whisper_trtllm

tritonserver --model-repository $model_repo_path \
    --pinned-memory-pool-byte-size=2048000000 \
    --cuda-memory-pool-byte-size=0:4096000000 \
    --http-port 10086 \
    --metrics-port 10087 > triton.log 2>&1 &

triton_pid=$!

# Wait for tritonserver to start and check if it's running
timeout=30
while [ $timeout -gt 0 ]; do
    if kill -0 $triton_pid 2>/dev/null; then
        echo "Triton server started successfully"
        python3 /workspace/run_server.py -w 32
        # python3 ./run_server.py -w 12
        exit 0
    fi
    sleep 1
    ((timeout--))
done

echo "Error: Triton server failed to start within 30 seconds"
exit 1

# Cleanup function
cleanup() {
    if [ ! -z "$triton_pid" ]; then
        kill $triton_pid
    fi
}

trap cleanup EXIT
