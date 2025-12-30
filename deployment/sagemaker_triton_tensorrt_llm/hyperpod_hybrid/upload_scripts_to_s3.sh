#!/bin/bash

set -e

# Load configuration
if [ -f "config.env" ]; then
    source config.env
    echo "✅ Configuration loaded from config.env"
else
    echo "❌ config.env not found. Please create it first."
    exit 1
fi

echo "=== Uploading Scripts to S3 Bucket ==="

S3_BUCKET="${S3_BUCKET_NAME}/${S3_SCRIPTS_PATH}"  # triton-models-xq/deployment_codes
SCRIPTS_DIR="../sagemaker_triton/model_data"

echo "Uploading scripts to s3://$S3_BUCKET/"
echo "Source directory: $SCRIPTS_DIR"
echo ""

# Upload individual script files
echo "Uploading run_server.py..."
aws s3 cp "$SCRIPTS_DIR/run_server.py" "s3://$S3_BUCKET/"

echo "Uploading whisper_api.py..."
aws s3 cp "$SCRIPTS_DIR/whisper_api.py" "s3://$S3_BUCKET/"

echo "Uploading deploy_config.sh..."
aws s3 cp "$SCRIPTS_DIR/deploy_config.sh" "s3://$S3_BUCKET/"

# Optional: Upload all files
# echo "Uploading all script files..."
# aws s3 cp "$SCRIPTS_DIR/" "s3://$S3_BUCKET/" --recursive --exclude="*.pyc" --exclude="__pycache__/*"

# Verify upload
echo ""
echo "Verifying uploaded files:"
aws s3 ls "s3://$S3_BUCKET/" --recursive | grep -E "\.(py|sh)$"

echo ""
echo "✅ Scripts successfully uploaded to S3!"
echo ""
echo "S3 bucket structure should now be:"
echo "  s3://${S3_BUCKET_NAME}/"  # triton-models-xq
echo "  ├── ${S3_G6E_MODEL_PATH}/           # G6E model files"  # test_turbo_g6e
echo "  ├── ${S3_G5_MODEL_PATH}/               # G5 model files"  # test_turbo
echo "  └── ${S3_SCRIPTS_PATH}/         # Deployment scripts"  # deployment_codes
echo "      ├── run_server.py         # API server script"
echo "      ├── whisper_api.py        # Whisper API implementation"
echo "      └── deploy_config.sh      # Config script"
echo ""
echo "You can now run: ./deploy_pv_scripts.sh"
