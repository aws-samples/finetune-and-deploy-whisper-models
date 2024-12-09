# Whisper ASR SageMaker Triton Deployment

This project demonstrates the deployment of the Whisper Automatic Speech Recognition (ASR) model on Amazon SageMaker using NVIDIA Triton Inference Server. It provides a scalable and efficient way to perform speech-to-text transcription using the Whisper large-v3 model.

## Overview

The project uses a custom Docker container to deploy the Whisper model on SageMaker, leveraging TensorRT-LLM for model compilation and optimization, with Triton Inference Server for deployment. It includes scripts for model deployment, endpoint creation, and transcription testing.

## Prerequisites

- An AWS account with SageMaker access
- Docker installed on your local machine
- Python 3.10+
- AWS CLI configured with appropriate permissions
- AWS ECR permissions
- NVIDIA GPU support (for local testing and development)

## Model Compilation and Deployment

Note: Models compiled with TensorRT-LLM can only be deployed on the same instance type used for compilation. For example, a model compiled on g5.xlarge can only be deployed on G5 series instances.

1. Clone this repository:
   ```
   git clone https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models.git
   cd finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/
   ```

2. Configure deployment parameters:
   Edit the `config.sh` file to set the following parameters:
   - `PROJECT_ROOT`: Project root directory path
   - `DOCKER_IMAGE`: Docker image name
   - `CONDA_ENV`: Conda environment name
   - `USE_LORA`: Whether to use LoRA model (true/false)
   - `HUGGING_FACE_MODEL_ID`: Hugging Face model ID
   - `LORA_PATH`: LoRA model path (if using)
   - `OUTPUT_MODEL_PATH`: Output model path
   - `OPENAI_WHISPER_DOWNLOAD_URL`: OpenAI Whisper model download URL
   - `S3_PATH`: S3 storage path

3. Run the preparation and deployment script:
   ```
   chmod +x ./prepare_and_deploy.sh && ./prepare_and_deploy.sh
   ```
   This script will automatically:
   - Build and push the Docker image to Amazon ECR
   - Prepare the model (download or merge LoRA model)
   - Compile the model
   - Upload the compiled model to S3
   - Modify model deployment scripts

4. Model Deployment:
   - Open the [deploy_and_test.ipynb](https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/deploy_and_test.ipynb) notebook in SageMaker Studio, Jupyter, or a local machine configured with AWS access
   - Follow the deployment code in the notebook

5. Testing:
   After deployment, you can use the SageMaker endpoint for transcription. The notebook includes example code for transcribing audio files, and you can also refer to [test_whisper_api.py](https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/test_whisper_api.py) for API usage.

## Docker Image

The Docker image is based on the NVIDIA Triton Server image (nvcr.io/nvidia/tritonserver:24.10-py3) and includes the following key components:

- FFmpeg for audio processing
- TensorRT-LLM tensorrt-llm==0.15.0.dev2024110500
- Triton Client libraries
- Custom Python dependencies (see requirements.txt)

## Python Dependencies

Key Python packages required for this project include:

- sagemaker and sagemaker-ssh-helper (optional) for SageMaker integration
- boto3 for AWS SDK
- openai-whisper for the Whisper ASR model
- tritonclient[grpc] for Triton Inference Server client
- fastapi and uvicorn for API serving
- Other audio processing and machine learning libraries (librosa, soundfile, transformers, etc.)

For a complete list, refer to the `requirements.txt` file.

## Key Components

- `ACCOUNT`：AWS Account ID
- `REGION`：AWS Service Region
- `prepare_and_deploy.sh`: Main script for automating preparation and deployment process
- `config.sh`: Configuration file containing various deployment parameters
- `deploy_and_test.ipynb`: Notebook for initiating model deployment on SageMaker endpoint and testing
- `Dockerfile.server`: Defines the custom container for SageMaker deployment
- `model_data/`: Contains scripts for SageMaker endpoint deployment
- `requirements.txt`: Lists Python dependencies
- `build_and_push.sh`: Script to build and push the Docker image to ECR
- `serve`: Entry point script for the Docker container

## Customization

You can customize the deployment by modifying:
- Deployment parameters in `config.sh`
- Instance type for SageMaker endpoint (default is ml.g5.4xlarge)
- Decoding method and other inference parameters
- Docker image configuration in `Dockerfile.server`

## Debugging

The project includes support for SSH debugging using the SageMaker SSH Helper. This allows you to connect to the SageMaker instance for troubleshooting and development purposes.

## Cleanup

To avoid incurring unnecessary charges, remember to delete the SageMaker endpoint, endpoint configuration, and model when they are no longer needed. You can use the provided cleanup code in the notebook:

```python
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
sess.delete_model(model.name)
```

## Contributing

Contributions to improve the project are welcome. Please follow the standard GitHub pull request process to propose changes.

## License

[Specify the license under which this project is released]

## Acknowledgements

This project uses the following open-source software:
- NVIDIA Triton Inference Server
- OpenAI Whisper
- TensorRT-LLM

Please refer to their respective licenses for terms of use.
