# Whisper ASR SageMaker Triton 部署

本项目演示了如何使用 NVIDIA Triton 推理服务框架在 Amazon SageMaker 上部署 Whisper 自动语音识别（ASR）模型。它提供了一种可扩展且高效的方法，使用 Whisper large-v3-turbo 模型进行语音到文本的转录。

## 概述

该项目使用自定义 Docker 容器在 SageMaker 上部署 Whisper 模型，利用 Tensorrt-llm 对模型进行编译，提升推理速度，服务器端使用 Triton 部署。它包含了模型部署、端点创建和转录测试的脚本。

## 先决条件

- 具有 SageMaker 访问权限的 AWS 账户
- 本地机器上安装的 Docker
- Python 3.10+
- 配置了适当权限的 AWS CLI
- 访问 AWS ECR 服务权限
- NVIDIA GPU 支持（用于本地测试和开发）

## 模型编译部署

使用 tensorrt-llm 编译后的模型只能在编译时所在的同一类型的机器实例上部署，例如：在 g5.xlarge 上编译的模型只能在 g5 系列的机器上进行部署
1. 克隆此仓库：
   ```
   git clone https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models.git
   cd finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/
   ```

2. 配置部署参数：
   编辑 `config.sh` 文件，设置以下参数：
   - `ACCOUNT`：AWS 账号ID
   - `REGION`：AWS 服务所在地区
   - `PROJECT_ROOT`：项目根目录路径
   - `DOCKER_IMAGE`：Docker 镜像名称
   - `CONDA_ENV`：Conda 环境名称
   - `USE_LORA`：是否使用 LoRA 模型（true/false）
   - `HUGGING_FACE_MODEL_ID`：Hugging Face 模型 ID
   - `LORA_PATH`：LoRA 模型路径（如果使用）
   - `OUTPUT_MODEL_PATH`：输出模型路径
   - `OPENAI_WHISPER_DOWNLOAD_URL`：OpenAI Whisper 模型下载 URL
   - `S3_PATH`：S3 存储路径

3. 运行准备和部署脚本：
   ```
   chmod +x ./prepare_and_deploy.sh && ./prepare_and_deploy.sh
   ```
   此脚本会自动执行以下步骤：
   - 构建并推送 Docker 镜像到 Amazon ECR
   - 准备模型（下载或合并 LoRA 模型）
   - 编译模型
   - 上传编译后的模型到 S3
   - 修改模型部署脚本

4. 模型部署：
   - 在 SageMaker Studio、Jupyter 或已配置好能够访问 AWS 服务的本地机器上打开 [deploy_and_test.ipynb](https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/deploy_and_test.ipynb) notebook
   - 按照 notebook 中的代码执行部署

5. 调用测试：
   部署后，您可以使用 SageMaker 端点进行转录。notebook 中包含了转录音频文件的示例代码，同时也可以参考 [test_whisper_api.py](https://github.com/aws-samples/finetune-and-deploy-whisper-models/blob/main/finetune-and-deploy-whisper-models/deployment/sagemaker_triton_tensorrt_llm/test_whisper_api.py) 进行调用

## Docker 镜像

Docker 镜像基于 NVIDIA Triton 服务器镜像（nvcr.io/nvidia/tritonserver:24.10-py3），并包含以下关键组件：

- 用于音频处理的 FFmpeg
- TensorRT-LLM tensorrt-llm==0.15.0.dev2024110500
- Triton 客户端库
- 自定义 Python 依赖项（参见 requirements.txt）

## Python 依赖项

该项目所需的关键 Python 包包括：

- sagemaker 和 sagemaker-ssh-helper（可选），用于 SageMaker 集成
- boto3，用于 AWS SDK
- openai-whisper，用于 Whisper ASR 模型
- tritonclient[grpc]，用于 Triton 推理服务器客户端
- fastapi 和 uvicorn，用于 API 服务
- 其他音频处理和机器学习库（librosa、soundfile、transformers 等）

完整列表请参阅 `requirements.txt` 文件。

## 关键组件

- `prepare_and_deploy.sh`：自动化准备和部署过程的主脚本
- `config.sh`：配置文件，包含部署所需的各种参数
- `deploy_and_test.ipynb`：用于发起模型在 SageMaker endpoint 的部署和测试
- `Dockerfile.server`：定义用于 SageMaker 部署的自定义容器
- `model_data/`：用于 SageMaker endpoint 部署代码
- `requirements.txt`：Python 依赖项
- `build_and_push.sh`：构建并将 Docker 镜像推送到 ECR 的脚本
- `serve`：Docker 容器的入口脚本

## 自定义

您可以通过修改以下内容来自定义部署：
- `config.sh` 中的部署参数
- SageMkaer endpoint 的实例类型（默认为 ml.g5.4xlarge）
- 解码方法和其他推理参数
- `Dockerfile.server` 中的 Docker 镜像配置

## 调试

该项目包括使用 SageMaker SSH Helper 进行 SSH 调试的支持。这允许您连接到 SageMaker 实例进行故障排除和开发。

## 清理

为避免产生不必要的费用，请记得在不再需要时删除 SageMaker 端点、端点配置和模型。您可以使用笔记本中提供的清理代码：

```python
sess.delete_endpoint(endpoint_name)
sess.delete_endpoint_config(endpoint_name)
sess.delete_model(model.name)
```

## 贡献

欢迎为改进项目做出贡献。请遵循标准的 GitHub 拉取请求流程来提出更改。

## 许可证

[指定该项目发布所依据的许可证]

## 致谢

本项目使用了以下开源软件：
- NVIDIA Triton 推理服务器
- OpenAI Whisper
- TensorRT-LLM

请参阅它们各自的许可证以了解使用条款。
