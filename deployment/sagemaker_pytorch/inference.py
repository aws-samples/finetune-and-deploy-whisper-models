import json
import logging

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info('Loading model...')
    # 加载基础模型
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'openai/whisper-large-v3',
        torch_dtype=torch.float16,
        device_map='auto',
    )
    logger.info(f'Base model dtype: {base_model.dtype}')

    # 加载 LoRA 权重并合并
    peft_model = PeftModel.from_pretrained(base_model, model_dir)
    merged_model = peft_model.merge_and_unload()
    logger.info(f'Merged model dtype: {merged_model.dtype}')

    # 加载处理器
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v3')

    return merged_model, processor


def input_fn(request_body, request_content_type):
    logger.info(f'Received input with content type: {request_content_type}')
    if request_content_type == 'application/octet-stream':
        # 假设输入是原始 PCM 音频数据，采样率为 16000
        audio = np.frombuffer(request_body, dtype=np.float32)
    elif request_content_type == 'application/json':
        # 假设输入是JSON格式的音频数据
        audio_data = json.loads(request_body)['audio']
        audio = np.array(audio_data, dtype=np.float32)
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

    logger.info(f'Input audio shape: {audio.shape}, dtype: {audio.dtype}')
    logger.info(
        f'Raw audio stats: min={audio.min()}, max={audio.max()}, mean={audio.mean()}'
    )
    logger.info(f'Audio sample (first 10 values): {audio[:10]}')
    return audio


def predict_fn(input_data, model):
    merged_model, processor = model
    audio = input_data

    logger.info('Processing audio input...')
    # 处理音频输入，并转换为 float16
    input_features = processor(audio, sampling_rate=16000,
                               return_tensors='pt').input_features
    logger.info(
        f'Input features shape: {input_features.shape}, dtype: {input_features.dtype}'
    )
    logger.info(
        f'Input features stats: min={input_features.min().item()}, max={input_features.max().item()}, mean={input_features.mean().item()}'
    )

    input_features = input_features.to(merged_model.device).half()
    logger.info(
        f'Input features after conversion - shape: {input_features.shape}, dtype: {input_features.dtype}'
    )

    logger.info('Generating transcription...')
    # 生成转录
    generated_ids = merged_model.generate(input_features)
    logger.info(f'Generated IDs: {generated_ids[0][:10]}')  # 只打印前10个ID

    transcription = processor.batch_decode(generated_ids,
                                           skip_special_tokens=True)[0]

    logger.info(
        f'Transcription generated: {transcription[:50]}...')  # 只打印前50个字符
    return transcription


def output_fn(prediction, response_content_type):
    logger.info(
        f'Formatting output with content type: {response_content_type}')
    if response_content_type == 'application/json':
        return json.dumps({'transcription': prediction})
    raise ValueError(f'Unsupported content type: {response_content_type}')
