import logging

import librosa
import numpy as np
from inference import input_fn, model_fn, output_fn, predict_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
logger.info('Loading model...')
model = model_fn('./checkpoint-135')  # 替换为你的模型目录

# 读取音频文件
audio_file = './test.wav'
logger.info(f'Reading audio file: {audio_file}')
try:
    audio, sr = librosa.load(audio_file, sr=None)
    logger.info(f'Audio loaded. Shape: {audio.shape}, Sample rate: {sr}')
except Exception as e:
    logger.error(f'Error loading audio file: {e}')
    raise

# 确保音频是单声道的 float32 类型
if audio.ndim > 1:
    audio = audio.mean(axis=1)
    logger.info(f'Converted to mono. New shape: {audio.shape}')
audio = audio.astype(np.float32)
logger.info(
    f'Converted to float32. Audio stats: min={audio.min()}, max={audio.max()}, mean={audio.mean()}'
)

# 如果采样率不是 16000，需要重采样
if sr != 16000:
    logger.warning(
        f'Sample rate is {sr}, not 16000. Resampling may be needed.')

logger.info(f'Audio shape before tobytes: {audio.shape}')
audio_bytes = audio.tobytes()
logger.info(f'Length of audio bytes: {len(audio_bytes)}')

# 模拟 input_fn
input_data = input_fn(audio_bytes, 'application/octet-stream')

# 运行预测
logger.info('Running prediction...')
prediction = predict_fn(input_data, model)

# 格式化输出
output = output_fn(prediction, 'application/json')

logger.info('Transcription result:')
logger.info(output)
