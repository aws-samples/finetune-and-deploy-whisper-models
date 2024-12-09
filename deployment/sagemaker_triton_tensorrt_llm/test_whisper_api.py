import requests
import base64
import json
import argparse
import time
import io
import numpy as np
import statistics
import librosa
import soundfile as sf
from pydub import AudioSegment

def encode_audio(audio_file_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(audio_file_path)
    
    # 检查是否为双通道
    if audio.channels == 2:
        print("检测到双通道音频，正在转换为单通道...")
        # 将双通道转换为单通道
        audio = audio.set_channels(1)
    
    # 将音频数据写入内存缓冲区
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    
    # 将缓冲区的内容编码为 base64
    return base64.b64encode(buffer.read()).decode('utf-8')

def test_ping(base_url):
    response = requests.get(f"{base_url}/ping")
    print("Ping Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    print()

def test_transcription(base_url, audio_file_path, language="", repo_id="whisper-large-v3", 
                      decoding_method="greedy_search", whisper_prompt="", 
                      repetition_penalty=1, max_new_tokens=96):
    encoded_audio = encode_audio(audio_file_path)

    payload = {
        "max_new_tokens": max_new_tokens,
        "whisper_prompt": whisper_prompt,
        "audio_data": encoded_audio,
        "repetition_penalty": repetition_penalty
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    response = requests.post(f"{base_url}/invocations", json=payload, headers=headers)
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return response, latency

def save_first_15_seconds(input_file, output_file):
    """
    使用 librosa 提取前15秒音频
    
    参数:
    - input_file: 输入音频文件路径
    - output_file: 输出音频文件路径
    """
    try:
        # 加载音频文件，只加载前15秒
        y, sr = librosa.load(input_file, sr=16000, duration=15)
        
        # 保存前15秒音频
        sf.write(output_file, y, sr)
        
        print(f"Successfully extracted first 15 seconds to {output_file}")
    
    except Exception as e:
        print(f"Error extracting audio: {e}")

def print_statistics(latencies):
    print("\n=== Latency Statistics (milliseconds) ===")
    print(f"Average: {statistics.mean(latencies):.2f}ms")
    print(f"Minimum: {min(latencies):.2f}ms")
    print(f"Maximum: {max(latencies):.2f}ms")
    print(f"Median: {statistics.median(latencies):.2f}ms")
    if len(latencies) > 1:
        print(f"Standard Deviation: {statistics.stdev(latencies):.2f}ms")
        print(f"95th Percentile: {np.percentile(latencies, 95):.2f}ms")
        print(f"99th Percentile: {np.percentile(latencies, 99):.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Whisper API")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Base URL of the Whisper API")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--language", type=str, default="", help="Language of the audio (optional)")
    parser.add_argument("--repo_id", type=str, default="whisper-large-v3", help="Model repository ID")
    parser.add_argument("--decoding_method", type=str, default="greedy_search", help="Decoding method")
    parser.add_argument("--prompt", type=str, default="", help="Whisper prompt (optional)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--max_new_tokens", type=int, default=96, help="Numbers of generate new tokens")
    parser.add_argument("--num_requests", type=int, default=1, help="Number of requests to send")
    parser.add_argument("--interval", type=float, default=0.0, help="Interval between requests in seconds")
    
    args = parser.parse_args()
    args.url ="http://127.0.0.1:8080"
    print("Testing Ping...")
    test_ping(args.url)

    print(f"\nSending {args.num_requests} requests with {args.interval}s interval...")
    latencies = []
    
    output_audio = "first_15_seconds.wav"
    save_first_15_seconds(args.audio, output_audio)
    
    for i in range(args.num_requests):
        print(f"\nRequest {i+1}/{args.num_requests}:")
        response, latency = test_transcription(
            args.url, output_audio, args.language, args.repo_id,
            args.decoding_method, args.prompt, args.repetition_penalty,
            args.max_new_tokens
        )
        
        print("Transcription Response:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        print(f"Status Code: {response.status_code}")
        print(f"Request Latency: {latency:.2f}ms")
        
        latencies.append(latency)
        
        # Sleep for the specified interval if not the last request
        if i < args.num_requests - 1 and args.interval > 0:
            time.sleep(args.interval)
    
    print("Testing Transcription...")
    encoded_audio = encode_audio(args.audio)
    # args.prompt = "<|startofprev|>This is MOBA game text<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
    # text = process(args.language, args.repo_id, args.decoding_method, args.prompt, encoded_audio)
    test_transcription(args.url, args.audio, args.language, args.repo_id, args.decoding_method, args.prompt, args.repetition_penalty)
    # Print timing statistics
    print_statistics(latencies)
