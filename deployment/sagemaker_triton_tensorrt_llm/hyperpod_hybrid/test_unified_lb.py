#!/usr/bin/env python3
import requests
import base64
import json
import time
import soundfile as sf
import io
import numpy as np

def create_test_audio():
    """Create simple test audio"""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def test_unified_endpoint(url, num_requests=5):
    """Test unified LoadBalancer with multiple requests"""
    print(f"Testing unified endpoint: {url}")
    
    # Test ping
    try:
        response = requests.get(f"{url}/ping", timeout=10)
        print(f"✅ Ping: {response.status_code}")
    except Exception as e:
        print(f"❌ Ping failed: {e}")
        return
    
    # Test multiple transcription requests
    audio_b64 = create_test_audio()
    payload = {
        'whisper_prompt': '',
        'audio_data': audio_b64,
        'repetition_penalty': 1.0
    }
    
    print(f"\nTesting {num_requests} transcription requests...")
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(f"{url}/invocations", json=payload, timeout=30)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"Request {i+1}: ✅ {latency:.0f}ms - {result.get('transcribe_text', 'No text')}")
            else:
                print(f"Request {i+1}: ❌ {response.status_code}")
        except Exception as e:
            print(f"Request {i+1}: ❌ {e}")
        
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    # Get service endpoint
    import subprocess
    try:
        # Load config if available
        service_name = "whisper-triton-unified-nlb"  # Default value
        api_port = "8080"  # Default value
        
        # Try to load from config.env if available
        try:
            import os
            if os.path.exists("config.env"):
                with open("config.env", "r") as f:
                    for line in f:
                        if line.startswith("SERVICE_NAME="):
                            service_name = line.split("=", 1)[1].strip()
                        elif line.startswith("API_PORT="):
                            api_port = line.split("=", 1)[1].strip()
        except:
            pass  # Use defaults if config loading fails
            
        result = subprocess.run([
            'kubectl', 'get', 'service', service_name, 
            '-o', 'jsonpath={.status.loadBalancer.ingress[0].hostname}'
        ], capture_output=True, text=True, check=True)
        
        hostname = result.stdout.strip()
        if hostname:
            url = f"http://{hostname}:{api_port}"
            test_unified_endpoint(url)
        else:
            print("❌ LoadBalancer hostname not ready yet")
    except Exception as e:
        print(f"❌ Error getting service endpoint: {e}")
        print("Please run: kubectl get services | grep whisper")
