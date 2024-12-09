# whisper_api.py

import time
from datetime import datetime
import numpy as np
import logging
import base64
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import soundfile as sf
import io
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, parse_obj_as
import asyncio
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor

server_url = "127.0.0.1:8001"
triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=False)
protocol_client = grpcclient

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceOpt(BaseModel):
    whisper_prompt: str = Field(default="")
    audio_data: str = Field(...)  # Base64 encoded audio data
    repetition_penalty: float = Field(default=1.0)  # New field for repetition penalty

# 创建一个线程池来管理Triton客户端连接
triton_client_pool = ThreadPoolExecutor(max_workers=10)

async def get_triton_client():
    return await asyncio.get_event_loop().run_in_executor(
        triton_client_pool, 
        lambda: grpcclient.InferenceServerClient(url=server_url, verbose=False)
    )

async def send_whisper(whisper_prompt, audio_data, model_name, repetition_penalty, padding_duration=15):
    start_time = time.time()
    
    # Decode base64 audio data
    audio_bytes = base64.b64decode(audio_data)
    
    # Load audio from bytes
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file)
    
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    try:
        duration = int(len(waveform) / sample_rate)
    except Exception as e:
        print(e)
        print("length of waveform: ", len(waveform), int(len(waveform)))

    # padding to nearest 10 seconds
    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate * ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )

    samples[0, : len(waveform)] = waveform

    inputs = [
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "TEXT_PREFIX", [1, 1], "BYTES"
        ),
        protocol_client.InferInput(
            "REPETITION_PENALTY", [1, 1], "FP32"  # Changed shape to [1, 1]
        ),
    ]
    # logger.info(f"inputs: {inputs}")
    inputs[0].set_data_from_numpy(samples)

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[1].set_data_from_numpy(input_data_numpy)

    # Create repetition_penalty as a 2D array with shape [1, 1]
    repetition_penalty_array = np.array([[repetition_penalty]], dtype=np.float32)
    logger.info(f"Repetition penalty array shape: {repetition_penalty_array.shape}")
    logger.info(f"Repetition penalty array: {repetition_penalty_array}")
    
    inputs[2].set_data_from_numpy(repetition_penalty_array)
    logger.info(f"Repetition penalty input shape after set_data_from_numpy: {inputs[2].shape()}")

    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
    sequence_id = np.random.randint(0, 1000000)

    inference_start_time = time.time()
    response = triton_client.infer(
        model_name, inputs, request_id=str(sequence_id), outputs=outputs
    )
    inference_end_time = time.time()

    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        decoding_results = decoding_results.decode("utf-8")

    end_time = time.time()
    logging.info(f"Whisper processing time: {end_time - start_time:.3f} seconds")
    logging.info(f"Inference time: {inference_end_time - inference_start_time:.3f} seconds")
    return decoding_results, duration

async def process(
    whisper_prompt: str,
    audio_data: str,
    repetition_penalty: float,
):
    overall_start_time = time.time()

    logging.info(f"whisper_prompt: {whisper_prompt}")
    logging.info(f"repetition_penalty: {repetition_penalty}")

    model_name = "whisper"

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    logging.info(f"Started at {date_time}")

    whisper_start_time = time.time()
    text, duration = await send_whisper(whisper_prompt, audio_data, model_name, repetition_penalty)
    whisper_end_time = time.time()

    date_time = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    overall_end_time = time.time()

    try:
        if duration == 0:
            rtf = overall_end_time - overall_start_time  # 或者设置为 0，取决于你的需求
        else:
            rtf = (overall_end_time - overall_start_time) / duration
    except Exception as e:
        print(e)
        print("duration: ", duration)


    logging.info(f"Finished at {date_time}")
    logging.info(f"Total processing time: {overall_end_time - overall_start_time:.3f} seconds")
    logging.info(f"Whisper processing time: {whisper_end_time - whisper_start_time:.3f} seconds")

    info = f"""
    Wave duration  : {duration:.3f} s
    Processing time: {overall_end_time - overall_start_time:.3f} s
    RTF: {overall_end_time - overall_start_time:.3f}/{duration:.3f} = {rtf:.3f}
    """
    if rtf > 1 and duration != 0:
        info += (
            "We are loading the model for the first run. "
            "Please run again to measure the real RTF."
        )

    logging.info(info)
    logging.info(f"\nresult: {text}")

    return text

@app.get("/ping")
async def ping():
    url = "http://127.0.0.1:10086/v2/health/ready"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
                else:
                    return JSONResponse({"code": 1, "message": "Fail"}, status_code=500)
    except Exception as e:
        logger.error(f"Error pinging Triton server: {str(e)}")
        return JSONResponse({"code": 1, "message": "Fail"}, status_code=500)

@app.post("/invocations")
async def invocations(request: Request):
    try:
        json_post_raw = await request.json()
        opt = parse_obj_as(InferenceOpt, json_post_raw)
        text = await process(opt.whisper_prompt, opt.audio_data, opt.repetition_penalty)
        return JSONResponse({"code": 0, "message": "Success", "transcribe_text": text}, status_code=200)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse({"code": 1, "message": f"Error: {str(e)}"}, status_code=500)
