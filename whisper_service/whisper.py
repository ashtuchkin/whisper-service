import asyncio
from datetime import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from whisper import load_model, transcribe

from whisper_service.config import config
from whisper_service.utils import encode_wav

_tls = threading.local()


def _init_worker():
    print(f"Loading model '{config.model}'")
    start_time = time.time()
    _tls.model = load_model(config.model)
    print(f"Loading successful after {time.time() - start_time:.2f} seconds")


# Main worker thread that'll process the files. We use a thread pool to limit the number of concurrent requests.
_worker_pool = ThreadPoolExecutor(max_workers=config.workers, initializer=_init_worker)
_tasks_in_progress = 0


class TranscribeParams(BaseModel):
    model: str = "whisper-1"  # Model to use; 'whisper-1' is the only one available now
    prompt: str | None = None  # An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    temperature: float | None = None
    language: str | None = None
    word_timestamps: bool = False  # Whether to return word timestamps


def _transcribe_audio(audio: np.ndarray, params: TranscribeParams, loop: asyncio.BaseEventLoop) -> dict:
    """
    audio should be a numpy array dtype=np.float32, sampling rate 16000, mono.
    """
    assert params.model == "whisper-1", "Only whisper-1 is supported for now"
    sampling_rate = 16000
    kwargs = {}
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.language is not None:
        kwargs["language"] = params.language
    if params.prompt is not None:
        kwargs["initial_prompt"] = params.prompt
    kwargs["word_timestamps"] = params.word_timestamps

    res_dict = transcribe(_tls.model, audio, **kwargs)

    if config.debug_speech_samples:
        # Don't block the thread
        debug_dict = res_dict | {"prompt": params.prompt, "temperature": params.temperature}
        loop.run_in_executor(None, _debug_write_samples, audio, sampling_rate, debug_dict)

    return res_dict


def _update_tasks_in_progress(delta: int):
    global _tasks_in_progress
    _tasks_in_progress += delta


def async_transcribe(audio: np.ndarray, params: TranscribeParams) -> asyncio.Future[dict]:
    """
    audio should be a numpy array dtype=np.float32, sampling rate 16000, mono.
    """
    loop = asyncio.get_event_loop()

    # Shed the load
    if _tasks_in_progress >= config.workers + config.max_queue_size:
        fut = loop.create_future()
        fut.set_exception(HTTPException(status_code=429, detail="Too many requests in progress"))
        return fut

    _update_tasks_in_progress(1)

    fut = loop.run_in_executor(_worker_pool, _transcribe_audio, audio, params, loop)

    fut.add_done_callback(lambda _: _update_tasks_in_progress(-1))

    return fut


def _debug_write_samples(audio: np.ndarray, sampling_rate: int, res_dict: dict):
    config.speech_samples_dir.mkdir(parents=True, exist_ok=True)
    curtime = datetime.now().isoformat(timespec='milliseconds').replace(":", "-")
    duration = len(audio) / sampling_rate
    text = res_dict["text"].replace("/", "-")

    filename = config.speech_samples_dir / f"{curtime} {duration:04.1f} {text} .wav"

    audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    filename.write_bytes(encode_wav(audio, sampling_rate))
    filename.with_suffix(".json").write_text(json.dumps(res_dict, indent=2))
