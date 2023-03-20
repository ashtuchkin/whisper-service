import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from whisper import load_model, transcribe

from whisper_service.config import config

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


def _transcribe_audio(audio: np.ndarray, params: TranscribeParams) -> dict:
    """
    audio should be a numpy array dtype=np.float32, sampling rate 16000, mono.
    """
    assert params.model == "whisper-1", "Only whisper-1 is supported for now"
    kwargs = {}
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.language is not None:
        kwargs["language"] = params.language
    if params.prompt is not None:
        kwargs["initial_prompt"] = params.prompt

    res_dict = transcribe(_tls.model, audio, **kwargs)

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

    fut = loop.run_in_executor(_worker_pool, _transcribe_audio, audio, params)

    fut.add_done_callback(lambda _: _update_tasks_in_progress(-1))

    return fut
