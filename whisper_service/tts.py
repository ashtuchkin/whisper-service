import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Tuple

import numpy as np
from pydantic import BaseModel

from whisper_service.config import config

if TYPE_CHECKING:
    from TTS.api import TTS

_tls = threading.local()


def _init_tts_worker():
    print(f"Importing TTS module")
    # Heavy imports
    import torch
    from TTS.api import TTS

    print(f"Loading TTS model '{config.tts_model}'")
    start_time = time.time()
    _tls.model = TTS(model_name=config.tts_model, progress_bar=False, gpu=torch.cuda.is_available())
    print(f"Loading TTS took {time.time() - start_time:.2f} seconds")


# Main worker thread that'll process the files. We use a thread pool to limit the number of concurrent requests.
_worker_pool = ThreadPoolExecutor(max_workers=config.tts_workers, initializer=_init_tts_worker)


class SynthesisParams(BaseModel):
    speaker_idx: str | None


def _voice_synthesize(text: str, params: SynthesisParams) -> Tuple[bytes, int]:
    model: TTS = _tls.model

    samples_list = _tls.model.tts(text, speaker=params.speaker_idx or config.tts_speaker_idx)

    samples = np.array(samples_list, dtype=np.float32)

    return samples, model.synthesizer.output_sample_rate


def async_voice_synthesize(
    text: str, params: SynthesisParams
) -> asyncio.Future[Tuple[np.ndarray, int]]:
    """
    Returns:
        A tuple of (audio, sample_rate). Audio is a numpy array dtype=np.float32, mono.
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_worker_pool, _voice_synthesize, text, params)
