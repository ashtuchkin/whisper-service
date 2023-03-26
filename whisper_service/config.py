from pathlib import Path
from pydantic import BaseSettings


class _WhisperServiceConfig(BaseSettings, env_prefix="WS_", env_file=".env"):
    # Whisper Model to use
    model: str = "medium.en"

    # Number of workers to use. Each worker will load the model in GPU memory.
    workers: int = 1

    # Maximum number of requests to queue before rejecting new requests.
    max_queue_size: int = 5

    # TTS model to use
    tts_model: str = "tts_models/en/vctk/vits"

    # Default TTS model speaker to use
    tts_speaker_idx: str = "p273"

    # Number of TTS workers to use. Each worker will load the model in GPU memory.
    tts_workers: int = 1

    # Whether to write speech samples to the 'speech_samples' directory
    debug_speech_samples: bool = False

    # Directory to write speech samples to (relative to the current working directory)
    speech_samples_dir: Path = "./speech_samples"


config = _WhisperServiceConfig()
