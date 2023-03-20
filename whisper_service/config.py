from pydantic import BaseSettings


class _WhisperServiceConfig(BaseSettings, env_prefix="WS_", env_file=".env"):
    # Whisper Model to use
    model: str = "medium.en"

    # Number of workers to use. Each worker will load the model in GPU memory.
    workers: int = 1

    # Maximum number of requests to queue before rejecting new requests.
    max_queue_size: int = 5


config = _WhisperServiceConfig()
