import asyncio
import io
import wave
from enum import Enum
from typing import Annotated, Dict, Type

import numpy as np
from content_size_limit_asgi import ContentSizeLimitMiddleware
from fastapi import Body, FastAPI, File, Form, Query, Response, UploadFile
from starlette.responses import JSONResponse
from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTXT, WriteVTT
from whisper_service.tts import SynthesisParams, async_voice_synthesize
from whisper_service.utils import encode_opus, encode_wav

from whisper_service.whisper import TranscribeParams, async_transcribe

app = FastAPI()

# Limit the size of the incoming audio file to 25 MB
app.add_middleware(ContentSizeLimitMiddleware, max_content_size=25 * 1024 * 1024)


@app.exception_handler(Exception)
async def generic_exception_handler(request, err):
    # Change here to LOGGER
    return JSONResponse(status_code=400, content={"error": repr(err)})


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


# Mimic OpenAI API, see https://platform.openai.com/docs/api-reference/audio/create
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: Annotated[UploadFile, File()],  # Only wav is supported
    model: Annotated[str, Form()],  # Model to use; 'whisper-1' is the only one available now
    prompt: Annotated[
        str | None, Form()
    ] = None,  # An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    response_format: Annotated[ResponseFormat, Form()] = ResponseFormat.JSON,
    temperature: Annotated[float | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = False,  # Whether to return word timestamps
) -> Response:
    # Read & decode the WAV file
    audio = _decode_wav_file(file)

    # Transcribe the audio
    res_dict = await async_transcribe(
        audio,
        TranscribeParams(
            model=model,
            prompt=prompt,
            temperature=temperature,
            language=language,
            word_timestamps=word_timestamps,
        ),
    )

    return _get_response(response_format, res_dict)


def _decode_wav_file(file: UploadFile) -> np.ndarray:
    if file.content_type and file.content_type not in ("audio/wav", "application/octet-stream"):
        raise ValueError(f"Only WAV files are supported. Given: {file.content_type}")

    with wave.open(file.file, "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Only mono audio is supported")
        if wav_file.getframerate() != 16000:
            raise ValueError("Only 16 kHz audio is supported")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Only 16-bit audio is supported")

        audio = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(audio, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0

    return audio


RESPONSE_WRITERS: Dict[ResponseFormat, Type[ResultWriter]] = {
    ResponseFormat.JSON: (WriteJSON, "application/json"),
    ResponseFormat.TEXT: (WriteTXT, "text/plain"),
    ResponseFormat.SRT: (WriteSRT, "text/plain"),
    ResponseFormat.VERBOSE_JSON: (WriteJSON, "application/json"),
    ResponseFormat.VTT: (WriteVTT, "text/vtt"),
}


def _get_response(response_format: ResponseFormat, res_dict: dict) -> Response:
    writer_class, content_type = RESPONSE_WRITERS[response_format]
    writer = writer_class("")

    if response_format == ResponseFormat.JSON:
        res_dict = {"text": res_dict["text"]}  # Just keep the text

    output = io.StringIO()
    writer.write_result(res_dict, output)

    return Response(output.getvalue(), media_type=content_type)


@app.get("/v1/audio/voice_synthesis")
async def voice_synthesis(
    text: Annotated[str, Query(max_length=1000)],
    speaker_idx: str | None = None,
    response_format: str = "opus",
) -> Response:
    params = SynthesisParams(
        speaker_idx=speaker_idx,
    )

    # Synthesize voice
    samples, sampling_rate = await async_voice_synthesize(text, params)

    # Convert to int16
    samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

    match response_format:
        case "wav":
            file_bytes = encode_wav(samples, sampling_rate)
            media_type = "audio/wav"
        case "opus":
            file_bytes = await encode_opus(samples, sampling_rate)
            media_type = "audio/ogg"
        case _:
            raise ValueError(f"Unknown response format: {response_format}")

    return Response(file_bytes, media_type=media_type)
