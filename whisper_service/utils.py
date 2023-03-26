import asyncio
import io
import wave
import numpy as np


async def encode_opus(samples: np.ndarray, sampling_rate: int) -> bytes:
    """ Encodes the given audio samples to Opus format. Samples must be np.int16 """
    assert samples.dtype == np.int16
    assert samples.ndim == 1

    # Note, default parameters are ok for us: 64kbps VBR, 48kHz, mono.
    # start_time = time.time()
    proc = await asyncio.create_subprocess_exec(
        "opusenc",
        "--raw",
        "--quiet",
        f"--raw-rate={sampling_rate}",
        "--raw-chan=1",
        "-",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        limit=256 * 1024,
    )
    # print(f"Process creation took {time.time() - start_time:.2f}s")  # 0.04s
    output, _ = await proc.communicate(input=samples.data)
    # print(f"Encoding took {time.time() - start_time:.2f}s")  # 0.11s - TODO: optimize
    return output


def encode_wav(samples: np.ndarray, sampling_rate: int) -> bytes:
    """ Encodes the given audio samples to WAV format. Samples must be np.int16 """
    assert samples.dtype == np.int16
    assert samples.ndim == 1

    wav_file = io.BytesIO()
    with wave.open(wav_file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(samples.data)
    return wav_file.getvalue()
