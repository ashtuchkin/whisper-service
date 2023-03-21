# Simple REST service for Speech Recognition & Generation on a local GPU

Speech recognition uses Whisper Medium model, mimicking OpenAI API.
Speech generation uses Coqui.ai TTS project with multispeaker VITS model.

Main benefit is using own GPU and low latency. Both models should fit into 8-10Gb of GPU mem.

## Running manually

```
poetry install

WS_MODEL=medium.en poetry run uvicorn whisper_service.main:app --host '' --port 8000  --limit-concurrency 10
```

## Running with systemd

```
sudo cp -f deploy/whisper.service /etc/systemd/system/whisper.service
sudo systemctl daemon-reload
sudo systemctl start whisper
sudo systemctl enable whisper
```


## TTS
Using coqui TTS:

tts_models/en/ljspeech/tacotron2-DDC + vocoder_models/en/ljspeech/hifigan_v2 is best-sounding. But incorrectly pronounces "labels".
Real-time factor is usually ~0.053. 350ms for a medium size sentence.

tts_models/en/ljspeech/fast_pitch + vocoder_models/en/ljspeech/hifigan_v2 is good too.

tts_models/en/vctk/vits - multi-speaker; good: 336! 273, 335
Seems like VITS requires a lot of GPU memory (rapidly increases by sentence length).

TortoiseTTS is too slow (2 mins for 1 sentence)

https://news.ycombinator.com/item?id=34215252
You're looking for Tacotron 2 or one of its offshoots that add multi-speaker, TorchMoji, etc. You'll want to pair it with the Hifi-Gan vocoder to get end-to-end text to speech. (Avoid Griffin-Lim and WaveGlow.)

Training own model: https://alexpeattie.com/talks/tts/?f=5#109
Needs >15 hours of speech if training from scratch.
Nice dataset: https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/

Hi there! I'm sorry, I am not capable of having a name as I am an AI language model. But you can call me whatever you'd like! How can I assist you today?

Either VITS or Tacotron2-DDC should be easily fine-tuned
https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/vits_tts/train_vits.py
