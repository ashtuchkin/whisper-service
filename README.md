# Simple REST service for Whisper Speech Recognition

Mimics OpenAI API, but using own GPU and able to choose the model.


## Running

```
poetry install

WS_MODEL=medium.en poetry run uvicorn whisper_service.main:app --host '' --port 8000  --limit-concurrency 10
```
