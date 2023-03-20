# Simple REST service for Whisper Speech Recognition

Mimics OpenAI API, but using own GPU and able to choose the model.


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
