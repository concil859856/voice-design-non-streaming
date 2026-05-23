# qwen3-voice-design

Non-streaming voice-design HTTP server for [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts). Drop-in replacement for the Chutes `/speak` endpoint the Vocence dashboard-backend used to call — same JSON-in / WAV-out wire format, so only the URL changes in the backend.

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Mode | Non-streaming (one-shot WAV) |
| Hardware | RTX 4090 (24 GB) |
| `cap` default | **1** (one request at a time; second caller gets `server_busy`) |
| Max text length | **2000 chars** (~90 s of speech) |
| Output | WAV (RIFF/WAVE, PCM16LE, 24 kHz mono) |

## Install

```bash
cd /workspace/development/qwen3-voice-design
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ../faster-qwen3-tts
pip install -e .
```

> Note: This server loads a SECOND Qwen3-TTS variant (`-VoiceDesign`) in addition to the cloning server's `-Base`. Both fit on a 24 GB RTX 4090 (~10-12 GB combined), but they share the same GPU lock — voice-design calls wait for any in-flight cloning chunk to finish (~120 ms boundary).

## Run

```bash
cp .env.example .env
# edit .env: set QWEN3_VD_API_KEY at minimum
nano .env

set -a; source .env; set +a
qwen3-voice-design

# or: python -m qwen3_voice_design
# or: qwen3-voice-design --port 8112 --api-key "$QWEN3_VD_API_KEY"
```

On boot:
1. Loads the VoiceDesign model in bf16 on `cuda` (~3.4 GB)
2. Runs a tiny warmup synthesis so CUDA graphs are captured
3. Listens on `127.0.0.1:8112` by default

## Wire protocol

### `GET /healthz`

```json
{
  "status": "ok",
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "sample_rate": 24000,
  "inflight": 0,
  "cap": 1,
  "dev_stub": false
}
```

### `POST /speak`

Auth: `Authorization: Bearer <QWEN3_VD_API_KEY>`

Request body (JSON):
```json
{
  "text": "Hello! This is what your designed voice sounds like.",
  "instruction": "a deep, warm male voice with a slight Scottish accent",
  "language": "English"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1 – 2000 chars |
| `instruction` | string | no | Voice design prompt. Defaults to "neutral voice" if empty. |
| `language` | string | no | One of `Auto`, `English`, `Chinese`, `Japanese`, `Korean`, `Spanish`, `French`, `German`, `Portuguese`, `Italian`, `Russian`. Default `Auto`. |

Response on success:
- `200 OK` with `Content-Type: audio/wav` and the WAV file as body
- `X-Synth-Wall-Ms` header carries the wall-clock synth time

Error responses (all JSON):
```json
{ "type": "error", "code": "<code>", "message": "..." }
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | Missing fields, text too long, invalid JSON |
| 401 | `auth` | Missing / invalid bearer token |
| 503 | `server_busy` | Concurrent caller while at `cap` |
| 504 | `timeout` | Synth exceeded `QWEN3_VD_SYNTH_TIMEOUT_S` |
| 500 | `engine_failed` | Model error (OOM, NaN, etc.) |

## Backend integration

In `dashboard-backend/.env`:
```bash
VOICE_DESIGN_BASE_URL=http://127.0.0.1:8112/speak
VOICE_DESIGN_API_KEY=<same as QWEN3_VD_API_KEY above>
```

When these are set, `studio_tts_service.synthesize_speak()` routes here instead of `https://{slug}.chutes.ai/speak`. The PromptTTS and VoiceDesign-preview code paths both flow through unchanged.

## Layout

```
qwen3_voice_design/
├── __init__.py
├── __main__.py        # python -m qwen3_voice_design
└── server.py          # aiohttp app + synthesis worker
samples/
├── joker.wav          # bundled warmup ref
└── joker.txt
```

## License

MIT.
