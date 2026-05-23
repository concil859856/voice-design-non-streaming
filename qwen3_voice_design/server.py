"""Non-streaming voice-design HTTP server for faster-qwen3-tts.

Drop-in replacement for the Chutes voice-design endpoint the Vocence
dashboard-backend used to call (``synthesize_speak`` in studio_tts_service).
Wire-compatible so the only backend change needed is pointing ``base_url``
at this server.

  POST /speak                  -> JSON {text, instruction, language?} → WAV bytes
  GET  /healthz                -> JSON {status, model_id, sample_rate, inflight, cap, dev_stub}

Designed for one RTX 4090 running Qwen3-TTS-12Hz-1.7B-VoiceDesign:

  * Single-GPU lock serializes the model. ``cap`` defaults to 1 — second
    request gets ``server_busy`` immediately rather than queueing, matching
    the user's "one request at a time" preference. Bump via env to allow
    queueing.
  * ``inflight`` decrement is structural via @asynccontextmanager — every
    close path (success, error, client disconnect, server panic) frees the
    slot exactly once. No counter leaks.
  * MAX_TEXT_CHARS default 2000 — 6.7× the previous Chutes 300-char cap.
    ~80-90 s of audio output, well within the 3072-token KV budget.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aiohttp import web

from faster_qwen3_tts import FasterQwen3TTS

_log = logging.getLogger("qwen3_voice_design")

# --- Output format (matches the Vocence backend's expectations) ------------
SAMPLE_RATE = 24_000  # Qwen3-TTS speech tokenizer output rate

# --- Spec languages (unknown values pass through to the model) ------------
_KNOWN_LANGUAGES = {
    "Auto", "English", "Chinese", "Japanese", "Korean", "Spanish", "French",
    "German", "Portuguese", "Italian", "Russian", "Arabic",
}

# --- Default warmup ref audio (bundled) -----------------------------------
_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
DEFAULT_WARMUP_REF_AUDIO = str(_SAMPLES_DIR / "joker.wav")
DEFAULT_WARMUP_REF_TEXT_PATH = _SAMPLES_DIR / "joker.txt"


def _load_default_warmup_ref_text() -> str:
    try:
        return DEFAULT_WARMUP_REF_TEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# --- Tunable knobs (env-overridable) ---------------------------------------
def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


MAX_TEXT_CHARS = _env_int("QWEN3_VD_MAX_CHARS", 2000)
DEFAULT_CAP = _env_int("QWEN3_VD_CAP", 1)
SYNTH_HARD_TIMEOUT_S = _env_float("QWEN3_VD_SYNTH_TIMEOUT_S", 120.0)
MAX_SEQ_LEN = _env_int("QWEN3_VD_MAX_SEQ_LEN", 3072)

# Sampling — match faster-qwen3-tts defaults; tweak via env if needed.
TEMPERATURE = _env_float("QWEN3_VD_TEMPERATURE", 0.9)
TOP_K = _env_int("QWEN3_VD_TOP_K", 50)
TOP_P = _env_float("QWEN3_VD_TOP_P", 1.0)
REPETITION_PENALTY = _env_float("QWEN3_VD_REPETITION_PENALTY", 1.05)
MAX_NEW_TOKENS = _env_int("QWEN3_VD_MAX_NEW_TOKENS", 4096)


# ---------------------------------------------------------------------------
# Inflight tracking (structural — every close path decrements once)
# ---------------------------------------------------------------------------

class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int:
        return self._cap

    @property
    def inflight(self) -> int:
        return self._count

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            if self._count >= self._cap:
                raise _ServerBusy()
            self._count += 1
        try:
            yield
        finally:
            async with self._lock:
                self._count = max(0, self._count - 1)


class _ServerBusy(Exception):
    pass


class _BadRequest(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------

def _float32_to_pcm16le(samples: np.ndarray) -> bytes:
    if samples.ndim > 1:
        samples = samples.reshape(-1)
    return np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Wrap float32 PCM samples in a complete WAV file (RIFF/WAVE, PCM16LE mono)."""
    pcm = _float32_to_pcm16le(samples)
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    data_len = len(pcm)
    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + data_len))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    out.write(b"data")
    out.write(struct.pack("<I", data_len))
    out.write(pcm)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def _parse_speak_request(payload: Any) -> tuple[str, str, str]:
    """Return (text, instruction, language). Raises _BadRequest on protocol errors."""
    if not isinstance(payload, dict):
        raise _BadRequest("request body must be a JSON object")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _BadRequest("`text` is required and must be a non-empty string")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise _BadRequest(f"`text` too long ({len(text)} chars; max {MAX_TEXT_CHARS})")

    instruction = payload.get("instruction") or ""
    if not isinstance(instruction, str):
        raise _BadRequest("`instruction` must be a string when present")
    instruction = instruction.strip() or "neutral voice"

    language = payload.get("language") or "Auto"
    if not isinstance(language, str) or not language.strip():
        language = "Auto"
    language = language.strip()
    if language not in _KNOWN_LANGUAGES:
        _log.debug("non-standard language %r — passing through to model", language)

    return text, instruction, language


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class VoiceDesignServer:
    def __init__(
        self,
        model: FasterQwen3TTS,
        *,
        api_key: str | None,
        cap: int,
        model_id: str,
    ) -> None:
        self._model = model
        self._model_id = model_id
        self._api_key = (api_key or "").strip()
        self._inflight = _InflightTracker(cap=cap)
        self._gpu_lock = threading.Lock()

    # ---- HTTP -------------------------------------------------------------
    async def healthz(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "model_id": self._model_id,
            "sample_rate": SAMPLE_RATE,
            "inflight": self._inflight.inflight,
            "cap": self._inflight.cap,
            "dev_stub": False,
        })

    async def speak(self, request: web.Request) -> web.Response:
        # Bearer auth (skip when no key configured — dev mode).
        if self._api_key:
            header = request.headers.get("Authorization", "")
            if header != f"Bearer {self._api_key}":
                return _error_json(401, "auth", "missing or invalid bearer token")

        # Parse JSON.
        try:
            payload = await request.json()
        except json.JSONDecodeError as e:
            return _error_json(400, "bad_request", f"invalid JSON: {e}")
        except Exception as e:
            return _error_json(400, "bad_request", f"failed to read body: {e}")

        try:
            text, instruction, language = _parse_speak_request(payload)
        except _BadRequest as e:
            return _error_json(400, "bad_request", e.message)

        # Slot acquisition — strict accept-or-reject. With cap=1 (default),
        # any concurrent caller gets 503 immediately rather than queueing.
        try:
            slot_cm = self._inflight.slot()
            await slot_cm.__aenter__()
        except _ServerBusy:
            return _error_json(503, "server_busy", f"inflight cap ({self._inflight.cap}) exhausted")

        t_start = time.perf_counter()
        try:
            try:
                wav_bytes = await asyncio.wait_for(
                    self._synthesize(text=text, instruction=instruction, language=language),
                    timeout=SYNTH_HARD_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return _error_json(504, "timeout", f"synth exceeded {SYNTH_HARD_TIMEOUT_S}s")
            except _BadRequest as e:
                return _error_json(400, "bad_request", e.message)
            except Exception as e:
                _log.exception("synthesis failed")
                return _error_json(500, "engine_failed", f"{type(e).__name__}: {e}")

            elapsed = time.perf_counter() - t_start
            _log.info(
                "speak ok text=%d chars audio=%d bytes wall=%.2fs",
                len(text), len(wav_bytes), elapsed,
            )
            return web.Response(
                body=wav_bytes,
                content_type="audio/wav",
                headers={"X-Synth-Wall-Ms": str(int(elapsed * 1000))},
            )
        finally:
            await slot_cm.__aexit__(None, None, None)

    # ---- model invocation --------------------------------------------------
    async def _synthesize(
        self,
        *,
        text: str,
        instruction: str,
        language: str,
    ) -> bytes:
        """Run generate_voice_design in a worker thread (it's blocking CUDA work)."""
        loop = asyncio.get_running_loop()

        def _run() -> bytes:
            with self._gpu_lock:
                audio_arrays, sr = self._model.generate_voice_design(
                    text=text,
                    instruct=instruction,
                    language=language,
                    non_streaming_mode=True,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    do_sample=True,
                    repetition_penalty=REPETITION_PENALTY,
                )
            if not audio_arrays:
                return _wav_bytes(np.zeros(1, dtype=np.float32), sr)
            audio = audio_arrays[0]
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            return _wav_bytes(np.asarray(audio, dtype=np.float32), sr)

        return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_json(http_status: int, code: str, message: str) -> web.Response:
    """JSON error response matching the streaming server's shape so the
    backend's error parsing works identically for both services."""
    return web.json_response(
        {"type": "error", "code": code, "message": message[:200]},
        status=http_status,
    )


def _warmup(model: FasterQwen3TTS) -> None:
    """Run one tiny voice-design synthesis at boot so CUDA graphs are captured
    before the first real request. Without this, the first /speak call eats
    2-5 s of graph capture latency."""
    _log.info("warming up CUDA graphs…")
    t0 = time.perf_counter()
    try:
        model.generate_voice_design(
            text="Hello there.",
            instruct="a friendly neutral voice",
            language="English",
            non_streaming_mode=True,
            max_new_tokens=64,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=REPETITION_PENALTY,
        )
        _log.info("warmup complete in %.2fs", time.perf_counter() - t0)
    except Exception as e:
        _log.warning("warmup failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# App factory + entrypoint
# ---------------------------------------------------------------------------

def build_app(
    *,
    model: FasterQwen3TTS,
    api_key: str | None,
    cap: int,
    model_id: str,
) -> web.Application:
    server = VoiceDesignServer(
        model,
        api_key=api_key,
        cap=cap,
        model_id=model_id,
    )
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app.router.add_get("/healthz", server.healthz)
    app.router.add_post("/speak", server.speak)
    app["server"] = server
    return app


def run(
    *,
    model_id: str,
    host: str,
    port: int,
    api_key: str | None,
    cap: int,
    device: str,
    dtype: str,
    warmup_enabled: bool,
) -> None:
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    _log.info("loading model %s on %s (%s)…", model_id, device, dtype)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        model_id,
        device=device,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        max_seq_len=MAX_SEQ_LEN,
    )
    _log.info(
        "model ready in %.1fs (sample rate %d Hz, max_seq_len %d)",
        time.perf_counter() - t0, model.sample_rate, MAX_SEQ_LEN,
    )

    if model.sample_rate != SAMPLE_RATE:
        _log.error(
            "model sample rate %d != server SAMPLE_RATE %d — output will sound wrong",
            model.sample_rate, SAMPLE_RATE,
        )

    if warmup_enabled:
        _warmup(model)

    app = build_app(model=model, api_key=api_key, cap=cap, model_id=model_id)

    _log.info(
        "serving on %s:%d  (cap=%d max_chars=%d max_seq_len=%d auth=%s)",
        host, port, cap, MAX_TEXT_CHARS, MAX_SEQ_LEN,
        "on" if api_key else "OFF",
    )
    web.run_app(app, host=host, port=port, print=None)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen3-voice-design",
        description="Non-streaming voice-design HTTP server (Qwen3-TTS VoiceDesign)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN3_VD_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        help="HF model id or local path (default: Qwen3-TTS-12Hz-1.7B-VoiceDesign)",
    )
    p.add_argument("--host", default=os.environ.get("QWEN3_VD_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=_env_int("QWEN3_VD_PORT", 8112))
    p.add_argument(
        "--api-key",
        default=os.environ.get("QWEN3_VD_API_KEY"),
        help="Required bearer token. Empty disables auth (dev only).",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help="Max concurrent requests. Default 1 — second caller gets server_busy.",
    )
    p.add_argument("--device", default=os.environ.get("QWEN3_VD_DEVICE", "cuda"))
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the boot warmup synthesis (first request pays the capture latency)",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("QWEN3_VD_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        model_id=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        cap=args.cap,
        device=args.device,
        dtype=args.dtype,
        warmup_enabled=not args.no_warmup,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
