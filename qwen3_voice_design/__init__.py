"""qwen3-voice-design — non-streaming voice-design HTTP server.

Replaces the Chutes /speak endpoint the Vocence dashboard-backend used to
call. Same JSON request / WAV response wire format, so backend code only
needs the URL changed.
"""
from .server import build_app, run, VoiceDesignServer

__version__ = "0.1.0"
__all__ = ["VoiceDesignServer", "build_app", "run"]
