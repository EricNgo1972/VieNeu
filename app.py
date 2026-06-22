"""
VieNeu-TTS → OpenAI-compatible TTS shim.

Why this exists
---------------
VieNeu-TTS has NO HTTP "text → audio" endpoint. Its :23333 server is a plain
LMDeploy LLM server that returns speech *codec tokens* over /v1/chat/completions;
the audio is reconstructed client-side by the `vieneu` Python SDK (NeuCodec decode).
Its Gradio app (port 7860) only speaks the Gradio call protocol.

This service wraps the `vieneu` SDK and exposes a real OpenAI-style endpoint:

    POST /v1/audio/speech   {model, input, voice, response_format} -> audio bytes

so any OpenAI-compatible client (e.g. Maple Video Studio) can synthesize over plain
HTTP. It also exposes /v1/audio/voices and /v1/models so clients can discover the
real Vietnamese voice list.

Modes (env VIENEU_MODE)
-----------------------
  local  (default) : runs v3 Turbo on CPU via ONNX (torch-free, self-contained,
                     no GPU, no dependency on any other server).
  remote           : offloads token generation to a VieNeu LMDeploy server
                     (VIENEU_API_BASE, e.g. http://gpu-host:23333/v1) and decodes
                     the codec locally. Use this to drive the v2 (GPU) model.
"""
import io
import os
import logging
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vieneu-shim")

# ── Config (all via env) ────────────────────────────────────────────────────
MODE          = os.getenv("VIENEU_MODE", "local").strip().lower()
MODEL_NAME    = os.getenv("VIENEU_MODEL", "pnnbao-ump/VieNeu-TTS-v2").strip()
API_BASE      = os.getenv("VIENEU_API_BASE", "http://localhost:23333/v1").strip()
EMOTION       = os.getenv("VIENEU_EMOTION", "natural").strip()        # natural | storytelling
DEFAULT_VOICE = os.getenv("VIENEU_DEFAULT_VOICE", "").strip()         # optional preset id
HF_TOKEN      = os.getenv("HF_TOKEN", "").strip() or None
SHIM_API_KEY  = os.getenv("SHIM_API_KEY", "").strip()                 # if set, require Bearer

# soundfile/libsndfile subtype per container format
_FORMATS = {
    "mp3":  ("MP3", "audio/mpeg"),
    "wav":  ("WAV", "audio/wav"),
    "flac": ("FLAC", "audio/flac"),
    "ogg":  ("OGG", "audio/ogg"),
    "opus": ("OGG", "audio/ogg"),
}

app = FastAPI(title="VieNeu-TTS OpenAI shim", version="1.0")

# Single model instance, guarded — `infer` is CPU-bound and not thread-safe.
_tts = None
_lock = threading.Lock()
# voice index: lower(id|description) -> canonical preset id
_voice_index: dict[str, str] = {}
_voices: list[tuple[str, str]] = []   # (description, id) as returned by the SDK


def _load():
    """Lazily construct the Vieneu engine and cache its preset voices."""
    global _tts, _voices, _voice_index
    if _tts is not None:
        return _tts
    with _lock:
        if _tts is not None:
            return _tts
        from vieneu import Vieneu  # imported here so /health works before models download
        log.info("Loading VieNeu (mode=%s, model=%s, emotion=%s)…", MODE, MODEL_NAME, EMOTION)
        if MODE == "remote":
            _tts = Vieneu(mode="remote", api_base=API_BASE, model_name=MODEL_NAME,
                          emotion=EMOTION, hf_token=HF_TOKEN)
        else:
            # Default v3 Turbo, CPU/ONNX, torch-free.
            _tts = Vieneu(emotion=EMOTION, hf_token=HF_TOKEN)
        try:
            _voices = list(_tts.list_preset_voices() or [])
        except Exception as e:  # noqa: BLE001
            log.warning("Could not list preset voices: %s", e)
            _voices = []
        idx = {}
        for desc, vid in _voices:
            if vid:
                idx[str(vid).lower()] = vid
            if desc:
                idx.setdefault(str(desc).lower(), vid)
        _voice_index = idx
        log.info("Loaded %d preset voices.", len(_voices))
        return _tts


def _resolve_voice(name: str | None):
    """Map a requested voice string to a preset voice dict. Returns None to use the
    model default (so an unknown/stale value degrades gracefully instead of 500ing)."""
    want = (name or DEFAULT_VOICE or "").strip()
    if not want:
        return None
    key = want.lower()
    vid = _voice_index.get(key)
    if vid is None:  # loose contains-match on description/id
        for k, v in _voice_index.items():
            if key in k:
                vid = v
                break
    if vid is None:
        log.warning("Voice %r not found among %d presets — using default.", want, len(_voices))
        return None
    try:
        return _tts.get_preset_voice(vid)
    except Exception as e:  # noqa: BLE001
        log.warning("get_preset_voice(%r) failed: %s — using default.", vid, e)
        return None


def _encode(audio: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    subtype, ctype = _FORMATS.get(fmt, _FORMATS["mp3"])
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    buf = io.BytesIO()
    try:
        sf.write(buf, audio, sample_rate, format=subtype)
    except Exception as e:  # noqa: BLE001 — e.g. libsndfile without MP3 → fall back to WAV
        if subtype != "WAV":
            log.warning("Encoding %s failed (%s) — falling back to WAV.", subtype, e)
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV")
            return buf.getvalue(), "audio/wav"
        raise
    return buf.getvalue(), ctype


def _check_auth(authorization: str | None):
    if SHIM_API_KEY:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token != SHIM_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Models ───────────────────────────────────────────────────────────────────
class SpeechRequest(BaseModel):
    model: str | None = None
    input: str = ""
    voice: str | None = None
    response_format: str | None = "mp3"
    speed: float | None = None  # accepted for compatibility; not used


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "model": MODEL_NAME,
            "loaded": _tts is not None, "voices": len(_voices)}


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "vieneu"}]}


@app.get("/v1/audio/voices")
def list_voices(authorization: str | None = Header(default=None)):
    """Voice discovery for OpenAI-compatible clients. `id` is what you pass as `voice`."""
    _check_auth(authorization)
    _load()
    return {"voices": [{"id": vid, "description": desc} for desc, vid in _voices]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` text is required.")
    fmt = (req.response_format or "mp3").strip().lower()
    _load()
    try:
        with _lock:
            voice_data = _resolve_voice(req.voice)
            audio = (_tts.infer(text=text, voice=voice_data) if voice_data is not None
                     else _tts.infer(text=text))
            sr = int(getattr(_tts, "sample_rate", 24000))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    if audio is None or len(np.asarray(audio).reshape(-1)) == 0:
        raise HTTPException(status_code=500, detail="Model returned empty audio.")
    data, ctype = _encode(audio, sr, fmt)
    return Response(content=data, media_type=ctype)


@app.exception_handler(HTTPException)
def _http_exc(_, exc: HTTPException):  # OpenAI-style error envelope
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"message": exc.detail, "code": exc.status_code}})
