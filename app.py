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

For subtitling there is a second endpoint:

    POST /v1/audio/speech_srt   {…same…} -> JSON {audio (base64), srt, cues}

It synthesizes sentence-by-sentence, measures each clip, and returns the audio plus
an SRT timeline (and raw cues) in one call. VieNeu returns audio only — no built-in
timestamps — so timing is derived from per-segment durations (sentence-level, not
word-level).

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
import re
import base64
import logging
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vieneu-shim")

# ── Config (all via env) ────────────────────────────────────────────────────
# VIENEU_MODE selects which model this instance serves (one model per instance).
#   local            → v3turbo  (v3 Turbo; CPU=ONNX torch-free, GPU=PyTorch auto)  [minimal install]
#   remote           → offload token-gen to a VieNeu :23333 LMDeploy server
# These need `pip install vieneu[gpu]` (the GPU image): they map straight to the SDK mode —
#   fast | gpu       → VieNeu-TTS-v2 (GPU, LMDeploy)
#   turbo | turbo_gpu→ v3 Turbo (PyTorch)
#   standard         → VieNeu-TTS v1
#   xpu              → Intel GPU
_RAW_MODE     = os.getenv("VIENEU_MODE", "local").strip().lower()
SDK_MODE      = {"": "v3turbo", "local": "v3turbo"}.get(_RAW_MODE, _RAW_MODE)
IS_REMOTE     = SDK_MODE in ("remote", "api")
MODEL_NAME    = os.getenv("VIENEU_MODEL", "").strip()
_MODEL_LABELS = {
    "v3turbo": "VieNeu-TTS-v3-Turbo", "turbo": "VieNeu-TTS-v3-Turbo",
    "turbo_gpu": "VieNeu-TTS-v3-Turbo (GPU)", "fast": "VieNeu-TTS-v2 (GPU)",
    "gpu": "VieNeu-TTS-v2 (GPU)", "standard": "VieNeu-TTS-v1", "xpu": "VieNeu-TTS-v3-Turbo (XPU)",
}
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

def _active_model() -> str:
    """The model this instance actually serves (one per instance), per VIENEU_MODE.
    remote = whatever runs on the :23333 server (VIENEU_MODEL)."""
    if IS_REMOTE:
        return MODEL_NAME or "pnnbao-ump/VieNeu-TTS-v2"
    return MODEL_NAME or _MODEL_LABELS.get(SDK_MODE, SDK_MODE)


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
        log.info("Loading VieNeu (sdk_mode=%s, model=%s, emotion=%s)…", SDK_MODE, _active_model(), EMOTION)
        kwargs = dict(emotion=EMOTION, hf_token=HF_TOKEN)
        if IS_REMOTE:
            kwargs.update(api_base=API_BASE, model_name=MODEL_NAME or "pnnbao-ump/VieNeu-TTS-v2")
        _tts = Vieneu(mode=SDK_MODE, **kwargs)
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


# ── Subtitle (SRT) helpers ───────────────────────────────────────────────────
def _fmt_ts(sec: float) -> str:
    """Seconds → SRT timestamp 'HH:MM:SS,mmm'."""
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segment_text(text: str, max_chars: int = 200) -> list[str]:
    """Split into subtitle-sized cues: by sentence, then by clause if still too long."""
    raw = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    segs: list[str] = []
    for s in raw:
        if len(s) <= max_chars:
            segs.append(s)
            continue
        cur = ""
        for part in re.split(r"(?<=[,;:])\s+", s):
            if len(cur) + len(part) + 1 <= max_chars:
                cur = f"{cur} {part}".strip()
            else:
                if cur:
                    segs.append(cur)
                cur = part
        if cur:
            segs.append(cur)
    return segs


def build_srt(cues: list[dict]) -> str:
    """cues: [{'text', 'start', 'end'}, …] (seconds) → SRT document."""
    blocks = [
        f"{i}\n{_fmt_ts(c['start'])} --> {_fmt_ts(c['end'])}\n{c['text'].strip()}"
        for i, c in enumerate(cues, 1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _synth_with_cues(text: str, voice_data, gap: float = 0.15):
    """Synthesize each cue separately so SRT timings match the merged audio exactly.
    Returns (audio float32 1-D, sample_rate, cues). Caller must hold `_lock`."""
    sr = int(getattr(_tts, "sample_rate", 24000))
    silence = np.zeros(int(sr * gap), dtype=np.float32)
    chunks: list[np.ndarray] = []
    cues: list[dict] = []
    t = 0.0
    for seg in _segment_text(text):
        a = (_tts.infer(text=seg, voice=voice_data) if voice_data is not None
             else _tts.infer(text=seg))
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == 0:
            continue
        dur = a.size / sr
        cues.append({"text": seg, "start": round(t, 3), "end": round(t + dur, 3)})
        chunks.append(a)
        chunks.append(silence)
        t += dur + gap
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, sr, cues


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
@app.get("/health")
def health():
    return {"status": "ok", "mode": _RAW_MODE, "engine": SDK_MODE, "model": _active_model(),
            "loaded": _tts is not None, "voices": len(_voices)}


@app.get("/", response_class=HTMLResponse)
def test_ui():
    """Self-contained test page: load voices, synthesize, play, download.
    Lets you verify the full pipeline without the Studio app."""
    return _TEST_HTML


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {"object": "list", "data": [{"id": _active_model(), "object": "model", "owned_by": "vieneu"}]}


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


@app.post("/v1/audio/speech_srt")
def speech_srt(req: SpeechRequest, authorization: str | None = Header(default=None)):
    """Synthesize + return audio and a matching SRT in one JSON response.

    Same request body as /v1/audio/speech. Timing is sentence-level (VieNeu has no
    word-level timestamps): the text is split into cues, each is synthesized
    separately, and start/end come from the real per-clip durations.

    Response: {model, format, sample_rate, duration, audio (base64), srt, cues}
    """
    _check_auth(authorization)
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` text is required.")
    fmt = (req.response_format or "mp3").strip().lower()
    _load()
    try:
        with _lock:
            voice_data = _resolve_voice(req.voice)
            audio, sr, cues = _synth_with_cues(text, voice_data)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    if audio.size == 0:
        raise HTTPException(status_code=500, detail="Model returned empty audio.")
    data, ctype = _encode(audio, sr, fmt)
    return JSONResponse({
        "model": _active_model(),
        "format": "wav" if ctype == "audio/wav" else fmt,  # reflects _encode fallback
        "sample_rate": sr,
        "duration": round(audio.size / sr, 3),
        "audio": base64.b64encode(data).decode(),
        "srt": build_srt(cues),
        "cues": cues,
    })


@app.exception_handler(HTTPException)
def _http_exc(_, exc: HTTPException):  # OpenAI-style error envelope
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"message": exc.detail, "code": exc.status_code}})


_TEST_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VieNeu-TTS shim · test</title>
<style>
  :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  body { max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #1f2330; }
  h1 { font-size: 1.25rem; margin: 0 0 .25rem; }
  .muted { color: #6b7280; font-size: .85rem; }
  label { display: block; font-weight: 600; font-size: .85rem; margin: 1rem 0 .35rem; }
  textarea, select, input { width: 100%; box-sizing: border-box; padding: .55rem .6rem;
    border: 1px solid #cdd2dc; border-radius: 8px; font-size: .95rem; }
  textarea { min-height: 110px; resize: vertical; }
  .row { display: flex; gap: .75rem; } .row > * { flex: 1; }
  .btns { display: flex; gap: .6rem; margin-top: 1rem; align-items: center; }
  button { padding: .55rem 1rem; border: 0; border-radius: 8px; font-weight: 600;
    cursor: pointer; background: #3b5bdb; color: #fff; }
  button.secondary { background: #e9ecf3; color: #1f2330; }
  button:disabled { opacity: .5; cursor: default; }
  #status { margin-top: 1rem; font-size: .9rem; white-space: pre-wrap; }
  .err { color: #c92a2a; } .ok { color: #2b8a3e; }
  audio { width: 100%; margin-top: 1rem; }
  .pill { display:inline-block; background:#e7f5ff; color:#1971c2; border-radius:999px;
    padding:.1rem .5rem; font-size:.75rem; margin-left:.4rem; }
</style></head>
<body>
  <h1>VieNeu-TTS shim <span id="badge" class="pill">…</span></h1>
  <div class="muted">Self-contained test page — exercises <code>POST /v1/audio/speech</code> directly.</div>

  <label>API key <span class="muted">(only if SHIM_API_KEY is set)</span></label>
  <input id="key" type="password" placeholder="leave blank if none" autocomplete="off">

  <label>Voice</label>
  <div class="row">
    <select id="voice"><option value="">— model default —</option></select>
    <button class="secondary" style="flex:0 0 auto" id="loadBtn" onclick="loadVoices()">Load voices</button>
  </div>

  <label>Text</label>
  <textarea id="text">Xin chào, đây là giọng đọc tiếng Việt được tổng hợp qua VieNeu-TTS.</textarea>

  <div class="row" style="margin-top:.5rem">
    <div><label style="margin-top:0">Format</label>
      <select id="fmt"><option>mp3</option><option>wav</option></select></div>
    <div></div>
  </div>

  <div class="btns">
    <button id="goBtn" onclick="synth()">Synthesize ▶</button>
    <a id="dl" style="display:none"></a>
  </div>

  <div id="status"></div>
  <audio id="player" controls style="display:none"></audio>

<script>
function authHeaders(extra) {
  const h = extra || {};
  const k = document.getElementById('key').value.trim();
  if (k) h['Authorization'] = 'Bearer ' + k;
  return h;
}
function setStatus(msg, cls) {
  const s = document.getElementById('status');
  s.textContent = msg; s.className = cls || '';
}
async function health() {
  try {
    const r = await fetch('health'); const j = await r.json();
    document.getElementById('badge').textContent =
      j.model + ' · ' + j.mode + (j.loaded ? ' · loaded' : ' · not loaded') + ' · ' + j.voices + ' voices';
  } catch { document.getElementById('badge').textContent = 'offline'; }
}
async function loadVoices() {
  const btn = document.getElementById('loadBtn'); btn.disabled = true;
  setStatus('Loading voices…');
  try {
    const r = await fetch('v1/audio/voices', { headers: authHeaders() });
    if (!r.ok) throw new Error((await r.json()).error?.message || r.status);
    const list = (await r.json()).voices || [];
    const sel = document.getElementById('voice');
    sel.length = 1; // keep the default option
    for (const v of list) {
      const o = document.createElement('option');
      o.value = v.id; o.textContent = v.description ? (v.id + ' — ' + v.description) : v.id;
      sel.appendChild(o);
    }
    setStatus('Loaded ' + list.length + ' voices.', 'ok');
  } catch (e) { setStatus('Could not load voices: ' + e.message, 'err'); }
  finally { btn.disabled = false; }
}
async function synth() {
  const go = document.getElementById('goBtn'); go.disabled = true;
  const player = document.getElementById('player'); player.style.display = 'none';
  const dl = document.getElementById('dl'); dl.style.display = 'none';
  const fmt = document.getElementById('fmt').value;
  setStatus('Synthesizing… (first call also loads the model — can take a while)');
  const t0 = performance.now();
  try {
    const r = await fetch('v1/audio/speech', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        input: document.getElementById('text').value,
        voice: document.getElementById('voice').value,
        response_format: fmt,
      }),
    });
    if (!r.ok) {
      let m = r.status; try { m = (await r.json()).error?.message || m; } catch {}
      throw new Error(m);
    }
    const blob = await r.blob();
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const url = URL.createObjectURL(blob);
    player.src = url; player.style.display = 'block';
    dl.href = url; dl.download = 'tts.' + fmt; dl.textContent = '⬇ download';
    dl.style.display = 'inline'; dl.className = 'muted';
    setStatus('Done in ' + secs + 's · ' + Math.round(blob.size/1024) + ' KB · ' + blob.type, 'ok');
    player.play().catch(()=>{});
  } catch (e) { setStatus('Failed: ' + e.message, 'err'); }
  finally { go.disabled = false; }
}
health();
</script>
</body></html>"""

