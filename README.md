# VieNeu-TTS → OpenAI-compatible TTS shim

A tiny FastAPI service that gives [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
a real OpenAI-style **`POST /v1/audio/speech`** endpoint (text → MP3), so any
OpenAI-compatible client — like **Maple Video Studio** — can use it over plain HTTP.

## Why this is needed

VieNeu-TTS has **no HTTP "text → audio" endpoint of its own**:

- Its `:23333` server is a plain **LMDeploy LLM server**. It only exposes
  `/v1/chat/completions`, and returns speech **codec tokens** (`<|speech_123|>…`),
  *not* audio. The waveform is reconstructed **client-side** by the `vieneu`
  Python SDK (NeuCodec decode).
- Its Gradio app (`:7860`, e.g. `https://tts.starb.ca/`) only speaks the Gradio
  call protocol — not OpenAI.

So pointing an OpenAI client at either one fails (`404 Not Found` on
`/v1/audio/speech`). This shim wraps the `vieneu` SDK and does the decoding,
exposing a clean OpenAI surface.

## Endpoints

| Method & path           | Purpose                                                        |
|-------------------------|----------------------------------------------------------------|
| `POST /v1/audio/speech` | `{model, input, voice, response_format}` → audio bytes (mp3/wav/flac/ogg) |
| `GET  /v1/audio/voices` | Discover preset voices: `{"voices":[{"id","description"}]}`     |
| `GET  /v1/models`       | OpenAI-style model list                                        |
| `GET  /health`          | Liveness + mode/model/voice count                              |

## Modes

Set with `VIENEU_MODE`:

- **`local`** (default) — runs **v3 Turbo on CPU via ONNX** (torch-free, **no GPU**,
  no dependency on any other server). Fully self-contained. Recommended.
- **`remote`** — offloads token generation to your VieNeu **LMDeploy** server
  (`VIENEU_API_BASE`, e.g. `http://gpu-host:23333/v1`) and decodes the codec
  locally. Use this to drive the **v2 (GPU)** model.

## Run

### Docker Compose (recommended)

```bash
cd vieneu-tts-shim
docker compose up -d --build
# first start downloads the ONNX models into the hf-cache volume (be patient)
docker compose logs -f
```

### Plain Docker

```bash
docker build -t vieneu-openai-shim .
docker run -d --name vieneu-tts-shim -p 8000:8000 \
  -v vieneu_hf:/data/hf \
  -e VIENEU_MODE=local \
  vieneu-openai-shim
```

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Configuration (env vars)

| Var                   | Default                        | Notes                                        |
|-----------------------|--------------------------------|----------------------------------------------|
| `VIENEU_MODE`         | `local`                        | `local` or `remote`                          |
| `VIENEU_EMOTION`      | `natural`                      | `natural` or `storytelling`                  |
| `VIENEU_MODEL`        | `pnnbao-ump/VieNeu-TTS-v2`     | remote model id (remote mode)                |
| `VIENEU_API_BASE`     | `http://localhost:23333/v1`    | remote LMDeploy base URL (remote mode)       |
| `VIENEU_DEFAULT_VOICE`| *(empty)*                      | preset id used when a request omits `voice`  |
| `SHIM_API_KEY`        | *(empty)*                      | if set, requires `Authorization: Bearer <key>` |
| `HF_TOKEN`            | *(empty)*                      | only for private/gated HF models             |

> **GPU note:** `local` mode is CPU/ONNX and torch-free by design. The v1/v2 GPU
> models require VieNeu's PyTorch stack — for those, run VieNeu's own GPU server
> and use this shim in **`remote`** mode (codec decode stays light/CPU).

## Smoke test

```bash
# voices
curl -s http://localhost:8000/v1/audio/voices | python3 -m json.tool

# synthesize → out.mp3
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào, đây là giọng đọc tiếng Việt.","voice":"","response_format":"mp3"}' \
  -o out.mp3 && file out.mp3
```

## Point Maple Video Studio at it

In **Settings → Voiceover (TTS)**:

- **Provider:** OpenAI
- **Endpoint:** `http://<shim-host>:8000`  (the app appends `/v1/audio/speech`)
- **API key:** leave blank, or the `SHIM_API_KEY` value if you set one
- **Model:** any value (the shim is bound to one model) — e.g. `pnnbao-ump/VieNeu-TTS-v2`
- **Default voice:** click **Load voices** to pull the real Vietnamese voice list
  from `GET /v1/audio/voices`, then pick one.
