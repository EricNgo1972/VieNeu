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
| `GET  /`                | **Built-in test page** — load voices, synthesize, play, download |
| `POST /v1/audio/speech` | `{model, input, voice, response_format}` → audio bytes (mp3/wav/flac/ogg) |
| `GET  /v1/audio/voices` | Discover preset voices: `{"voices":[{"id","description"}]}`     |
| `GET  /v1/models`       | OpenAI-style model list                                        |
| `GET  /health`          | Liveness + mode/model/voice count (used by the Docker healthcheck) |

### Test page

Open `http://<shim-host>:8080/` in a browser to exercise the whole pipeline without the
Studio app: it calls **Load voices**, lets you type Vietnamese text, synthesizes, and
plays/downloads the result — so you can confirm the wrapper works (and audition voices)
independently. If you set `SHIM_API_KEY`, paste it into the page's API-key field.

## Models — one per instance, chosen by `VIENEU_MODE`

Each running instance serves **exactly one** model (the OpenAI `model` field in requests
is accepted but ignored). Pick it with `VIENEU_MODE`:

| `VIENEU_MODE`     | Model                         | Needs    | Image            |
|-------------------|-------------------------------|----------|------------------|
| `local` (default) | **v3 Turbo** (CPU via ONNX)   | nothing  | `Dockerfile`     |
| `v3turbo`         | **v3 Turbo** (GPU, PyTorch)   | GPU      | `Dockerfile.gpu` |
| `fast` / `gpu`    | **VieNeu-TTS-v2** (LMDeploy)  | GPU      | `Dockerfile.gpu` |
| `standard`        | **VieNeu-TTS v1**             | GPU      | `Dockerfile.gpu` |
| `remote`          | model on your `:23333` server | server   | either           |

> **Why CPU is slow:** the minimal (torch-free) install only has **v3 Turbo on CPU/ONNX**
> (~tens of seconds/sentence). `tts.starb.ca` runs on a **GPU**, which is far faster. To
> match it, deploy the **GPU image** (`Dockerfile.gpu`) on an NVIDIA host — the SDK
> auto-switches to the PyTorch CUDA engine. To serve **v2/v1** at all, you need the GPU image.

### GPU build (matches tts.starb.ca speed)

On an NVIDIA host with the driver + [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit):

```bash
docker compose -f docker-compose.gpu.yml up -d --build      # VIENEU_MODE=v3turbo by default
# or plain docker:
docker build -f Dockerfile.gpu -t vieneu-openai-shim:gpu .
docker run -d --gpus all -p 8080:8080 -v vieneu_hf:/data/hf \
  -e VIENEU_MODE=v3turbo vieneu-openai-shim:gpu
```

The GPU image installs `vieneu[gpu]` (PyTorch CUDA 12.8 + transformers/lmdeploy), so it's
large and **must be built/tested on the GPU host** (it can't be validated on a CPU-only box).

### Remote mode

`VIENEU_MODE=remote` offloads token generation to your VieNeu **LMDeploy** server
(`VIENEU_API_BASE`, e.g. `http://gpu-host:23333/v1`) and decodes the codec locally. Note the
codec decode still runs on the shim's CPU, so for top speed prefer the GPU image above.

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
docker run -d --name vieneu-tts-shim -p 8080:8080 \
  -v vieneu_hf:/data/hf \
  -e VIENEU_MODE=local \
  vieneu-openai-shim
```

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Build & publish the image (GitHub Actions → GHCR)

A **manual** workflow (`.github/workflows/build.yml`) builds the container and pushes it to
GitHub Container Registry. It does **not** run on push — trigger it yourself:

1. GitHub → **Actions** → **build-and-push** → **Run workflow** (optionally pass an extra tag).
2. It publishes `ghcr.io/ericngo1972/vieneu:latest` (lowercased automatically).

Pull and run it on the host (the package is private, so log in first):

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u EricNgo1972 --password-stdin   # PAT with read:packages
docker run -d --name vieneu-tts-shim -p 8080:8080 \
  -v vieneu_hf:/data/hf -e VIENEU_MODE=local \
  ghcr.io/ericngo1972/vieneu:latest
```

Or pin the image in `docker-compose.yml` (replace `build: .` with
`image: ghcr.io/ericngo1972/vieneu:latest`).

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
curl -s http://localhost:8080/v1/audio/voices | python3 -m json.tool

# synthesize → out.mp3
curl -s http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào, đây là giọng đọc tiếng Việt.","voice":"","response_format":"mp3"}' \
  -o out.mp3 && file out.mp3
```

## Point Maple Video Studio at it

In **Settings → Voiceover (TTS)**:

- **Provider:** OpenAI
- **Endpoint:** `http://<shim-host>:8080`  (the app appends `/v1/audio/speech`)
- **API key:** leave blank, or the `SHIM_API_KEY` value if you set one
- **Model:** any value (the shim is bound to one model) — e.g. `pnnbao-ump/VieNeu-TTS-v2`
- **Default voice:** click **Load voices** to pull the real Vietnamese voice list
  from `GET /v1/audio/voices`, then pick one.
