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

For the full walkthrough — prerequisites, build, verifying the GPU is used, and exposing it
— see **[Deploy to a GPU server](#deploy-to-a-gpu-server)** below.

### Remote mode

`VIENEU_MODE=remote` offloads token generation to your VieNeu **LMDeploy** server
(`VIENEU_API_BASE`, e.g. `http://gpu-host:23333/v1`) and decodes the codec locally. Note the
codec decode still runs on the shim's CPU, so for top speed prefer the GPU image above.

## Deploy to a GPU server

Run the **GPU image** (`Dockerfile.gpu`) on an NVIDIA host to match `tts.starb.ca` speed —
the SDK auto-switches to the PyTorch CUDA engine. The image installs `vieneu[gpu]` (PyTorch
CUDA 12.8 + transformers/lmdeploy), so it's large and **must be built on the GPU host**.

### 1. Prerequisites on the host

- An NVIDIA GPU with a recent driver — `nvidia-smi` must work on the host.
- Docker Engine + Docker Compose.
- The **NVIDIA Container Toolkit** (lets containers see the GPU):

  ```bash
  # Ubuntu/Debian
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ```

- Verify Docker can see the GPU:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
  ```

### 2. Get the code on the GPU host

```bash
git clone https://github.com/EricNgo1972/VieNeu.git vieneu-tts-shim
cd vieneu-tts-shim
```

### 3. Build & run

```bash
# Compose (recommended) — already requests all GPUs:
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml logs -f      # watch the first model download

# …or plain docker:
docker build -f Dockerfile.gpu -t vieneu-openai-shim:gpu .
docker run -d --name vieneu-tts-shim --gpus all -p 8080:8080 \
  -v vieneu_hf:/data/hf -e VIENEU_MODE=v3turbo --restart unless-stopped \
  vieneu-openai-shim:gpu
```

Pick the model with `VIENEU_MODE` (`v3turbo` default, `fast`/`gpu` for v2, `standard` for v1
— see the [Models table](#models--one-per-instance-chosen-by-vieneu_mode)). The first start
downloads weights into the `hf-cache` volume; later restarts are fast.

### 4. Confirm it's actually on the GPU

```bash
curl -s http://localhost:8080/health     # engine should be the GPU engine, not "onnx"
nvidia-smi                               # the python process should appear on the GPU
# time a synth — expect ~1–2 s, not the ~20 s CPU path
time curl -s http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào từ GPU.","voice":"","response_format":"mp3"}' -o gpu.mp3
```

On a CUDA host v3 Turbo loads the **PyTorch** engine (startup log:
`✅ VieNeu-TTS v3 Turbo ready (backend=pytorch)`) instead of `backend=onnx`.

### 5. Expose it (so Maple Video Studio can reach it)

Studio talks to the shim over HTTP at `:8080`. On a server, put it behind whatever you
already use:

- **Cloudflare tunnel** (same as `tts.starb.ca`): point a tunnel hostname at
  `http://localhost:8080`, and set `SHIM_API_KEY` since it's now public.
- **Reverse proxy** (nginx/Caddy/Traefik): proxy your TTS hostname → `127.0.0.1:8080`.
- **Private network only:** reach it directly at `http://<gpu-host-ip>:8080`.

Then point Studio at that URL — see [Point Maple Video Studio at it](#point-maple-video-studio-at-it).

> **Port tip:** the shim and the Studio container both default to `8080`. On the same host,
> remap one (e.g. `-p 8081:8080` for the shim). Behind separate tunnels/hosts there's no conflict.

## Run (CPU / no GPU)

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

A **manual** workflow (`.github/workflows/build.yml`) builds the images and pushes them to
GitHub Container Registry. It does **not** run on push — trigger it yourself:

1. GitHub → **Actions** → **build-and-push** → **Run workflow** → pick `both`, `gpu`, or `cpu`.
2. It publishes (image name lowercased automatically):
   - `ghcr.io/ericngo1972/vieneu:latest` and `:cpu` — CPU image (v3 Turbo on ONNX)
   - `ghcr.io/ericngo1972/vieneu:gpu` — **GPU image** (PyTorch CUDA; for the GPU server)

GPU build runs on a CPU CI runner (building needs no GPU — only *running* does).

**Deploy the GPU image on the GPU server** (the package is private, so log in first):

```bash
# read:packages PAT — create at github.com/settings/tokens
echo "$GHCR_PAT" | docker login ghcr.io -u EricNgo1972 --password-stdin

docker run -d --name vieneu-tts-shim --gpus all -p 8080:8080 \
  -v vieneu_hf:/data/hf -e VIENEU_MODE=v3turbo --restart unless-stopped \
  ghcr.io/ericngo1972/vieneu:gpu
```

Or pin it in `docker-compose.gpu.yml` (replace the `build:` block with
`image: ghcr.io/ericngo1972/vieneu:gpu`) and run
`docker compose -f docker-compose.gpu.yml up -d`. The CPU image is `…:latest` for non-GPU hosts.

## Configuration (env vars)

| Var                   | Default                        | Notes                                        |
|-----------------------|--------------------------------|----------------------------------------------|
| `VIENEU_MODE`         | `local`                        | model/engine — `local`·`v3turbo`·`fast`·`gpu`·`standard`·`remote` (see Models table) |
| `VIENEU_EMOTION`      | `natural`                      | `natural` or `storytelling`                  |
| `VIENEU_MODEL`        | *(empty)*                      | override model id; mainly for `remote` mode  |
| `VIENEU_API_BASE`     | `http://localhost:23333/v1`    | remote LMDeploy base URL (remote mode)       |
| `VIENEU_DEFAULT_VOICE`| *(empty)*                      | preset id used when a request omits `voice`  |
| `SHIM_API_KEY`        | *(empty)*                      | if set, requires `Authorization: Bearer <key>` |
| `HF_TOKEN`            | *(empty)*                      | only for private/gated HF models             |

> **GPU note:** `local` is CPU/ONNX, torch-free, and serves only v3 Turbo. The GPU-backed
> engines (`v3turbo`/`fast`/`gpu`/`standard`) need the GPU image (`Dockerfile.gpu`) — see
> [Deploy to a GPU server](#deploy-to-a-gpu-server).

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
