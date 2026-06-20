<div align="center">

# 🦙 Llama-Nexus

### Self-hosted, multimodal AI inference for Meta's open models — running 100% in your browser on WebGPU.

**No API keys · No cloud GPU · No data egress · Server cost: `$0.00`**

`Llama 3.2` &nbsp;·&nbsp; `WebGPU` &nbsp;·&nbsp; `4-bit (q4f16)` &nbsp;·&nbsp; `Transformers.js v3` &nbsp;·&nbsp; `FastAPI`

</div>

---

Llama-Nexus is a premium, research-lab-grade frontend that runs Meta's open
**Llama 3.2** family — plus companion open models for vision and speech —
entirely **client-side on WebGPU**, quantized to **4-bit**. You upload an image,
hold the mic, or type; the model streams an answer back in real time. The Python
backend does **zero** inference — it only serves the app and keeps a local
SQLite log of your conversations and prompt templates.

This is the embodiment of Meta's open-access spirit: take open weights, run them
on hardware you already own, and pay nobody for the privilege.

## ✨ Features

- **🧠 100% local inference** — weights are fetched once from the Hugging Face
  Hub, cached in the browser, then executed on your GPU via WebGPU. The server
  never sees a token.
- **🖼️🎙️📝 Multimodal** — image understanding, microphone → text (local
  Whisper), and text chat, fused into one context.
- **⚡ 4-bit quantized** — `q4f16` by default to keep VRAM tiny and throughput high.
- **📊 Live dashboard** — real-time model download progress, **estimated VRAM**,
  **tokens/sec**, token count, and latency.
- **💾 Local persistence** — conversation logs + reusable prompt templates in SQLite.
- **🐳 One-command self-host** — tiny Docker image, no GPU required on the server.

## 🧩 The model matrix

Meta's **Llama 3.2 Vision (11B)** is intentionally **out of scope for the browser
target**: it has no WebGPU ONNX build and needs ~6 GB even at 4-bit. Instead,
Nexus runs the Llama-family models that *do* fit a browser, and transparently
routes the other modalities to lightweight open models:

| Modality            | Default model                              | Runtime                         |
| ------------------- | ------------------------------------------ | ------------------------------- |
| 💬 Text / audio chat | `onnx-community/Llama-3.2-1B-Instruct`     | `AutoModelForCausalLM`, WebGPU  |
| 💬 Heavier chat      | `onnx-community/Llama-3.2-3B-Instruct`     | `AutoModelForCausalLM`, WebGPU  |
| 🖼️ Image → text      | `HuggingFaceTB/SmolVLM-256M-Instruct`      | `AutoModelForVision2Seq`, WebGPU|
| 🎙️ Speech → text     | `onnx-community/whisper-base`              | ASR pipeline, WebGPU            |

The router is simple: **attach an image → the vision model handles the turn;
otherwise the Llama text model does.** Speech is always transcribed locally with
Whisper first, then folded into the text prompt. Every model ID is a one-line
swap in `app/templates/index.html` (the `<select>` options) — point them at any
ONNX-exported model on the Hub.

> ℹ️ **Library note:** Transformers.js v3 — what powers the WebGPU path — ships
> on npm as **`@huggingface/transformers`** (the v3 rename of
> `@xenova/transformers`). Nexus loads it from a CDN; there is no `npm install`.

## 🚀 Quickstart

### Requirements
- A **WebGPU browser**: Chrome / Edge **121+**, or another browser with WebGPU enabled.
- Python **3.10+** (for local run) **or** Docker.
- A GPU with enough VRAM for your chosen model (≈**1–2 GB** for Llama-3.2-1B @ `q4f16`).

### Option A — Docker (recommended)

```bash
docker build -t llama-nexus .
docker run --rm -p 8000:8000 -v "$(pwd)/data:/data" llama-nexus
```

Open **http://localhost:8000**, click **⚡ Preload chat model**, and start chatting.
The first load downloads the weights (cached for every subsequent visit).

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
# or: python -m app.main
```

Then open **http://localhost:8000**.

> **Why the COOP/COEP headers?** WebGPU + multithreaded WASM fallbacks require the
> page to be *cross-origin isolated*. The FastAPI app sets
> `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` on every response
> automatically — so serving through this backend "just works." If you put a
> reverse proxy in front, preserve those headers.

## 🏗️ Architecture

```
┌──────────────────────── Browser (your GPU) ─────────────────────────┐
│                                                                      │
│  index.html (Tailwind dashboard)                                     │
│       │  postMessage()                                               │
│       ▼                                                              │
│  llama_core.js  ── Web Worker ──►  Transformers.js v3 / WebGPU       │
│       • Llama-3.2 (text)            • 4-bit q4f16                     │
│       • SmolVLM (image)             • streaming + tok/s + VRAM est.  │
│       • Whisper (speech→text)                                        │
│                                                                      │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  fetch()  (logs & templates only)
                                 ▼
                    ┌──────────────────────────┐
                    │  FastAPI (app/main.py)    │   ← no inference here
                    │  • serves UI + worker     │
                    │  • COOP/COEP headers      │
                    │  • SQLite: chats + prompts│
                    └──────────────────────────┘
```

### Project layout

```
meta-llama-nexus/
├── app/
│   ├── main.py                 # FastAPI control plane + SQLite persistence
│   ├── static/
│   │   └── llama_core.js       # WebGPU inference Web Worker (the engine)
│   ├── templates/
│   │   └── index.html          # Tailwind research-lab dashboard + controller
│   └── data/                   # SQLite db (created at runtime)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔌 Backend API

The control plane is a thin REST layer (all data stays on your machine):

| Method   | Endpoint                                | Purpose                          |
| -------- | --------------------------------------- | -------------------------------- |
| `GET`    | `/`                                     | The dashboard                    |
| `GET`    | `/healthz`                              | Liveness probe                   |
| `GET`    | `/api/conversations`                    | List sessions                    |
| `POST`   | `/api/conversations`                    | Create a session                 |
| `GET`    | `/api/conversations/{id}`               | Fetch a session + messages       |
| `POST`   | `/api/conversations/{id}/messages`      | Append a message                 |
| `DELETE` | `/api/conversations/{id}`               | Delete a session                 |
| `GET`    | `/api/templates`                        | List prompt templates            |
| `POST`   | `/api/templates`                        | Save a prompt template           |
| `DELETE` | `/api/templates/{id}`                   | Delete a template                |

## ⚙️ Configuration

| Env var          | Default               | Description                          |
| ---------------- | --------------------- | ------------------------------------ |
| `NEXUS_HOST`     | `127.0.0.1`           | Bind address                         |
| `NEXUS_PORT`     | `8000`                | Port                                 |
| `NEXUS_DB_PATH`  | `app/data/nexus.db`   | SQLite location                      |
| `NEXUS_RELOAD`   | _(unset)_             | Set to enable uvicorn auto-reload    |

## ❓ Troubleshooting

- **"WebGPU unavailable" banner** — your browser/GPU doesn't expose
  `navigator.gpu`. Update to Chrome/Edge 121+ and ensure hardware acceleration is on.
- **First answer is slow** — the very first request downloads weights and compiles
  WebGPU shaders. Subsequent runs are fast (weights are cached). Use **⚡ Preload**.
- **Out of memory** — switch the dtype to `q4`, pick the 1B model, or the
  `SmolVLM-256M` vision model.

## 📜 License & open-access ethos

Llama-Nexus is released under the **MIT License** — fork it, ship it, sell on top
of it. The Llama models themselves are governed by Meta's **Llama 3.2 Community
License**; review it before commercial deployment. Built in gratitude to Meta AI,
Hugging Face, and the ONNX Runtime team for making truly open, local AI possible.

<div align="center">
<sub>Run open models. Own your stack. Pay nobody. — <b>Llama-Nexus</b></sub>
</div>
