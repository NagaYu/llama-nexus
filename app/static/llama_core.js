/**
 * llama_core.js — Llama-Nexus inference Web Worker.
 *
 * Runs entirely off the main thread. All model weights are fetched from the
 * Hugging Face Hub and executed in-browser on WebGPU at 4-bit precision via
 * Transformers.js v3 (`@huggingface/transformers` — the v3 rename of
 * `@xenova/transformers`). Nothing is sent to any server.
 *
 * Pipelines (all lazily loaded, cached as singletons):
 *   • Text / audio chat   → Llama-3.2-1B/3B-Instruct  (AutoModelForCausalLM)
 *   • Image understanding → SmolVLM-{256M,500M}-Instruct (AutoModelForVision2Seq)
 *   • Microphone → text   → Whisper-base                (ASR pipeline)
 *
 * Why a routing layer? Meta's Llama 3.2 *text* checkpoints (1B/3B) are the only
 * Llama-family models small enough to run in a browser, but they are text-only.
 * When the user attaches an image we transparently route to a lightweight open
 * VLM, and when they speak we transcribe locally with Whisper first. The 11B
 * Vision model has no WebGPU ONNX build and needs ~6 GB even at 4-bit, so it is
 * intentionally out of scope for the browser target.
 *
 * Message protocol (main thread → worker):
 *   { type: 'load',       payload: { task, modelId, dtype } }
 *   { type: 'transcribe', payload: { modelId, dtype, audio, sampling_rate } }
 *   { type: 'generate',   payload: { task, modelId, dtype, messages, image, options } }
 *   { type: 'interrupt' }
 *
 * Worker → main thread:
 *   { type: 'progress', payload: {...} }   // download / load progress
 *   { type: 'ready',    payload: {...} }   // a pipeline finished loading
 *   { type: 'transcription', payload: { text } }
 *   { type: 'token',    payload: { token } }            // streamed text chunk
 *   { type: 'metrics',  payload: { tps, numTokens, elapsedMs } }
 *   { type: 'complete', payload: { text, tps, numTokens } }
 *   { type: 'vram',     payload: { bytes, perModel } }  // estimated footprint
 *   { type: 'error',    payload: { message } }
 */

import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoProcessor,
  AutoModelForVision2Seq,
  TextStreamer,
  StoppingCriteria,
  StoppingCriteriaList,
  pipeline,
  RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

// --------------------------------------------------------------------------- //
// Global runtime configuration
// --------------------------------------------------------------------------- //

// We only ship remote (Hub) models; never look for a local /models path.
env.allowLocalModels = false;
// Cache weights in the browser's Cache Storage so a reload is instant & offline.
env.useBrowserCache = true;

const SUPPORTED_TASKS = Object.freeze({
  TEXT: "text-generation",
  VISION: "image-text-to-text",
  ASR: "automatic-speech-recognition",
});

// --------------------------------------------------------------------------- //
// Estimated VRAM accounting
// --------------------------------------------------------------------------- //
//
// Browsers do not expose real GPU memory usage to JS. As a faithful proxy we
// sum the on-disk byte size of every weight file each model loads — that figure
// closely tracks the resident WebGPU buffer footprint. Reported as "Est. VRAM".

const vramByModel = new Map(); // modelId -> { seen: Set<file>, bytes: number }

function accountFile(modelId, file, total) {
  if (!total) return;
  let entry = vramByModel.get(modelId);
  if (!entry) {
    entry = { seen: new Set(), bytes: 0 };
    vramByModel.set(modelId, entry);
  }
  if (entry.seen.has(file)) return;
  entry.seen.add(file);
  entry.bytes += total;
  postVram();
}

function postVram() {
  let bytes = 0;
  const perModel = {};
  for (const [id, e] of vramByModel.entries()) {
    bytes += e.bytes;
    perModel[id] = e.bytes;
  }
  self.postMessage({ type: "vram", payload: { bytes, perModel } });
}

// --------------------------------------------------------------------------- //
// Lazy singleton registry
// --------------------------------------------------------------------------- //

class PipelineRegistry {
  // key = `${task}:${modelId}:${dtypeKey}` -> Promise<resources>
  static #cache = new Map();

  static #key(task, modelId, dtype) {
    return `${task}:${modelId}:${JSON.stringify(dtype)}`;
  }

  /** Returns cached resources, or builds them, broadcasting progress as it goes. */
  static get(task, modelId, dtype) {
    const key = this.#key(task, modelId, dtype);
    if (!this.#cache.has(key)) {
      this.#cache.set(key, this.#build(task, modelId, dtype));
    }
    return this.#cache.get(key);
  }

  static #progressCallback(modelId) {
    return (data) => {
      // Surface download/init progress to the dashboard verbatim, then book-keep VRAM.
      self.postMessage({ type: "progress", payload: { modelId, ...data } });
      if (data.status === "progress" || data.status === "done") {
        accountFile(modelId, data.file, data.total);
      }
    };
  }

  static async #build(task, modelId, dtype) {
    const progress_callback = this.#progressCallback(modelId);
    const common = { device: "webgpu", dtype, progress_callback };

    if (task === SUPPORTED_TASKS.TEXT) {
      const tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback });
      const model = await AutoModelForCausalLM.from_pretrained(modelId, common);
      const resources = { kind: "text", tokenizer, model };
      self.postMessage({ type: "ready", payload: { task, modelId } });
      return resources;
    }

    if (task === SUPPORTED_TASKS.VISION) {
      const processor = await AutoProcessor.from_pretrained(modelId, { progress_callback });
      const model = await AutoModelForVision2Seq.from_pretrained(modelId, common);
      const resources = { kind: "vision", processor, model };
      self.postMessage({ type: "ready", payload: { task, modelId } });
      return resources;
    }

    if (task === SUPPORTED_TASKS.ASR) {
      // Whisper's encoder is sensitive to aggressive quantization; keep it in
      // fp16/fp32 and only 4-bit the (much larger) decoder for the best quality
      // per byte. Callers may override via the dtype payload.
      const asrDtype =
        typeof dtype === "string"
          ? { encoder_model: "fp32", decoder_model_merged: dtype }
          : dtype;
      const transcriber = await pipeline(task, modelId, {
        device: "webgpu",
        dtype: asrDtype,
        progress_callback,
      });
      const resources = { kind: "asr", transcriber };
      self.postMessage({ type: "ready", payload: { task, modelId } });
      return resources;
    }

    throw new Error(`Unsupported task: ${task}`);
  }
}

// --------------------------------------------------------------------------- //
// Interruptible generation
// --------------------------------------------------------------------------- //

let interrupted = false;

class InterruptCriteria extends StoppingCriteria {
  _call() {
    // StoppingCriteria expects one boolean per sequence in the batch (batch = 1).
    return [interrupted];
  }
}

/**
 * Shared streaming driver for both text and vision generation.
 * Counts tokens for an accurate, real-time tokens/sec readout.
 */
async function streamGenerate({ tokenizer, model, inputs, options }) {
  interrupted = false;

  let numTokens = 0;
  let startTime = 0;
  let fullText = "";

  const stoppingCriteria = new StoppingCriteriaList();
  stoppingCriteria.push(new InterruptCriteria());

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    // Fires per decoded text chunk → stream to UI.
    callback_function: (text) => {
      fullText += text;
      self.postMessage({ type: "token", payload: { token: text } });
    },
    // Fires per generated token id → exact count + live throughput.
    token_callback_function: () => {
      if (startTime === 0) startTime = performance.now(); // first token = TTFT boundary
      numTokens += 1;
      const elapsedMs = performance.now() - startTime;
      if (elapsedMs > 0) {
        const tps = (numTokens / elapsedMs) * 1000;
        self.postMessage({
          type: "metrics",
          payload: { tps, numTokens, elapsedMs },
        });
      }
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: options?.max_new_tokens ?? 512,
    do_sample: options?.do_sample ?? false,
    temperature: options?.temperature ?? 1.0,
    top_p: options?.top_p ?? 1.0,
    repetition_penalty: options?.repetition_penalty ?? 1.1,
    streamer,
    stopping_criteria: stoppingCriteria,
  });

  const elapsedMs = startTime ? performance.now() - startTime : 0;
  const tps = elapsedMs > 0 ? (numTokens / elapsedMs) * 1000 : 0;
  self.postMessage({
    type: "complete",
    payload: { text: fullText, tps, numTokens },
  });
}

// --------------------------------------------------------------------------- //
// Task handlers
// --------------------------------------------------------------------------- //

async function handleTranscribe({ modelId, dtype, audio, sampling_rate }) {
  const { transcriber } = await PipelineRegistry.get(SUPPORTED_TASKS.ASR, modelId, dtype);
  // `audio` arrives as a transferred Float32Array (mono, already resampled).
  const output = await transcriber(audio, {
    sampling_rate: sampling_rate ?? 16000,
    chunk_length_s: 30,
    stride_length_s: 5,
    return_timestamps: false,
  });
  const text = (output?.text ?? "").trim();
  self.postMessage({ type: "transcription", payload: { text } });
}

async function handleTextGenerate({ modelId, dtype, messages, options }) {
  const { tokenizer, model } = await PipelineRegistry.get(
    SUPPORTED_TASKS.TEXT,
    modelId,
    dtype,
  );
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });
  await streamGenerate({ tokenizer, model, inputs, options });
}

async function handleVisionGenerate({ modelId, dtype, messages, image, options }) {
  const { processor, model } = await PipelineRegistry.get(
    SUPPORTED_TASKS.VISION,
    modelId,
    dtype,
  );

  // Decode the transferred image bytes into a RawImage the processor understands.
  const rawImage = await RawImage.fromBlob(
    new Blob([image.buffer], { type: image.mimeType || "image/png" }),
  );

  // Inject an explicit image placeholder into the latest user turn so the
  // processor knows where the visual tokens belong.
  const chat = messages.map((m) => ({ ...m }));
  const lastUser = [...chat].reverse().find((m) => m.role === "user");
  if (lastUser) {
    lastUser.content = [
      { type: "image" },
      { type: "text", text: typeof lastUser.content === "string" ? lastUser.content : "" },
    ];
  }

  const prompt = processor.apply_chat_template(chat, { add_generation_prompt: true });
  const inputs = await processor(prompt, [rawImage]);

  await streamGenerate({ tokenizer: processor.tokenizer, model, inputs, options });
}

// --------------------------------------------------------------------------- //
// Message dispatch
// --------------------------------------------------------------------------- //

self.addEventListener("message", async (event) => {
  const { type, payload } = event.data || {};

  try {
    switch (type) {
      case "load": {
        await PipelineRegistry.get(payload.task, payload.modelId, payload.dtype);
        break;
      }
      case "transcribe": {
        await handleTranscribe(payload);
        break;
      }
      case "generate": {
        if (payload.task === SUPPORTED_TASKS.VISION || payload.image) {
          await handleVisionGenerate({ ...payload, task: SUPPORTED_TASKS.VISION });
        } else {
          await handleTextGenerate(payload);
        }
        break;
      }
      case "interrupt": {
        interrupted = true;
        break;
      }
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      payload: { message: err?.message ?? String(err), stack: err?.stack },
    });
  }
});

// Announce liveness so the main thread can verify the worker booted.
self.postMessage({ type: "progress", payload: { status: "worker-ready" } });
