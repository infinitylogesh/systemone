## How it works

Nothing is generated. For each question the prompt ends exactly where the answer
label goes, and **one next-token read** gives the probability of every allowed label:

```
<system: the questions, each with single-token labels>   A: billing  B: technical  C: other
<user: the state>
<assistant turn>department: ▮        ← P(" A"), P(" B"), P(" C") at this position = the answer
```

- **Labels** are single tokens: `yes`/`no` for `noul`, `A`–`Z` for `choice`, `1`–`9` for
  `score`. The server maps them back to your option names and to Jev's response shapes.
  The `noul` answer is P(yes). The `score` answer is the expected 0-based level.
- **All questions in one call.** They go to vLLM as one batched `/v1/completions`
  request. Prefix caching processes the shared prompt once, so each extra question costs
  a few tokens.
- **Nothing model-specific is hard-coded.** At startup the server asks vLLM:

  | question | how |
  |---|---|
  | which model? | `GET /v1/models` |
  | where does the answer start? | render the chat template twice with `/tokenize` (no-thinking generation prompt; a continued assistant message) and take the longer one |
  | does each label tokenize to one token *in place*? | `/tokenize` the answer line with each label |
  | per-label logprobs supported? | a probe `/v1/completions` call (otherwise falls back to the top-20) |
  | thinking format? | which marker is a single special token: `<\|channel>` Gemma, `<\|channel\|>` harmony, `<think>` Qwen/DeepSeek |
  | vision? | compare prompt lengths with and without a 1-pixel image |

  What the probe finds, per family:

  | model | the answer goes right after |
  |---|---|
  | Gemma 4 | `<\|turn>model\n<\|channel>thought\n<channel\|>` (the empty thought block from the template) |
  | Qwen3.5 | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
  | gpt-oss | `<\|start\|>assistant<\|channel\|>final<\|message\|>` (the `final` channel is appended by systemone) |

  The chat template is rendered by vLLM, not by this package. So models whose template
  lives in the engine rather than on Hugging Face need no local template.

## Engines: vLLM and SGLang

`--upstream` can point at either engine, and at a mix of both behind one endpoint. The
engine is detected at startup (SGLang answers `/server_info`; vLLM doesn't) and reported
by `GET /v1/systemone/info`. What systemone needs from each:

| need | vLLM | SGLang |
|---|---|---|
| chat template → token ids | `/tokenize` with `messages` | `/tokenize` with `messages` (no `add_generation_prompt` field: a continued message gets none) |
| tokenize / detokenize text | `/tokenize`, `/detokenize` → `prompt` | `/tokenize`, `/detokenize` → `text` |
| the read: one step, logprobs for chosen token ids | `/v1/completions` + `logprob_token_ids` | native `/generate` + `token_ids_logprob`, batched `input_ids` |
| media read (image / video / audio) | chat + `logprob_token_ids` | chat top logprobs, matched to labels by their text |
| `think` generation | `/v1/completions` + `stop_token_ids` | `/generate` + `stop_token_ids` |

With SGLang:

```bash
python -m sglang.launch_server --model-path google/gemma-4-E4B-it --port 30000
systemone serve --upstream http://127.0.0.1:30000
```

**SGLang caveat: audio.** SGLang 0.5.20 **crashes the whole server** when a model without an
audio encoder (e.g. Gemma 4 31B) is sent audio; vLLM answers 400. Its `/model_info` also
reports `has_audio_understanding: true` for such models. So on SGLang, systemone never
probes with audio. It reads the checkpoint's `config.json` (local cache or the Hub) for an
`audio_config`, treats "unknown" as no audio, and refuses audio for those models with a 422.
Gemma 4 31B NVFP4 (ModelOpt) runs on SGLang on this GPU (ready in ~4.5 min).

If the GPU is shared with other processes, note that SGLang's `--mem-fraction-static` must
cover the model weights *plus* the KV cache. The weights alone needed more than 0.35 here,
so it ran with 0.6.

**Parity check: the same Gemma 4 E4B on both engines**, through systemone with identical
questions:

| | vLLM | SGLang 0.5.20 |
|---|---|---|
| features | 28/31 | 28/31 (the same 3 borderline misses; images, `think` pass) |
| typed-decisions accuracy / ECE after fit | 0.640 / 0.067 | 0.640 / 0.068 |
| AG News / Emotion / SST-2 / SST-2 choice | 0.733 / 0.543 / 0.933 / 0.947 | 0.737 / 0.547 / 0.937 / 0.950 |
| 1 question p50 · 5-question case p50 | **24 ms** · **56 ms** | 54 ms · 65 ms |
| 4 questions at 32 / 64 in flight | 133 / 119 req/s | 129 / **174** req/s |

The answers match to within 0.4 points. vLLM is faster per request. SGLang scaled
further at 64 in flight, but vLLM's E4B was capped at 32 concurrent sequences, and on this
GPU SGLang used its Triton attention kernels. Raw data:
`results/engines-e4b-vllm-vs-sglang-2026-09-21.json`.

## Request options

Standard Jev fields (`state`, `questions` with `type` / `instructions` / `criteria`), plus:

| field | effect |
|---|---|
| `depends_on: [ids]` | the question is read after those, with their answers written into its prompt |
| `ask_if: {id: [answers]}` | asked only if that question's answer is listed, otherwise `null` |
| `mode: "joint"` | read questions in order, each seeing the earlier answers (exact greedy decoding of the answer block); the default `"independent"` reads them all in parallel |
| `think: N` | the model reasons for up to N tokens in its own thinking format, then the answer is read with that reasoning in the prompt |
| `images` / `video` / `audio` | media data URLs; a model that lacks that tower returns 422 |
| `video_frames` / `fps` | JPEG frames packed into an MJPEG clip (browsers need not encode video) |
| `instructions` | extra context added to the system prompt |

**Several models behind one endpoint.** If the upstream serves several models (a base
plus LoRA adapters, say), the request's `model` field picks one; an unknown name such as
Jev's `"jev-latest"` gets the default (`--model`). `--calibration-dir DIR` gives each model
its own `DIR/<served name>.json` temperatures.

`GET /v1/systemone/info` shows what the probe found and which models are selectable. Set `SYSTEMONE_API_KEY` to require
`Authorization: Bearer <key>`.