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

## OpenRouter: hosted models, no GPU

`--openrouter NAME=MODEL` serves a model hosted on [OpenRouter](https://openrouter.ai)
(key in `OPENROUTER_API_KEY`), alone or next to `--upstream` engines:

```bash
systemone serve --openrouter gemma4-openrouter=google/gemma-4-31b-it --calibration-dir calibration
```

### Why the answer-slot read isn't available

The read above needs three things from the engine, and OpenRouter has none of them:

| the answer-slot read needs | OpenRouter |
|---|---|
| a prompt that *ends* at `department:` | providers start a new reply instead of continuing a prefilled assistant message (a prefill of `The customer's dep` gets `churn`, not `artment`) |
| token-id prompts from the model's own template | no `/tokenize`; `/v1/completions` does continue a hand-rendered prompt, but only without logprobs: with logprobs on, OpenRouter wraps the text in the chat template again |
| the logprob of each chosen label id | top 20 logprobs per generated position only (no `logprob_token_ids`) |

So the model has to *write* the answer, and the probabilities are read from the top 20
alternatives at the token where it writes the label.

### The per-question read (default)

One chat request per question, `max_tokens: 1`, all of a stage's questions in parallel:

```
system:  the questions, exactly as for vLLM (same system_text())
user:    {"body": "We were billed twice for March. Refund it today or we cancel."}

         Now answer only question department. Reply with just its label (A / B / C) and nothing else.
reply:   A        top_logprobs: 'A' −0.0, 'department' −8.5, '#' −13.9, 'depart' −14.1, ' A' −14.2, …
```

This is the closest match to the answer-slot read:

- each question is read **independently from the same prompt**, like vLLM's batched read
  in the default `independent` mode;
- the read is **the next token after the prompt**: nothing is generated before it and
  nothing needs parsing.

The only differences are that the prompt ends in an instruction rather than in the
answer line itself, and that a label outside the top 20 gets a floor (the lowest
returned logprob − 5 nats) instead of its exact value. Measured on typed-decisions,
the per-question read matches local vLLM's accuracy, and its fitted temperatures are
close to vLLM's (noul 14.5 vs 14.0), so the probabilities have the same shape.

### The single read (`--openrouter-read single`)

One request per stage: the model writes an `id: label` line per question, and each
question's distribution is the top 20 at the first non-space token after the colon on
its line. The same case, captured live:

```
reply:  department: A\nurgency: 3\nchurn: yes           (14 tokens)
tokens: 'department' ':' ' A' '\n' 'urg' 'ency' ':' ' ' '3' '\n' 'churn' ':' ' yes'
                          ▲ department                   ▲ urgency (after a separate ' ')   ▲ churn
' A'  top: ' A' −0.0, ' aspiration' −18.8, ' billing' −19.2, …   (B and C are not in the top 20)
```

It sends the prompt once per case instead of once per question (≈ 5× cheaper for 5
questions, and 5× fewer requests against rate limits), at a cost in fidelity:

- later questions are read **after the model's own earlier answers** (like `mode: "joint"`),
  so one early mistake can pull the rest;
- the label has to be **found in the reply**: ids split across tokens (`urg` + `ency`),
  a space emitted as its own token, and the model sometimes writes the question's text
  instead of its id (`What is the topic of this news article?: C`). The parser takes a
  question's line by its id, else by its position, and falls back to a per-question
  request for a question it can't find.

### Providers

Only some providers return logprobs (for Gemma 4 31B, 4 of 14 list them). The probe:

1. keeps the providers whose OpenRouter listing includes `logprobs` and `top_logprobs`;
2. sends each a 1-token request and keeps those that actually return them (Venice
   lists them but returns none), fastest first.

Every request sends that list with `allow_fallbacks: false` and
`require_parameters: true`, so OpenRouter never routes elsewhere. A reply without
logprobs, a rate limit or a provider error is retried (4 tries, backing off); a bad key
or exhausted credits is not. `--openrouter-providers` sets the list by hand. The model's
input modalities come from the same listing: images work; video and audio are refused
because they haven't been tested through these providers; `think` isn't available.

### Measured: Gemma 4 31B

typed-decisions test split (2,000 decisions), temperatures fitted on 300 train cases:

| | typed acc | ECE raw → cal | Brier (cal) | requests per 5-question case | cost / 1k cases |
|---|---|---|---|---|---|
| local vLLM, NVFP4 (answer-slot read) | 0.703–0.709 | 0.268 → 0.101 | 0.105 | 1 batched | your GPU |
| **OpenRouter, per_question** | **0.707** | 0.262 → 0.098 | **0.095** | 5 | ≈ $0.30 |
| OpenRouter, single | 0.697 | 0.286 → 0.093 | 0.104 | 1 | ≈ $0.07 |

Per-call latency is ≈ 0.9 s at p50 (per_question) and ≈ 1.2 s (single, which also
decodes the lines), almost all of it the network hop and the provider's queue; the local
answer-slot read takes 81 ms per case. Answers depend on the provider serving the
request (CoreWeave runs fp4, Parasail bf16/fp8). Temperatures:
`calibration/gemma4-openrouter.json` (per_question) and
`calibration/gemma4-openrouter-single.json` (serve the single read under that name).

In code: `src/systemone/openrouter.py`. `engine.decide()` hands a request for an
OpenRouter model to `OpenRouterModel.decide()`; the schema, calibration, response shape,
server and demos are shared with the vLLM / SGLang path.

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