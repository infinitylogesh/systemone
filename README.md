# systemone

TypeSafe Jev's `/v1/systemone` typed-decision API (`noul` / `choice` / `score`, with
probabilities) on **any LLM served by vLLM or SGLang**, with no model-specific configuration.

## Results

| model | typed acc | ECE (cal) | AG News | ms / case | req/s |
|---|---|---|---|---|---|
| **Gemma 4 31B** | **0.709** | 0.105 | 0.867 | 81 | 62 |
| **Qwen3.5-35B-A3B** | 0.672 | 0.091 | 0.837 | 206¹ | 35 |
| **gpt-oss-20b** | 0.597 | 0.055 | 0.583 | 42 | **189** |
| gpt-oss-20b, `think: 64` | 0.626 | 0.077 | 0.860 | 304 | 40 |
| DiffusionGemma 26B-A4B | 0.666 | 0.089 | 0.817 | 153 | 34 |
| Laya, zero-shot | 0.362 | **0.026** | **0.933** | **32** | – |
| *Laya, fine-tuned (published)* | *0.766* | *0.213* | *0.953* | – | – |
| *TypeSafe Jev 1.13 (published)* | *0.727* | *0.144* | *0.910* | *710* | – |

- **typed acc**: accuracy on typed-decisions (2,000 decisions).
- **ECE (cal)**: calibration error after the per-model temperature fit (lower is better).
- **ms / case**: p50 for a 5-question typed-decisions case.
- **req/s**: 4-question requests per second with 32 in flight.

Jev figures are third-party published, not measured here, so treat them as indicative.
Laya's zero-shot 0.362 matches its own published 0.361 on the same split, which
cross-checks the scoring.

## Quick start

```bash

git clone https://github.com/infinitylogesh/systemone.git
pip install -e ./systemone            # standard library only

# already running `vllm serve <model>`? put systemone in front of it:
# upstream is the url of the vllm / sglang server
systemone serve --upstream http://localhost:8000

# or start both at once:
systemone launch Qwen/Qwen3.5-35B-A3B-FP8 -- --gpu-memory-utilization 0.85
```

### Example Request

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "<model_name>",
  "state": {"body": "We were billed twice for March. Refund it today or we cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "churn": {"type": "noul", "instructions": "Does the customer threaten to leave?"}
  }
}'
```

Tested on **Gemma 4**, **Qwen3.5**, **gpt-oss** : each started with `systemone launch <model>`

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

## Calibration

Raw LLM probabilities are often overconfident: Gemma 4 averages 0.99 confidence while
being right 62% of the time on Emotion. `systemone bench` fits one temperature per
(question type, option count) on labelled data. `systemone serve --calibration file.json`
applies them, and re-reads the file when it changes. An option count without its own fit
uses the same question type's fit at the nearest count. Temperatures fitted on the
typed-decisions train split for every tested model are in [`calibration/`](calibration/).

## Fine-tuning (RLCD)

`systemone rlcd prepare` builds training data with the serving stack's own tokens, and
`systemone rlcd train` fine-tunes the answer-label distribution with LoRA on top of TRL
(`--objective rlcd | proper | ce`). On Gemma 4 31B it lifts typed-decisions accuracy from
0.702 to 0.79 and Brier from 0.111 to 0.041. Plain cross-entropy did as well as the RL
objective. See [RLCD.md](RLCD.md).

## Demos

`systemone serve --demo` also serves `/studio` (camera, motion, audio) and `/arcade`
(maze, snake). Setup, eval scripts and measured numbers: [demo/README.md](demo/README.md).

## Benchmark

```bash
pip install datasets
systemone bench --backend mymodel=http://localhost:8011 --out results.json --calibration-dir cal/
systemone report results.json        # markdown table incl. Jev's published row
```

Suites:
- `features`: 31 request cases covering types, state shapes, 6 languages, `depends_on`,
  `ask_if`, `think`, images and validation.
- `public`: AG News, DAIR Emotion, SST-2.
- `typed`: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
  400 cases and 2,000 decisions, scored the same way as Jev's published numbers.
- `latency`: sequential p50 and a concurrency sweep.

Any endpoint that speaks `/v1/systemone` can be benchmarked; add `,basic` for a backend
that only implements Jev's core contract.



**Takeaways**
- **Zero-shot LLMs through systemone land near Jev on typed-decisions:** Gemma 4 0.709 vs
  0.727. Only the typed-decisions fine-tune of Laya (0.766) is more accurate.
- **With a fitted temperature, every tested LLM has lower ECE than Jev** (0.055–0.105 vs
  0.144). Gemma 4 and Qwen3.5 also beat it on Brier (0.105 and 0.095 vs 0.148) and on
  score MAE.
- **Latency per 5-question case:** gpt-oss, DeepSeek and Gemma 4 take 42–81 ms against
  Jev's published 710 ms, **9–17× faster**. Qwen3.5 takes 206 ms (3.4×, see ¹).
- **Pick by constraint:**
  - Gemma 4 31B for accuracy;
  - Qwen3.5 for calibration out of the box (raw ECE 0.033, no fitting needed);
  - gpt-oss-20b for throughput (189 req/s, 42 ms per case);
  - gpt-oss with `think: 64` when its quality matters more.
- **Reasoning-trained models need a little thinking.** Read with no reasoning, gpt-oss
  guesses: it puts ~50% on the last option regardless of content. `think: 64` lifts AG News
  from 0.583 to 0.860 at ~290 ms. The answer position itself is fine: 99.6% of the
  probability lands on legal labels (`examples/diagnose_slot.py`).

## Credits:

- [Laya](https://github.com/NandhaKishorM/laya) - For opensourcing the zero-shot classification with encoder-only model approach and fine-tuning on typed-decisions.
- [mmastrac](https://github.com/mmastrac) and his vLLM PR [57250](https://github.com/vllm-project/vllm/pull/57250) - Original inspiration for this work to extend systemone api for decoder only models.

## Limitations

- **Labels must be one token.** That limits `choice` to 26 options and `score` to 26
  levels; the server rejects anything else with a clear 422.
- **Question order.** Questions in `independent` mode don't see each other's answers; use
  `depends_on` or `mode: "joint"` where they should.
- **Images** go through the chat endpoint. For Gemma the prompt then lacks the empty
  thought block, a minor difference from the text path.
- **Engine.** vLLM only for now; the probe needs `/tokenize`, `/detokenize` and token-id
  prompts on `/v1/completions`.
- **Not tested on DeepSeek V3.2/V4** (160 GB+, which doesn't fit the test GPU). The probe
  path is the same one used for R1.
