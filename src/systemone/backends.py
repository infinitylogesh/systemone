"""The engine-specific calls, one class per serving engine.

systemone needs five things from an engine: render a chat template to token ids,
tokenize / detokenize text, a batched one-step read that returns the logprobs of
chosen token ids, the same read through the chat endpoint (for media), and a plain
generation (for `think`). vLLM and SGLang both provide them, through different routes.
`detect()` picks the backend from what the server answers.
"""

from typing import Dict, List, Optional, Tuple


def detect(up) -> "Backend":
    """SGLang answers /server_info (or the older /get_server_info); vLLM doesn't."""
    for path in ("/server_info", "/get_server_info"):
        try:
            info = up.get(path)
            if isinstance(info, dict):
                return SGLang(up, info.get("version"))
        except Exception:
            pass
    return VLLM(up)


class Backend:
    name = "?"

    def __init__(self, up, version: Optional[str] = None):
        self.up, self.version = up, version

    # each returns token ids / text / {token_id: logprob} rows
    def render(self, model, messages, generation, cont, kwargs) -> List[int]:
        raise NotImplementedError

    def encode(self, model, text) -> List[int]:
        raise NotImplementedError

    def decode(self, model, ids) -> str:
        raise NotImplementedError

    def read(
        self, model, prompts, label_union, topk
    ) -> Tuple[List[Dict[int, float]], Dict]:
        raise NotImplementedError

    def read_chat(
        self, model, messages, label_ids, topk, label_text
    ) -> Tuple[Dict[int, float], Dict]:
        raise NotImplementedError

    def generate(self, model, prompt, max_tokens, stop_ids) -> Tuple[List[int], bool]:
        raise NotImplementedError

    def supports_label_ids(self, model, sample) -> bool:
        return True

    # Sending a model media it can't take must be safe for the probe to try it.
    safe_to_probe_audio = True

    def audio_from_config(self, model) -> Optional[bool]:
        return None


def _hf_config(model_path: str) -> Optional[dict]:
    """A model's config.json: from the local Hugging Face cache, else the Hub. None if unknown."""
    import glob
    import json
    import os
    import urllib.request

    if os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "config.json")
    ):
        return json.load(open(os.path.join(model_path, "config.json")))
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(
        home,
        "hub",
        "models--" + model_path.replace("/", "--"),
        "snapshots",
        "*",
        "config.json",
    )
    for path in glob.glob(pattern):
        return json.load(open(path))
    try:
        url = f"https://huggingface.co/{model_path}/resolve/main/config.json"
        return json.load(urllib.request.urlopen(url, timeout=10))
    except Exception:
        return None


class VLLM(Backend):
    """vLLM's OpenAI server: /tokenize, /detokenize, /v1/completions with token-id prompts,
    `logprob_token_ids` and `return_tokens_as_token_ids`."""

    name = "vllm"

    def __init__(self, up, version=None):
        super().__init__(up, version)
        self.label_ids = True

    def render(self, model, messages, generation, cont, kwargs):
        return self.up.post(
            "/tokenize",
            {
                "model": model,
                "messages": messages,
                "add_generation_prompt": generation,
                "continue_final_message": cont,
                "chat_template_kwargs": kwargs or {},
            },
        )["tokens"]

    def encode(self, model, text):
        return self.up.post(
            "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}
        )["tokens"]

    def decode(self, model, ids):
        return self.up.post("/detokenize", {"model": model, "tokens": list(ids)})[
            "prompt"
        ]

    def supports_label_ids(self, model, sample):
        from .model import UpstreamError

        try:
            self.up.post(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": [sample[:8]],
                    "max_tokens": 1,
                    "logprobs": 1,
                    "logprob_token_ids": [sample[0]],
                },
            )
            self.label_ids = True
        except UpstreamError:
            self.label_ids = False
        return self.label_ids

    def read(self, model, prompts, label_union, topk):
        body = {
            "model": model,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": topk,
            "return_tokens_as_token_ids": True,
        }
        if self.label_ids:
            body["logprob_token_ids"] = label_union[:128]
        d = self.up.post("/v1/completions", body)
        rows = [
            {
                int(k.split(":")[1]): v
                for k, v in c["logprobs"]["top_logprobs"][0].items()
            }
            for c in sorted(d["choices"], key=lambda c: c["index"])
        ]
        return rows, d.get("usage", {})

    def read_chat(self, model, messages, label_ids, topk, label_text):
        body = {
            "model": model,
            "messages": messages,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": topk,
            "return_tokens_as_token_ids": True,
        }
        if self.label_ids:
            body["logprob_token_ids"] = list(label_ids)
        d = self.up.post("/v1/chat/completions", body)
        row = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return {int(t["token"].split(":")[1]): t["logprob"] for t in row}, d.get(
            "usage", {}
        )

    def generate(self, model, prompt, max_tokens, stop_ids):
        d = self.up.post(
            "/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
                "stop_token_ids": stop_ids,
                "skip_special_tokens": False,
            },
        )
        c = d["choices"][0]
        return [int(t.split(":")[1]) for t in c["logprobs"]["tokens"]], c.get(
            "finish_reason"
        ) == "stop"


class SGLang(Backend):
    """SGLang: the same /tokenize and chat routes, and its native /generate, which takes
    batched `input_ids` and returns logprobs for chosen ids (`token_ids_logprob`) as
    (logprob, token_id, text) triples. Its OpenAI chat route returns top logprobs as
    token *text*, so the media path matches labels by their decoded text."""

    name = "sglang"
    # SGLang 0.5.20 crashes (the whole server exits) when a model without an audio encoder
    # gets audio, e.g. Gemma 4 31B; vLLM answers 400. So never probe SGLang with audio.
    safe_to_probe_audio = False

    def audio_from_config(self, model):
        """Whether the checkpoint has an audio encoder, from its config (not from
        /model_info, whose has_audio_understanding reflects the model family's code)."""
        try:
            path = self.up.get("/model_info").get("model_path")
        except Exception:
            return None
        cfg = _hf_config(path) if path else None
        if cfg is None:
            return None
        return bool(cfg.get("audio_config"))

    def render(self, model, messages, generation, cont, kwargs):
        # SGLang has no add_generation_prompt field: a continued final message gets none
        return self.up.post(
            "/tokenize",
            {
                "model": model,
                "messages": messages,
                "continue_final_message": cont,
                "chat_template_kwargs": kwargs or {},
            },
        )["tokens"]

    def encode(self, model, text):
        return self.up.post(
            "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}
        )["tokens"]

    def decode(self, model, ids):
        return self.up.post(
            "/detokenize",
            {"model": model, "tokens": list(ids), "skip_special_tokens": False},
        )["text"]

    @staticmethod
    def _items(d):
        return d if isinstance(d, list) else [d]

    def read(self, model, prompts, label_union, topk):
        d = self.up.post(
            "/generate",
            {
                "input_ids": prompts,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True,
                "top_logprobs_num": topk,
                "token_ids_logprob": [label_union] * len(prompts),
                "logprob_start_len": -1,
            },
        )
        rows, prompt_tokens = [], None
        for item in self._items(d):
            meta = item["meta_info"]
            row = {
                int(t[1]): float(t[0])
                for t in (meta.get("output_top_logprobs") or [[]])[0]
                if t[0] is not None
            }
            for t in (meta.get("output_token_ids_logprobs") or [[]])[0] or []:
                if t[0] is not None:
                    row[int(t[1])] = float(t[0])
            rows.append(row)
            prompt_tokens = prompt_tokens or meta.get("prompt_tokens")
        return rows, {"prompt_tokens": prompt_tokens}

    def read_chat(self, model, messages, label_ids, topk, label_text):
        d = self.up.post(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "continue_final_message": True,
                "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": topk,
            },
        )
        row = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        by_text = {}
        for t in row:
            by_text.setdefault(t["token"], t["logprob"])
        out = {}
        for i, text in zip(
            label_ids, label_text
        ):  # match each label by its text, exactly or ignoring spaces
            lp = by_text.get(text)
            if lp is None:
                lp = next(
                    (v for k, v in by_text.items() if k.strip() == text.strip()), None
                )
            if lp is not None:
                out[i] = lp
        # the other returned tokens keep their text as a key; only their probabilities matter
        for k, v in by_text.items():
            out.setdefault(-(abs(hash(k)) % 10**9) - 1, v)
        return out, d.get("usage", {})

    def generate(self, model, prompt, max_tokens, stop_ids):
        d = self._items(
            self.up.post(
                "/generate",
                {
                    "input_ids": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0,
                        "stop_token_ids": stop_ids,
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                    "top_logprobs_num": 0,
                },
            )
        )[0]
        meta = d["meta_info"]
        ids = [int(t[1]) for t in meta.get("output_token_logprobs") or []]
        reason = meta.get("finish_reason") or {}
        return ids, (
            reason.get("type") if isinstance(reason, dict) else reason
        ) == "stop"
