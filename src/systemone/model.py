"""Everything model-specific, learned from the serving engine instead of hard-coded.

At startup `Model.probe()` asks the upstream vLLM server:
  * which model it serves                      GET  /v1/models
  * where an answer starts in its chat template POST /tokenize (messages)
  * how its answer labels tokenize             POST /tokenize (text)
  * whether it returns logprobs for chosen ids POST /v1/completions
and picks a thinking profile from the special tokens the tokenizer has. No
tokenizer or chat template is loaded here, so any model the server can chat
with works, including ones whose template lives in the engine (DeepSeek V3.2+).
"""

import http.client
import json
import math
import threading
import urllib.parse
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .media import PROBE_JPEG, frames_to_video_url, wav_url
from .schema import SchemaError

THINK_OFF = {"enable_thinking": False, "thinking": False}
THINK_ON = {"enable_thinking": True, "thinking": True}
TOPK = 20


class UpstreamError(RuntimeError):
    pass


class Upstream:
    """A minimal keep-alive JSON client (one connection per thread)."""

    def __init__(
        self, base_url: str, api_key: Optional[str] = None, timeout: float = 600
    ):
        u = urllib.parse.urlparse(base_url.rstrip("/"))
        self.scheme, self.host, self.port = u.scheme or "http", u.hostname, u.port
        self.prefix = u.path
        self.headers = {"content-type": "application/json"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.local = threading.local()

    def _conn(self):
        c = getattr(self.local, "conn", None)
        if c is None:
            cls = (
                http.client.HTTPSConnection
                if self.scheme == "https"
                else http.client.HTTPConnection
            )
            c = self.local.conn = cls(self.host, self.port, timeout=self.timeout)
        return c

    def request(self, method: str, path: str, body=None):
        
        data = None if body is None else json.dumps(body).encode()
        
        for attempt in (0, 1):
            c = self._conn()
            try:
                c.request(method, self.prefix + path, data, self.headers)
                r = c.getresponse()
                raw = r.read()
                break
            except (http.client.HTTPException, OSError):
                c.close()
                self.local.conn = None
                if attempt:
                    raise
        out = json.loads(raw or b"{}")
        
        if r.status >= 400:
            msg = out.get("error", out) if isinstance(out, dict) else out
            if isinstance(msg, dict):
                msg = msg.get("message", msg)
            raise UpstreamError(f"upstream {r.status} on {path}: {msg}")
        return out

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body):
        return self.request("POST", path, body)


# Thinking formats: how a thought opens and closes, and what comes after it.
PROFILES = {
    # Gemma 4: <|channel>thought\n ... <channel|>, answer right after
    "gemma": {
        "probe": "<|channel>",
        "open": "<|channel>thought\n",
        "close": "<channel|>",
        "after": "",
    },
    # gpt-oss (harmony): reasoning in the analysis channel, answer in the final one
    "harmony": {
        "probe": "<|channel|>",
        "open": "<|channel|>analysis<|message|>",
        "close": "<|end|>",
        "after": "<|start|>assistant<|channel|>final<|message|>",
    },
    # Qwen3 / Qwen3.5 / DeepSeek R1 and V3.x: <think> ... </think>
    "think_tags": {
        "probe": "<think>",
        "open": "<think>\n",
        "close": "\n</think>\n\n",
        "after": "",
    },
}


def _common_prefix(a: List[int], b: List[int]) -> List[int]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


class Model:
    def __init__(self, upstream: Upstream, model: Optional[str] = None):
        from .backends import detect

        self.up = upstream
        self.be = detect(upstream)  # vLLM or SGLang, from what the server answers
        self.model = model
        self.root = model
        self.answer_suffix: List[int] = []
        self.profile: Optional[str] = None
        self.label_ids_supported = True
        self.vision = False
        self.video = False
        self.audio = False
        self.info: Dict = {}

    # ------------------------------------------------------------------ tokenizing
    def render(self, messages, generation=True, cont=False, kwargs=None) -> List[int]:
        return self.be.render(self.model, messages, generation, cont, kwargs)

    @lru_cache(maxsize=65536)
    def encode(self, text: str) -> Tuple[int, ...]:
        return tuple(self.be.encode(self.model, text))

    @lru_cache(maxsize=65536)
    def decode(self, ids: Tuple[int, ...]) -> str:
        return self.be.decode(self.model, ids)

    # ------------------------------------------------------------------ probing
    def probe(self) -> Dict:
        models = self.up.get("/v1/models")["data"]
        if not models:
            raise UpstreamError("upstream serves no model")
        m = (
            next((m for m in models if m["id"] == self.model), None)
            if self.model
            else models[0]
        )
        if m is None:
            raise UpstreamError(
                f"upstream does not serve {self.model!r}; it has {[x['id'] for x in models]}"
            )
        self.model, self.root = m["id"], m.get("root") or m["id"]

        # Thinking profile: the first whose marker is a single special token.
        for name, p in PROFILES.items():
            if len(self.encode(p["probe"])) == 1:
                self.profile = name
                break

        # Where an answer starts. Render the no-thinking generation prompt, and the
        # template's own start of an assistant turn (what precedes two different
        # contents); the longer one, when one extends the other, is the answer start.
        # Gemma adds its empty thought block only to the generation prompt; gpt-oss
        # adds the final channel only to a continued assistant message.
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        gen = self.render(msgs, kwargs=THINK_OFF)
        c1 = self.render(
            msgs + [{"role": "assistant", "content": "Xq"}], False, True, THINK_OFF
        )
        c2 = self.render(
            msgs + [{"role": "assistant", "content": "7z"}], False, True, THINK_OFF
        )
        cont = _common_prefix(c1, c2)
        if cont[: len(gen)] == gen:
            suffix = cont[len(gen) :]
        elif gen[: len(cont)] == cont:
            suffix = []
        else:
            suffix = []  # the two renders disagree; trust the generation prompt
        start = gen + suffix
        # A template that opens a thought on its own (DeepSeek R1) leaves it open: close it.
        if self.profile:
            p = PROFILES[self.profile]
            op = list(self.encode(p["open"]))
            if op and start[-len(op) :] == op:
                suffix += list(self.encode(p["close"]))
        self.answer_suffix = suffix

        # Does the server return logprobs for requested token ids?
        self.label_ids_supported = self.be.supports_label_ids(self.model, gen)

        # Does the model see images? A text-only model's server may drop image parts
        # silently, so compare prompt lengths with and without a tiny image.
        self.vision = self._probe_vision()
        self.video = self.vision and self._probe_part(
            {
                "type": "video_url",
                "video_url": {"url": frames_to_video_url([PROBE_JPEG] * 2, 2)},
            }
        )
        if self.be.safe_to_probe_audio:
            self.audio = self._probe_part(
                {"type": "audio_url", "audio_url": {"url": wav_url([0] * 3200)}}
            )
        else:  # an engine that can crash on audio it can't take: read the model's config instead
            self.audio = bool(
                self.be.audio_from_config(self.model)
            )  # unknown -> no audio: refuse, don't crash

        self.info = {
            "engine": self.be.name + (f" {self.be.version}" if self.be.version else ""),
            "model": self.model,
            "root": self.root,
            "thinking_profile": self.profile,
            "answer_start_tail": self.decode(tuple(start[-12:])),
            "appended_after_generation_prompt": self.decode(tuple(suffix))
            if suffix
            else "",
            "logprob_token_ids": self.label_ids_supported,
            "vision": self.vision,
            "video": self.video,
            "audio": self.audio,
        }
        return self.info

    def _probe_part(self, part) -> bool:
        """Does a media part lengthen the prompt? (A model without that tower refuses it
        or drops it.)"""

        def n(content):
            d = self.up.post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "max_tokens": 1,
                    "chat_template_kwargs": THINK_OFF,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            return d.get("usage", {}).get("prompt_tokens", 0)

        try:
            return n([part, {"type": "text", "text": "x"}]) > n(
                [{"type": "text", "text": "x"}]
            )
        except (UpstreamError, OSError):
            return False

    def _probe_vision(self) -> bool:
        pixel = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLv"
            "AAAAAElFTkSuQmCC"
        )

        def n(content):
            d = self.up.post(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "max_tokens": 1,
                    "chat_template_kwargs": THINK_OFF,
                    "messages": [{"role": "user", "content": content}],
                },
            )
            return d.get("usage", {}).get("prompt_tokens", 0)

        try:
            plain = n([{"type": "text", "text": "x"}])
            with_image = n(
                [
                    {"type": "image_url", "image_url": {"url": pixel}},
                    {"type": "text", "text": "x"},
                ]
            )
            return with_image > plain
        except UpstreamError:
            return False

    # ------------------------------------------------------------------ prompts
    def prompt_ids(self, system: str, state: str) -> List[int]:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": state},
        ]
        return self.render(msgs, kwargs=THINK_OFF) + self.answer_suffix

    @lru_cache(maxsize=16384)
    def slot(
        self, qid: str, labels: Tuple[str, ...], lead: str
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Tokens of `lead` + "qid:" up to the label, and each label's token id.
        Every label must change exactly one token, at the same position."""
        encs = [self.encode(f"{lead}{qid}: {lab}") for lab in labels]
        if len({len(e) for e in encs}) != 1:
            raise SchemaError(
                f"question {qid!r}: its labels are not single tokens for this model"
            )
        diff = {i for e in encs[1:] for i in range(len(e)) if e[i] != encs[0][i]}
        if len(diff) != 1:
            raise SchemaError(
                f"question {qid!r}: its labels do not share one token position for this model"
            )
        pos = diff.pop()
        ids = tuple(e[pos] for e in encs)
        if len(set(ids)) != len(ids):
            raise SchemaError(f"question {qid!r}: two labels tokenize to the same id")
        return encs[0][:pos], ids

    # ------------------------------------------------------------------ reads
    def read(
        self, prompts: List[List[int]], label_ids: List[Tuple[int, ...]]
    ) -> List[Dict]:
        """One batched next-token read per prompt. Returns each label's logprob."""
        union = sorted({i for ids in label_ids for i in ids})
        rows, usage = self.be.read(self.model, prompts, union, TOPK)
        return [_label_logprobs(top, ids) for top, ids in zip(rows, label_ids)], usage

    def read_chat(
        self,
        system: str,
        content: list,
        lead_ids: Tuple[int, ...],
        ids: Tuple[int, ...],
    ) -> Dict:
        """An image state: the chat endpoint, continuing the answer line by one token."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
            {"role": "assistant", "content": self.decode(lead_ids)},
        ]
        top, usage = self.be.read_chat(
            self.model, messages, ids, TOPK, [self.decode((i,)) for i in ids]
        )
        return _label_logprobs(top, ids), usage

    # ------------------------------------------------------------------ think
    def think(self, system: str, state: str, budget: int) -> Tuple[List[int], Dict]:
        """Let the model reason for up to `budget` tokens, then return a prefix that
        ends right where the answer starts, with the thought in it."""
        if not self.profile:
            raise SchemaError(
                "think: this model has no thinking format this server knows"
            )
        p = PROFILES[self.profile]
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": state},
        ]
        kwargs = (
            dict(THINK_ON, reasoning_effort="medium")
            if self.profile == "harmony"
            else THINK_ON
        )
        prompt = self.render(msgs, kwargs=kwargs)
        op = list(self.encode(p["open"]))
        if prompt[-len(op) :] != op:
            prompt += op
        close = list(self.encode(p["close"]))
        stop = [t for t in close if self.decode((t,)).strip()][
            :1
        ]  # the close tag's special token
        ids, stopped = self.be.generate(self.model, prompt, budget, stop)
        closed = bool(stop) and stop[0] in ids
        if closed:
            ids = ids[: ids.index(stop[0])]
        info = {
            "tokens": len(ids),
            "closed": closed or stopped,
            "text": self.decode(tuple(ids)),
        }
        after = list(self.encode(p["after"])) if p["after"] else []
        return prompt + ids + close + after, info


def _label_logprobs(top: Dict[int, float], ids: Tuple[int, ...]) -> Dict:
    floor = min(top.values()) - 5.0  # a label outside the returned set
    lp = [top.get(i, floor) for i in ids]
    mass = sum(math.exp(x) for x in lp)
    ent = -sum(math.exp(v) * v for v in top.values())
    return {
        "label_logprobs": lp,
        "label_mass": mass,  # how much of the model's probability sits on legal labels
        "entropy": ent,
        "argmax_is_label": max(top, key=top.get) in ids,
    }
