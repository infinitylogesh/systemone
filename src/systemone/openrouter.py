"""Decisions from a model hosted on OpenRouter, through its chat API.

OpenRouter exposes no tokenizer, no raw prompt with logprobs, and its providers start
a new reply instead of continuing a prefilled one, so the next-token read at an answer
slot (engine.decide) isn't possible. Instead the model writes its answers and the
distributions come from the reply's top logprobs:

  single        one request per stage: the model writes an "id: label" line per
                question, and each question's distribution is the top_logprobs at
                the first token after the colon on its line (the default; one prompt
                per case, so about 5x cheaper for 5 questions)
  per_question  one request per question with max_tokens 1: the reply's first
                token is that question's label (about 1 point more accurate on
                typed-decisions, and 5x the prompt tokens)

Labels are matched by token text. Only providers that return top_logprobs qualify;
the probe picks them from OpenRouter's endpoint list.
"""

import math
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .model import Upstream, UpstreamError
from .schema import (
    SchemaError,
    answer_name,
    jev_answer,
    parse_questions,
    softmax,
    stages,
    state_text,
    system_text,
)

BASE = "https://openrouter.ai/api"
READS = ("single", "per_question")
TOPK = 20
POOL = ThreadPoolExecutor(64)


class _Engine:
    """What play.py and the info route show as the engine."""

    name, version = "openrouter", None


class OpenRouterModel:
    """A served name backed by an OpenRouter model. engine.decide hands requests for
    it to decide() below."""

    chat_only = True
    be = _Engine()

    def __init__(
        self,
        slug: str,
        api_key: str,
        name: Optional[str] = None,
        read: str = "per_question",
        providers: Optional[List[str]] = None,
        base_url: str = BASE,
    ):
        if read not in READS:
            raise ValueError(f"openrouter read must be one of {READS}")
        if not api_key:
            raise UpstreamError(
                "openrouter: no API key (set OPENROUTER_API_KEY or --openrouter-key)"
            )
        self.slug, self.model, self.root, self.read_mode = slug, name or slug, slug, read
        self.up = Upstream(base_url, api_key, timeout=120)
        self.providers = providers
        self.vision = self.video = self.audio = False
        self.info: Dict = {}

    def probe(self) -> Dict:
        """The model's input modalities, and the providers that return top_logprobs."""
        d = self.up.get(
            f"/v1/models/{urllib.parse.quote(self.slug, safe='/')}/endpoints"
        )["data"]
        eps = d.get("endpoints") or []
        ok = [
            e["provider_name"]
            for e in eps
            if {"logprobs", "top_logprobs"} <= set(e.get("supported_parameters") or [])
        ]
        if self.providers is None:
            # the listing can claim logprobs a provider then leaves out: ask each one,
            # keep those that return them, fastest first
            timed = [r for r in POOL.map(self._try_provider, ok) if r]
            if not timed:
                raise UpstreamError(
                    f"openrouter: no provider of {self.slug} returns logprobs "
                    f"(providers: {[e['provider_name'] for e in eps]})"
                )
            self.providers = [p for _, p in sorted(timed)]
        mods = set((d.get("architecture") or {}).get("input_modalities") or [])
        self.vision = "image" in mods
        # video and audio pass through untested providers: refuse them rather than guess
        self.video = self.audio = False
        self.info = {
            "model": self.model,
            "openrouter_model": self.slug,
            "engine": "openrouter",
            "read": self.read_mode,
            "providers": self.providers,
            "vision": self.vision,
            "video": self.video,
            "audio": self.audio,
            "think": False,
        }
        return self.info

    def _try_provider(self, provider) -> Optional[Tuple[float, str]]:
        t = time.time()
        try:
            self._chat(
                [{"role": "user", "content": "Reply with yes."}], 1, [provider], tries=1
            )
        except UpstreamError:
            return None
        return time.time() - t, provider

    # ------------------------------------------------------------------ requests
    def _chat(
        self, messages, max_tokens, providers=None, tries=4
    ) -> Tuple[List[dict], dict]:
        body = {
            "model": self.slug,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOPK,
            "provider": {
                "order": providers or self.providers,
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        err = None
        for attempt in range(tries):
            try:
                d = self.up.post("/v1/chat/completions", body)
                content = ((d.get("choices") or [{}])[0].get("logprobs") or {}).get(
                    "content"
                )
                if content:
                    return content, d.get("usage") or {}
                err = UpstreamError("openrouter: a reply without logprobs")
            except UpstreamError as e:
                # a bad key, no credits or a request OpenRouter rejects won't change on a
                # retry; rate limits, timeouts and provider errors often do
                if any(f"upstream {c} " in str(e) for c in ("401", "402", "403")):
                    raise
                err = e
            except OSError as e:  # connection dropped or timed out
                err = UpstreamError(f"openrouter: {e!r}")
            if attempt + 1 < tries:
                time.sleep(0.25 * 2**attempt)
        raise err

    def _read_lines(self, system, content, lead, group):
        """One request for a group of questions: the top_logprobs at each answer."""
        ask = (("Answers so far:\n" + "\n".join(lead) + "\n\n") if lead else "") + (
            "Answer these questions now, one line each, in this order: "
            + ", ".join(q.id for q in group)
            + "."
        )
        budget = 8 + sum(len(q.id) // 3 + 5 for q in group)
        toks, usage = self._chat(_messages(system, content, ask), budget)
        text = "".join(t["token"] for t in toks)
        starts, pos = [], 0
        for t in toks:
            starts.append(pos)
            pos += len(t["token"])
        # each non-empty line; a question's line starts with its id, else it is the line
        # at the question's position (models sometimes echo the question text instead)
        spans, st = [], 0
        for line in text.split("\n"):
            if line.strip():
                spans.append(
                    (st, st + len(line), line.strip().lstrip("*-# ").lower())
                )
            st += len(line) + 1
        by_id = {}
        for n, (_, _, low) in enumerate(spans):
            q = next(
                (
                    q
                    for q in group
                    if q.id not in by_id and low.startswith(q.id.lower() + ":")
                ),
                None,
            )
            if q:
                by_id[q.id] = n
        free = [n for n in range(len(spans)) if n not in by_id.values()]
        found = {}
        for q in group:
            n = by_id.get(q.id)
            if n is None and free:
                n = free.pop(0)
            if n is None:
                continue
            s0, e0, _ = spans[n]
            colon = text.rfind(":", s0, e0)
            i = colon + 1 if colon >= 0 else s0
            k = next(
                (
                    m
                    for m, sm in enumerate(starts)
                    if sm + len(toks[m]["token"]) > i
                    and toks[m]["token"][max(0, i - sm) :].strip()
                ),
                None,
            )
            if k is not None and starts[k] < e0:
                found[q.id] = toks[k]["top_logprobs"]
        return found, usage

    def _read_one(self, system, content, lead, q):
        ask = (("Answers so far:\n" + "\n".join(lead) + "\n\n") if lead else "") + (
            f"Now answer only question {q.id}. Reply with just its label "
            f"({' / '.join(q.labels)}) and nothing else."
        )
        toks, usage = self._chat(_messages(system, content, ask), 1)
        return toks[0]["top_logprobs"], usage

    # ------------------------------------------------------------------ decide
    def decide(self, body: dict, calibration) -> Dict:
        from .engine import media_of

        started = time.time()
        qs = parse_questions(body)
        mode = body.get("mode", "independent")
        if mode not in ("independent", "joint"):
            raise SchemaError('mode must be "independent" or "joint"')
        if body.get("think"):
            raise SchemaError("think: not available for OpenRouter models")
        text = state_text(body.get("state"))
        media = media_of(body, self)
        content = media + [{"type": "text", "text": text}] if media else text
        system = system_text(qs, body.get("instructions"))

        probs: Dict[str, Optional[List[float]]] = {}
        diag: Dict[str, Dict] = {}
        lines: List[str] = []
        by_id = {q.id: q for q in qs}
        calls, fallbacks, in_toks, out_toks = 0, 0, 0, 0

        for stage in stages(qs):
            todo = []
            for q in stage:
                if any(
                    answer_name(by_id[d], probs.get(d)) not in v
                    for d, v in q.ask_if.items()
                ):
                    probs[q.id] = None
                else:
                    todo.append(q)
            groups = [[q] for q in todo] if mode == "joint" else ([todo] if todo else [])
            for group in groups:
                lead = list(lines)
                tops: Dict[str, list] = {}
                if self.read_mode == "single":
                    tops, usage = self._read_lines(system, content, lead, group)
                    calls += 1
                    in_toks += usage.get("prompt_tokens", 0)
                    out_toks += usage.get("completion_tokens", 0)
                miss = [q for q in group if q.id not in tops]
                if self.read_mode == "single":
                    fallbacks += len(miss)
                for q, (top, usage) in zip(
                    miss,
                    POOL.map(lambda q: self._read_one(system, content, lead, q), miss),
                ):
                    tops[q.id] = top
                    calls += 1
                    in_toks += usage.get("prompt_tokens", 0)
                    out_toks += usage.get("completion_tokens", 0)
                for q in group:
                    lp, mass = _label_logprobs(tops[q.id], q.labels)
                    t = calibration.temperature(q)
                    probs[q.id] = softmax(lp, t)
                    diag[q.id] = {
                        "label_logprobs": lp,
                        "label_mass": mass,
                        "temperature": t,
                    }
                for q in group:
                    top = max(range(len(q.labels)), key=lambda i: probs[q.id][i])
                    lines.append(f"{q.id}: {q.labels[top]}")
        return {
            "model": self.model,
            "answers": {
                q.id: (None if probs[q.id] is None else jev_answer(q, probs[q.id]))
                for q in qs
            },
            "usage": {"input_tokens": in_toks, "output_tokens": out_toks},
            "diagnostics": {
                "engine": "openrouter",
                "read": self.read_mode,
                "mode": mode,
                "timing": {
                    "total_ms": (time.time() - started) * 1e3,
                    "requests": calls,
                    "fallbacks": fallbacks,
                },
                "questions": diag,
            },
        }


def _messages(system, content, ask):
    if isinstance(content, str):
        user = content + "\n\n" + ask
    else:
        user = content + [{"type": "text", "text": ask}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _label_logprobs(top: List[dict], labels) -> Tuple[List[float], float]:
    """Each label's logprob: the best returned token whose text is the label (a label
    outside the top 20 gets 5 nats below the lowest one returned)."""
    floor = min(t["logprob"] for t in top) - 5.0
    lp = []
    for lab in labels:
        m = [t["logprob"] for t in top if t["token"].strip() == lab]
        lp.append(max(m) if m else floor)
    return lp, sum(math.exp(x) for x in lp)
