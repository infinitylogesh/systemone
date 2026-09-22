"""One decision: prompt, per-question next-token reads, Jev answers."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .model import Model
from .schema import (
    Question,
    SchemaError,
    answer_name,
    jev_answer,
    parse_questions,
    softmax,
    stages,
    state_text,
    system_text,
)

POOL = ThreadPoolExecutor(64)


class Calibration:
    """Temperatures per (question type, option count), fitted by `systemone bench`
    on labelled data. Re-read whenever the file changes."""

    def __init__(self, path: Optional[str] = None):
        self.path, self.mtime, self.temps = path, None, {}

    def temperature(self, q: Question) -> float:
        if self.path:
            try:
                m = os.path.getmtime(self.path)
                if m != self.mtime:
                    self.temps = json.load(open(self.path)).get("temperatures", {})
                    self.mtime = m
            except OSError:
                pass
        t = self.temps
        key = f"{q.type}:{len(q.labels)}"
        if key in t:
            return float(t[key])
        # no fit for this option count: the same type's fit with the nearest count
        same = [
            (abs(int(k.split(":")[1]) - len(q.labels)), v)
            for k, v in t.items()
            if k.startswith(q.type + ":") and k.split(":")[1].isdigit()
        ]
        if same:
            return float(min(same)[1])
        return float(t.get(q.type, t.get("default", 1.0)))


def _urls(body: dict, field: str, kind: str) -> List[str]:
    """A field's media as data URLs: each item a data:<kind>/... URL or {content_type, base64}."""
    out = []
    items = body.get(field) or []
    
    if not isinstance(items, list):
        raise SchemaError(f"{field}: a list of data URLs")
    
    for i, m in enumerate(items):
        if isinstance(m, str) and m.startswith(f"data:{kind}/"):
            out.append(m)
        elif (
            isinstance(m, dict)
            and str(m.get("content_type", "")).startswith(f"{kind}/")
            and isinstance(m.get("base64"), str)
        ):
            out.append(f"data:{m['content_type']};base64,{m['base64']}")
        else:
            raise SchemaError(
                f"{field}[{i}]: a data:{kind}/... URL or an object with content_type and base64"
            )
    return out


def media_of(body: dict, model: Model) -> List[dict]:
    """Chat content parts for the request's images, videos, video frames and audio."""
    from .media import frames_to_video_url

    images = _urls(body, "images", "image")
    videos = _urls(body, "video", "video")
    frames = _urls(body, "video_frames", "image")
    audio = _urls(body, "audio", "audio")
    
    if frames:
        if len(frames) > 32:
            raise SchemaError("video_frames: at most 32 frames")
        fps = body.get("fps", 8)
        if not isinstance(fps, (int, float)) or not 0.1 <= fps <= 60:
            raise SchemaError("fps: between 0.1 and 60")
        try:
            videos.append(frames_to_video_url(frames, float(fps)))
        except (ValueError, IndexError) as e:
            raise SchemaError(f"video_frames: JPEG frames only ({e})")
    
    for items, ok, what in (
        (images, model.vision, "image"),
        (videos, model.video, "video"),
        (audio, model.audio, "audio"),
    ):
        if items and not ok:
            raise SchemaError(f"{what}: {model.root} does not take {what} input")
    return (
        [{"type": "image_url", "image_url": {"url": u}} for u in images]
        + [{"type": "video_url", "video_url": {"url": u}} for u in videos]
        + [{"type": "audio_url", "audio_url": {"url": u}} for u in audio]
    )


def decide(model: Model, body: dict, calibration: Calibration) -> Dict:
    if getattr(model, "chat_only", False):  # an OpenRouter model: see openrouter.py
        return model.decide(body, calibration)
    started = time.time()
    qs = parse_questions(body)
    mode = body.get("mode", "independent")

    if mode not in ("independent", "joint"):
        raise SchemaError('mode must be "independent" or "joint"')
    think = body.get("think", 0)

    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 8192:
        raise SchemaError("think: a thought budget in tokens, 0 to 8192")

    text = state_text(body.get("state"))
    images = media_of(body, model)  # images, video and audio parts
    system = system_text(qs, body.get("instructions"))

    thought = None
    if images:
        if think:
            raise SchemaError("think: supported for text states only")
        content = images + [{"type": "text", "text": text}]
    elif think:
        prefix, thought = model.think(system, text, think)
    else:
        prefix = model.prompt_ids(system, text)

    probs: Dict[str, Optional[List[float]]] = {}
    diag: Dict[str, Dict] = {}
    lines: List[str] = []
    reads, prompt_tokens = 0, None
    by_id = {q.id: q for q in qs}

    for stage in stages(qs):
        todo = []
        for q in stage:
            if any(
                answer_name(by_id[d], probs.get(d)) not in v
                for d, v in q.ask_if.items()
            ):
                probs[q.id] = None  # its ask_if condition failed
            else:
                todo.append(q)
        groups = [[q] for q in todo] if mode == "joint" else ([todo] if todo else [])

        for group in groups:
            lead = "\n".join(lines) + "\n" if lines else ""
            slots = [model.slot(q.id, tuple(q.labels), lead) for q in group]
            if images:
                futs = [
                    POOL.submit(model.read_chat, system, content, toks, ids)
                    for toks, ids in slots
                ]
                res = [f.result() for f in futs]
                outs, usage = [r[0] for r in res], res[0][1]
            else:
                outs, usage = model.read(
                    [prefix + list(toks) for toks, _ in slots],
                    [ids for _, ids in slots],
                )
            reads += 1
            prompt_tokens = prompt_tokens or usage.get("prompt_tokens")

            for q, o in zip(group, outs):
                t = calibration.temperature(q)
                probs[q.id] = softmax(o["label_logprobs"], t)
                diag[q.id] = dict(o, temperature=t)

            for q in group:
                lines.append(
                    f"{q.id}: {q.labels[max(range(len(q.labels)), key=lambda i: probs[q.id][i])]}"
                )
    return {
        "model": model.model,
        "answers": {
            q.id: (None if probs[q.id] is None else jev_answer(q, probs[q.id]))
            for q in qs
        },
        "usage": {
            "input_tokens": prompt_tokens or 0,
            "output_tokens": reads + (thought["tokens"] if thought else 0),
        },
        "diagnostics": {
            "engine": model.be.name
            + (f" {model.be.version}" if model.be.version else ""),
            "mode": mode,
            "timing": {"total_ms": (time.time() - started) * 1e3, "reads": reads},
            "thought": thought,
            "questions": diag,
        },
    }
