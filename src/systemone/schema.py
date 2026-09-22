"""Jev request parsing, the system prompt, dependency stages and response shapes.

A request carries `questions`, a map of id -> {"type", "instructions", "criteria"}:
  noul:   yes/no; optional criteria {"true": ..., "false": ...}
  choice: criteria maps option name -> description (or null)
  score:  criteria is an ordered list of levels
Each question is shown to the model with single-token labels (yes/no, A-Z, 1-9) so
that one next-token read gives its whole answer distribution.
"""

import json
import math
from typing import Dict, List, Optional

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = 26


class SchemaError(ValueError):
    """The request can't be answered as asked (HTTP 422)."""


class Question:
    __slots__ = (
        "id",
        "type",
        "instructions",
        "choices",
        "labels",
        "depends_on",
        "ask_if",
    )

    def __init__(self, qid, qtype, instructions, choices, labels, depends_on, ask_if):
        self.id = qid
        self.type = qtype
        self.instructions = instructions
        self.choices = choices  # [(name, description or None)]
        self.labels = labels  # what the model writes, one per choice
        self.depends_on = depends_on
        self.ask_if = ask_if

    @property
    def names(self) -> List[str]:
        return [c[0] for c in self.choices]


def parse_questions(body: dict) -> List[Question]:
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    out = []
    for qid, q in qs.items():
        qid = str(qid)
        if not qid.strip() or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, without ':' or newlines"
            )
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, crit = q.get("type"), q.get("criteria")
        ins = q.get("instructions", "")
        ins = ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with true and false"
                )
            crit = crit or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names to descriptions"
                )
            choices = [(str(n), d if d is None else str(d)) for n, d in crit.items()]
            labels = list(LETTERS[: len(choices)])
        elif kind == "score":
            if not isinstance(crit, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list of levels"
                )
            choices = [(str(level), None) for level in crit]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else list(LETTERS[: len(choices)])
            )
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > MAX_OPTIONS:
            raise SchemaError(f"question {qid!r}: at most {MAX_OPTIONS} alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a non-empty list of answers"
            )
        out.append(
            Question(
                qid,
                kind,
                ins.strip(),
                choices,
                labels,
                list(dict.fromkeys(deps + list(ask_if))),
                ask_if,
            )
        )
    by_id = {q.id: q for q in out}
    for q in out:
        for dep in q.depends_on:
            if dep not in by_id or dep == q.id:
                raise SchemaError(
                    f"question {q.id!r}: depends on unknown question {dep!r}"
                )
        for dep, vals in q.ask_if.items():
            if any(v not in by_id[dep].names for v in vals):
                raise SchemaError(
                    f"question {q.id!r}: ask_if values for {dep!r} must be among {by_id[dep].names}"
                )
    stages(out)  # refuses a cycle
    return out


def stages(qs: List[Question]) -> List[List[Question]]:
    """Questions grouped so that each comes after everything it depends on;
    declaration order is kept within a stage."""
    pending, done, levels = list(qs), set(), []
    while pending:
        level = [q for q in pending if all(d in done for d in q.depends_on)]
        if not level:
            raise SchemaError(
                "dependency cycle among " + ", ".join(q.id for q in pending)
            )
        levels.append(level)
        done |= {q.id for q in level}
        pending = [q for q in pending if q.id not in done]
    return levels


def system_text(qs: List[Question], instructions: Optional[str] = None) -> str:
    s = (
        "Answer a fixed set of questions about the state the user provides. Each question lists "
        "its allowed answers; reply with exactly one label per question.\n"
    )
    if instructions:
        s += "\n" + str(instructions).strip() + "\n"
    for q in qs:
        s += f"\nQuestion {q.id}: {q.instructions}\n"
        for (name, desc), label in zip(q.choices, q.labels):
            if q.type == "noul":
                s += f"  {label}: {desc.strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({desc.strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    return (
        s
        + '\nReply with one line per question, in this order, formatted as "id: label".'
    )


def state_text(state) -> str:
    if state is None:
        raise SchemaError("state: required")
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    zs = [x / temperature for x in xs]
    m = max(zs)
    es = [math.exp(z - m) for z in zs]
    t = sum(es)
    return [e / t for e in es]


def jev_answer(q: Question, probs: List[float]) -> Dict:
    """Jev's response shape for one question."""
    top = max(range(len(probs)), key=lambda i: probs[i])
    if q.type == "noul":
        return {"type": "noul", "noul": probs[0]}
    if q.type == "choice":
        return {
            "type": "choice",
            "choice": q.names[top],
            "probabilities": dict(zip(q.names, probs)),
            "confidence": probs[top],
        }
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),  # 0-indexed, like Jev
        "legend": {str(i): n for i, n in enumerate(q.names)},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "confidence": probs[top],
    }


def answer_name(q: Question, probs: Optional[List[float]]) -> Optional[str]:
    """The name ask_if compares against: yes/no, an option name or a level name."""
    if probs is None:
        return None
    return q.names[max(range(len(probs)), key=lambda i: probs[i])]
