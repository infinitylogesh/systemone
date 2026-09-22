"""Build RLCD training records with the serving stack's own tokens.

Each record is one question: the exact token ids systemone sends to vLLM for an
independent read (chat prompt + answer start + "qid:"), the label token ids at
the slot, and the gold distribution over those labels. Building them through a
running `systemone`-compatible vLLM server guarantees training and serving see
the same tokens.

Input: LocalLLaMA/typed-decisions (default) or a JSONL of Jev cases
{"state", "questions", "gold": {qid: {"probabilities": {...}} | {"label": name}}}.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from ..model import Model, Upstream
from ..schema import parse_questions, state_text, system_text


def gold_distribution(q, g):
    """The gold distribution in the question's label order."""
    names = q.names
    if q.type == "noul":
        p = g.get("noul")
        if p is None:
            probs = g.get("probabilities") or {}
            p = probs.get("true", 1.0 if str(g.get("label")).lower() in ("true", "yes") else 0.0)
        return [float(p), 1.0 - float(p)]
    probs = g.get("probabilities")
    if probs:
        keys = names if q.type == "choice" else [str(i) for i in range(len(names))]
        v = [float(probs.get(k, 0.0)) for k in keys]
    else:
        lab = str(g.get("label"))
        idx = names.index(lab) if lab in names else int(lab)
        v = [1.0 if i == idx else 0.0 for i in range(len(names))]
    s = sum(v)
    return [x / s for x in v] if s else [1.0 / len(v)] * len(v)


def load_cases(source, split, skip_first, limit):
    if source == "typed-decisions":
        from datasets import load_dataset

        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split).shuffle(seed=0)
        rows = [dict(id=r["id"], state=json.loads(r["state"]), questions=json.loads(r["questions"]), gold=json.loads(r["gold"]))
                for r in ds]
    else:
        rows = [json.loads(line) for line in open(source) if line.strip()]
    rows = rows[skip_first:]
    return rows[:limit] if limit else rows


def build(model, case):
    body = {"questions": case["questions"]}
    qs = parse_questions(body)
    system = system_text(qs, case.get("instructions"))
    prefix = model.prompt_ids(system, state_text(case["state"]))
    out = []
    for q in qs:
        if q.id not in case["gold"]:
            continue
        toks, ids = model.slot(q.id, tuple(q.labels), "")
        out.append({"case": case.get("id"), "qid": q.id, "qtype": q.type, "input_ids": prefix + list(toks),
                    "label_ids": list(ids), "target": gold_distribution(q, case["gold"][q.id])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="systemone rlcd prepare")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000", help="vLLM serving the base model")
    ap.add_argument("--source", default="typed-decisions", help="'typed-decisions' or a Jev-case JSONL")
    ap.add_argument("--split", default="train")
    ap.add_argument("--skip-first", type=int, default=300,
                    help="hold out the first N shuffled cases (systemone bench fits temperatures on them)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model = Model(Upstream(a.upstream))
    print(model.probe())
    cases = load_cases(a.source, a.split, a.skip_first, a.limit)
    with ThreadPoolExecutor(16) as pool:
        recs = [r for rs in pool.map(lambda c: build(model, c), cases) for r in rs]
    random.Random(0).shuffle(recs)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    lens = sorted(len(r["input_ids"]) for r in recs)
    print(f"wrote {len(recs)} questions from {len(cases)} cases to {a.out}; "
          f"tokens p50 {lens[len(lens) // 2]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
