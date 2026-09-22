"""Conformance and quality/latency benchmark for any /v1/systemone endpoint.

  systemone bench --backend qwen=http://127.0.0.1:8011 --backend laya=http://127.0.0.1:18012,basic \\
                  --suite features,public,typed,latency --out results.json

A backend marked `basic` only implements Jev's core contract, so the extension
cases (depends_on, ask_if, think, images, ...) are skipped for it.

Suites
  features  32 request cases: every question type, state shapes, 6 languages,
            depends_on / ask_if, think, images, validation errors
  public    AG News (4), DAIR Emotion (6), SST-2 as noul and as choice
  typed     LocalLLaMA/typed-decisions test split, scored like Jev's published
            numbers (accuracy, soft accuracy, Brier, ECE, score MAE), raw and with
            temperatures fitted on the train split (written to --calibration-dir)
  latency   sequential p50/p90 for 1 and 4 questions, typed-decisions cases, and a
            concurrency sweep
Needs `datasets` for public/typed.
"""

import argparse
import base64
import json
import math
import os
import random
import statistics
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

TOKEN = os.environ.get("SYSTEMONE_BENCH_TOKEN", "")
EXTRA = {}  # merged into every public/typed/latency request, e.g. {"think": 64}


def post(url, body, timeout=600):
    t = time.perf_counter()
    if EXTRA and "questions" in body:
        body = dict(EXTRA, **body)
    headers = {"content-type": "application/json"}
    if TOKEN:
        headers["authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code, out = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        code = e.code
        try:
            out = json.loads(e.read() or b"{}")
        except ValueError:
            out = {}
    except Exception as e:  # connection reset, timeout
        code, out = 0, {"error": {"message": repr(e)}}
    return code, out, (time.perf_counter() - t) * 1000


def top(a):
    if a is None:
        return None
    if a["type"] == "noul":
        return "yes" if a["noul"] >= 0.5 else "no"
    if a["type"] == "choice":
        return a["choice"]
    p = a["probabilities"]
    return a["legend"][max(p, key=p.get)]


def conf(a):
    if a is None:
        return None
    return max(a["noul"], 1 - a["noul"]) if a["type"] == "noul" else a.get("confidence")


def png(rgb, size=64, canvas=128):
    """A solid square on white, as a data URL (no PIL needed)."""
    o = (canvas - size) // 2
    rows = []
    for y in range(canvas):
        row = bytearray(b"\x00")
        for x in range(canvas):
            row += (
                bytes(rgb)
                if o <= x < o + size and o <= y < o + size
                else b"\xff\xff\xff"
            )
        rows.append(bytes(row))

    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    data = b"\x89PNG\r\n\x1a\n" + ch(
        b"IHDR", struct.pack(">IIBBBBB", canvas, canvas, 8, 2, 0, 0, 0)
    )
    data += ch(b"IDAT", zlib.compress(b"".join(rows))) + ch(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


# =============================================================================== features
TICKET_Q = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}
TICKET_EN = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
TICKET_EXPECT = {
    "department": "billing",
    "churn_risk": "yes",
    "refund_requested": "yes",
}
MULTI = [
    (
        "hi",
        "मुझसे मार्च के लिए दो बार शुल्क लिया गया। कृपया आज ही डुप्लिकेट राशि वापस करें, नहीं तो हम अपना प्लान रद्द कर देंगे।",
    ),
    (
        "de",
        "Wir wurden für März doppelt belastet. Bitte erstatten Sie die doppelte Zahlung heute, sonst kündigen wir unseren Vertrag.",
    ),
    (
        "ja",
        "3月分が二重に請求されました。本日中に重複分を返金してください。さもなければプランを解約します。",
    ),
    (
        "es",
        "Nos cobraron dos veces en marzo. Por favor reembolsen el cargo duplicado hoy o cancelaremos nuestro plan.",
    ),
    (
        "ta",
        "மார்ச் மாதத்திற்கு எங்களிடம் இரண்டு முறை கட்டணம் வசூலிக்கப்பட்டது. இன்றே இரட்டைக் கட்டணத்தைத் திருப்பித் தாருங்கள், இல்லையெனில் எங்கள் திட்டத்தை ரத்து செய்வோம்.",
    ),
    (
        "ar",
        "تم خصم المبلغ مرتين لشهر مارس. يرجى استرداد المبلغ المكرر اليوم وإلا سنلغي اشتراكنا.",
    ),
]


def feature_cases():
    C = []

    def case(name, body, expect=None, status=200, extension=False):
        body.setdefault("model", "jev-latest")
        C.append(
            dict(
                name=name,
                body=body,
                expect=expect or {},
                status=status,
                extension=extension,
            )
        )

    outage = {"outage": {"type": "noul", "instructions": "Is there a service outage?"}}
    case(
        "noul yes",
        {
            "state": "The server has been down for 3 hours and customers cannot log in.",
            "questions": outage,
        },
        {"outage": "yes"},
    )
    case(
        "noul no",
        {
            "state": "Thanks for the quick help yesterday, everything works great now!",
            "questions": outage,
        },
        {"outage": "no"},
    )
    case(
        "noul with true/false criteria",
        {
            "state": "Congratulations!!! You WON a $1000 gift card. Click http://bit.ly/xx to claim now.",
            "questions": {
                "spam": {
                    "type": "noul",
                    "instructions": "Is this message spam?",
                    "criteria": {
                        "true": "unsolicited promotion or scam",
                        "false": "legitimate message",
                    },
                }
            },
        },
        {"spam": "yes"},
        extension=True,
    )
    case(
        "choice 4-way",
        {"state": TICKET_EN, "questions": {"department": TICKET_Q["department"]}},
        {"department": "billing"},
    )
    case(
        "choice null descriptions",
        {
            "state": 'fn main() { let v: Vec<i32> = vec![1,2,3]; println!("{:?}", v); }',
            "questions": {
                "lang": {
                    "type": "choice",
                    "instructions": "Which programming language is this?",
                    "criteria": {
                        "python": None,
                        "rust": None,
                        "go": None,
                        "java": None,
                        "c": None,
                    },
                }
            },
        },
        {"lang": "rust"},
        extension=True,
    )
    case(
        "choice 12 options",
        {
            "state": "Can you set an alarm for 6:30 tomorrow morning?",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What is the user's intent?",
                    "criteria": {
                        o: None
                        for o in [
                            "weather",
                            "alarm_set",
                            "alarm_remove",
                            "music_play",
                            "calendar_set",
                            "email_send",
                            "news_query",
                            "timer",
                            "takeaway_order",
                            "transport_ticket",
                            "iot_lights",
                            "joke",
                        ]
                    },
                }
            },
        },
        {"intent": "alarm_set"},
    )
    case(
        "score ordinal high",
        {
            "state": "URGENT: production database is corrupted, all orders failing, CEO on the call.",
            "questions": {"urgency": TICKET_Q["urgency"]},
        },
        {"urgency": "critical deadline or blocking issue"},
    )
    case(
        "score ordinal low",
        {
            "state": "Whenever you get a chance, could you update my billing address? No rush.",
            "questions": {"urgency": TICKET_Q["urgency"]},
        },
        {"urgency": "not urgent"},
    )
    case(
        "score 1-5 sentiment",
        {
            "state": "The food was cold, the waiter was rude, and we waited an hour. Never coming back.",
            "questions": {
                "stars": {
                    "type": "score",
                    "instructions": "How many stars would this reviewer give?",
                    "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
                }
            },
        },
        {"stars": "1 star"},
    )
    case(
        "4 mixed questions, JSON state",
        {"state": TICKET_EN, "questions": TICKET_Q},
        TICKET_EXPECT,
    )
    case(
        "string state",
        {
            "state": "From: user@acme.com\nSubject: Duplicate charge\n\n"
            + TICKET_EN["body"],
            "questions": TICKET_Q,
        },
        TICKET_EXPECT,
    )
    case(
        "conversation state",
        {
            "state": [
                {"role": "user", "content": "My card got charged twice for March."},
                {"role": "assistant", "content": "Sorry about that! Let me check."},
                {
                    "role": "user",
                    "content": "Just refund it today or I'm cancelling the subscription.",
                },
            ],
            "questions": TICKET_Q,
        },
        TICKET_EXPECT,
    )
    for lang, text in MULTI:
        case(
            f"multilingual {lang}",
            {"state": {"body": text}, "questions": TICKET_Q},
            TICKET_EXPECT,
        )
    case(
        "depends_on",
        {
            "state": "My laptop screen flickers and then goes black after the latest driver update.",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "What kind of issue is this?",
                    "criteria": {"hardware": None, "software": None, "account": None},
                },
                "escalate": {
                    "type": "noul",
                    "instructions": "Given the category, should this go to tier-2 engineering?",
                    "depends_on": ["category"],
                },
            },
        },
        {"category": "software"},
        extension=True,
    )
    cancel = {
        "cancel": {"type": "noul", "instructions": "Does the user want to cancel?"},
        "reason": {
            "type": "choice",
            "instructions": "Why does the user want to cancel?",
            "criteria": {
                "price": None,
                "quality": None,
                "support": None,
                "other": None,
            },
            "ask_if": {"cancel": ["yes"]},
        },
    }
    case(
        "ask_if (asked)",
        {
            "state": "I want to cancel my subscription, your prices are too high.",
            "questions": cancel,
        },
        {"cancel": "yes", "reason": "price"},
        extension=True,
    )
    case(
        "ask_if (skipped)",
        {"state": "How do I change my profile picture?", "questions": cancel},
        {"cancel": "no", "reason": None},
        extension=True,
    )
    for m in ("independent", "joint"):
        case(
            f"mode={m}",
            {
                "state": "The package arrived a day late but the product itself is fine I guess.",
                "mode": m,
                "questions": {
                    "tone": {
                        "type": "score",
                        "instructions": "How upset is the customer?",
                        "criteria": ["calm", "annoyed", "furious"],
                    },
                    "complaint": {
                        "type": "noul",
                        "instructions": "Is this a complaint?",
                    },
                },
            },
            {"complaint": "yes"},
            extension=True,
        )
    case(
        "ambiguous state",
        {"state": "ok", "questions": {"department": TICKET_Q["department"]}},
    )
    case(
        "think=256",
        {
            "state": "A train leaves at 14:50 and the trip takes 95 minutes. The meeting starts at 16:30.",
            "questions": {
                "on_time": {
                    "type": "noul",
                    "instructions": "Will the passenger arrive before the meeting starts?",
                }
            },
            "think": 256,
        },
        {"on_time": "yes"},
        extension=True,
    )
    for name, rgb in (
        ("red", (220, 30, 30)),
        ("green", (30, 180, 60)),
        ("blue", (30, 60, 220)),
    ):
        case(
            f"image color {name}",
            {
                "state": {"task": "Look at the image."},
                "images": [png(rgb)],
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "What color is the square in the image?",
                        "criteria": {
                            "red": None,
                            "green": None,
                            "blue": None,
                            "yellow": None,
                        },
                    }
                },
            },
            {"color": name},
            extension=True,
        )
    case(
        "bad type -> 422",
        {"state": "x", "questions": {"q": {"type": "rank", "instructions": "?"}}},
        status=422,
    )
    case(
        "missing state -> 422",
        {"questions": {"q": {"type": "noul", "instructions": "?"}}},
        status=422,
        extension=True,
    )
    case(
        "27 options -> 422",
        {
            "state": "x",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": {f"opt{i}": None for i in range(27)},
                }
            },
        },
        status=422,
        extension=True,
    )
    return C


def run_features(backends):
    rows = []
    for c in feature_cases():
        for be in backends:
            if c["extension"] and be["basic"]:
                continue
            code, out, ms = post(be["url"], json.loads(json.dumps(c["body"])))
            row = dict(name=c["name"], backend=be["name"], status=code, ms=round(ms, 1))
            msg = (
                (out.get("error") or {}).get("message", "")
                if isinstance(out.get("error"), dict)
                else ""
            )
            if c["status"] != 200:
                row["ok"] = code == c["status"]
            elif code == 422 and "does not take image input" in msg:
                row.update(
                    ok=None, detail="n/a: text-only model"
                )  # a correct refusal, not a failure
            elif code != 200:
                row.update(ok=False, detail=json.dumps(out)[:300])
            else:
                ans = out["answers"]
                row["answers"] = {
                    k: {
                        "label": top(v),
                        "conf": None if conf(v) is None else round(conf(v), 3),
                    }
                    for k, v in ans.items()
                }
                row["misses"] = [
                    k for k, want in c["expect"].items() if top(ans.get(k)) != want
                ]
                row["ok"] = not row["misses"]
            print(
                f"{'N/A ' if row['ok'] is None else 'PASS' if row['ok'] else 'FAIL'} [{be['name']:>8}] {c['name']:30} {row['ms']:7.0f}ms "
                f"{json.dumps(row.get('answers', row.get('detail', '')), ensure_ascii=False)[:160]}",
                flush=True,
            )
            rows.append(row)
    return rows


# =============================================================================== shared eval helpers
def ece(confs, corrects, bins=10):
    b = [[] for _ in range(bins)]
    for c, k in zip(confs, corrects):
        b[min(bins - 1, int(c * bins))].append((c, k))
    n = len(confs)
    return sum(
        len(x)
        / n
        * abs(statistics.mean(c for c, _ in x) - statistics.mean(k for _, k in x))
        for x in b
        if x
    )


def answer_vector(a, q):
    """The answer as probabilities in the question's own option order."""
    if a["type"] == "noul":
        return [a["noul"], 1 - a["noul"]]  # [true, false]
    if a["type"] == "choice":
        return [a["probabilities"].get(k, 1e-9) for k in q["criteria"]]
    return [a["probabilities"].get(str(i), 1e-9) for i in range(len(q["criteria"]))]


def retemper(p, t):
    z = [math.log(max(x, 1e-12)) / t for x in p]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def fit_temperature(pairs):
    """argmin_T mean cross-entropy(gold, softmax(log p / T)), T in [0.1, 20]."""
    if len(pairs) < 10:
        return 1.0

    def loss(t):
        return -statistics.mean(
            sum(g * math.log(max(q, 1e-12)) for g, q in zip(gold, retemper(p, t)))
            for p, gold in pairs
        )

    lo, hi = math.log(0.1), math.log(20.0)
    for _ in range(60):  # golden-section search on log T
        a = hi - (hi - lo) * 0.618
        b = lo + (hi - lo) * 0.618
        if loss(math.exp(a)) < loss(math.exp(b)):
            hi = b
        else:
            lo = a
    return round(math.exp((lo + hi) / 2), 4)


def pmap(fn, items, conc):
    with ThreadPoolExecutor(max(1, conc)) as pool:
        return list(pool.map(fn, items))


# =============================================================================== public datasets
PUBLIC = {
    "ag_news": dict(
        hf=("fancyzhx/ag_news", None, "test"),
        text="text",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        q={
            "type": "choice",
            "instructions": "What is the topic of this news article?",
            "criteria": {
                "World": "world news, politics, international affairs",
                "Sports": "sports and athletes",
                "Business": "business, companies, economy, markets",
                "Sci/Tech": "science and technology",
            },
        },
    ),
    "emotion": dict(
        hf=("dair-ai/emotion", "split", "test"),
        text="text",
        labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
        q={
            "type": "choice",
            "instructions": "Which emotion does the writer express?",
            "criteria": {
                k: None for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]
            },
        },
    ),
    "sst2": dict(
        hf=("stanfordnlp/sst2", None, "validation"),
        text="sentence",
        labels=[0, 1],
        q={
            "type": "noul",
            "instructions": "Is the sentiment of this movie review positive?",
        },
    ),
    "sst2_choice": dict(
        hf=("stanfordnlp/sst2", None, "validation"),
        text="sentence",
        labels=["negative", "positive"],
        q={
            "type": "choice",
            "instructions": "What is the sentiment of this movie review?",
            "criteria": {"negative": None, "positive": None},
        },
    ),
}


def run_public(backends, n, conc):
    from datasets import load_dataset

    out = {}
    for dname, spec in PUBLIC.items():
        repo, cfg, split = spec["hf"]
        ds = list(load_dataset(repo, cfg, split=split).shuffle(seed=0).select(range(n)))
        for be in backends:

            def one(ex):
                code, res, ms = post(
                    be["url"],
                    {
                        "model": "jev-latest",
                        "state": ex[spec["text"]],
                        "questions": {"label": spec["q"]},
                    },
                )
                if code != 200:
                    return None, None, ms
                a = res["answers"]["label"]
                if a["type"] == "noul":
                    return int(a["noul"] >= 0.5), conf(a), ms
                return spec["labels"].index(a["choice"]), a["confidence"], ms

            t = time.perf_counter()
            preds = pmap(one, ds, conc if not be["serial"] else 1)
            wall = time.perf_counter() - t
            ok = [
                p is not None and p == ex["label"] for (p, _, _), ex in zip(preds, ds)
            ]
            confs = [c for _, c, _ in preds if c is not None]
            r = dict(
                acc=round(sum(ok) / n, 3),
                n=n,
                errors=sum(p is None for p, _, _ in preds),
                mean_conf=round(statistics.mean(confs), 3) if confs else None,
                ece=round(
                    ece(
                        [c for (_, c, _) in preds if c is not None],
                        [k for k, (_, c, _) in zip(ok, preds) if c is not None],
                    ),
                    3,
                ),
                wall_s=round(wall, 1),
            )
            out[f"{dname}/{be['name']}"] = r
            print(f"PUBLIC {dname:12} {be['name']:>8} {r}", flush=True)
    return out


# =============================================================================== typed-decisions
def load_typed(split, n):
    from datasets import load_dataset

    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    rows = (
        list(ds) if n <= 0 else list(ds.shuffle(seed=0).select(range(min(n, len(ds)))))
    )
    return [
        dict(
            id=r["id"],
            workflow=r["workflow"],
            state=json.loads(r["state"]),
            questions=json.loads(r["questions"]),
            gold=json.loads(r["gold"]),
        )
        for r in rows
    ]


def gold_vector(g, q):
    if q["type"] == "noul":
        t = g.get("noul", g.get("probabilities", {}).get("true", 0.5))
        return [t, 1 - t]
    if q["type"] == "choice":
        v = [g["probabilities"].get(k, 0.0) for k in q["criteria"]]
    else:
        v = [g["probabilities"].get(str(i), 0.0) for i in range(len(q["criteria"]))]
    s = sum(v)
    return [x / s for x in v] if s else [1 / len(v)] * len(v)


def typed_metrics(records, temps=None):
    """Scored like the typed-decisions notebook (and Jev's published row)."""
    acc, soft, brier, confs, corr, mae = [], [], [], [], [], []
    for rec in records:
        for qid, q in rec["questions"].items():
            p = rec["pred"].get(qid)
            if p is None:
                continue
            g = rec["gold"][qid]
            key = f"{q['type']}:{len(p)}"
            if temps:
                p = retemper(p, temps.get(key, temps.get("default", 1.0)))
            gv = gold_vector(g, q)
            if q["type"] == "choice":
                names = list(q["criteria"])
                k = float(
                    names[max(range(len(p)), key=lambda i: p[i])] == str(g["label"])
                )
            elif q["type"] == "noul":
                k = float(
                    ("true" if p[0] >= 0.5 else "false") == str(g["label"]).lower()
                )
            else:
                lvl = max(range(len(p)), key=lambda i: p[i])
                k = float(lvl == int(g.get("label", round(g.get("score", 0)))))
                mae.append(
                    abs(sum(i * x for i, x in enumerate(p)) - g.get("score", 0.0))
                )
            acc.append(k)
            corr.append(k)
            confs.append(max(p))
            if q["type"] in ("choice", "noul"):
                soft.append(sum(a * b for a, b in zip(p, gv)))
                brier.append(sum((a - b) ** 2 for a, b in zip(p, gv)))
    m = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
    return dict(
        accuracy=m(acc),
        soft_accuracy=m(soft),
        brier=m(brier),
        ece=round(ece(confs, corr), 4),
        score_mae=m(mae),
        mean_conf=m(confs),
        decisions=len(acc),
    )


def run_typed_split(be, cases, conc):
    def one(c):
        code, res, ms = post(
            be["url"],
            {"model": "jev-latest", "state": c["state"], "questions": c["questions"]},
        )
        pred = {}
        if code == 200:
            for qid, q in c["questions"].items():
                a = res["answers"].get(qid)
                pred[qid] = answer_vector(a, q) if a else None
        return dict(c, pred=pred, ms=ms, ok=code == 200)

    return pmap(one, cases, 1 if be["serial"] else conc)


def run_typed(backends, n_test, n_calib, conc, calib_dir=None):
    test = load_typed("test", n_test)
    train = load_typed("train", n_calib) if n_calib else []
    out = {}
    for be in backends:
        t = time.perf_counter()
        recs = run_typed_split(be, test, conc)
        wall = time.perf_counter() - t
        res = dict(
            raw=typed_metrics(recs),
            errors=sum(not r["ok"] for r in recs),
            wall_s=round(wall, 1),
            per_workflow={},
        )
        for wf in sorted({r["workflow"] for r in recs}):
            res["per_workflow"][wf] = typed_metrics(
                [r for r in recs if r["workflow"] == wf]
            )["accuracy"]
        if train:
            cal = run_typed_split(be, train, conc)
            groups = {}
            for rec in cal:
                for qid, q in rec["questions"].items():
                    p = rec["pred"].get(qid)
                    if p is not None:
                        groups.setdefault(f"{q['type']}:{len(p)}", []).append(
                            (p, gold_vector(rec["gold"][qid], q))
                        )
            temps = {k: fit_temperature(v) for k, v in groups.items()}
            res["temperatures"] = temps
            res["calibrated"] = typed_metrics(recs, temps)
            if calib_dir:
                os.makedirs(calib_dir, exist_ok=True)
                path = os.path.join(calib_dir, f"{be['name']}.json")
                json.dump(
                    {
                        "fitted_on": f"LocalLLaMA/typed-decisions train ({len(train)} cases)",
                        "temperatures": temps,
                    },
                    open(path, "w"),
                    indent=1,
                )
                res["calibration_file"] = path
        out[be["name"]] = res
        print(
            f"TYPED {be['name']:>8} raw={res['raw']} cal={res.get('calibrated')} T={res.get('temperatures')}",
            flush=True,
        )
    return out


# =============================================================================== latency
def run_latency(backends, concs=(1, 8, 32, 64)):
    from_typed = load_typed("test", 60)
    one_q = {
        "model": "jev-latest",
        "state": TICKET_EN,
        "questions": {"churn_risk": TICKET_Q["churn_risk"]},
    }
    four_q = {"model": "jev-latest", "state": TICKET_EN, "questions": TICKET_Q}
    out = {}
    for be in backends:

        def fresh(body):
            b = json.loads(json.dumps(body))
            b["state"] = dict(
                b["state"], id=f"r{random.random()}"
            )  # a new state each time: no prompt cache hit on it
            return b

        for label, body in (("1q", one_q), ("4q", four_q)):
            for _ in range(3):
                post(be["url"], fresh(body))
            lat = sorted(post(be["url"], fresh(body))[2] for _ in range(30))
            out[f"{be['name']} {label}"] = dict(
                p50_ms=round(lat[15], 1), p90_ms=round(lat[27], 1)
            )
            print(
                f"LAT {be['name']:>8} {label:6} p50={lat[15]:.0f}ms p90={lat[27]:.0f}ms",
                flush=True,
            )
        lat = sorted(
            post(
                be["url"],
                {
                    "model": "jev-latest",
                    "state": c["state"],
                    "questions": c["questions"],
                },
            )[2]
            for c in from_typed
        )
        out[f"{be['name']} typed-case"] = dict(
            p50_ms=round(lat[len(lat) // 2], 1),
            p90_ms=round(lat[int(len(lat) * 0.9)], 1),
        )
        print(
            f"LAT {be['name']:>8} typed  p50={lat[len(lat) // 2]:.0f}ms (5 questions, ~700-char JSON state)",
            flush=True,
        )
        if be["serial"]:
            continue
        for conc in concs:
            n = max(64, conc * 6)
            pmap(lambda _: post(be["url"], fresh(four_q)), range(conc), conc)
            t = time.perf_counter()
            res = pmap(lambda _: post(be["url"], fresh(four_q)), range(n), conc)
            wall = time.perf_counter() - t
            lat = sorted(r[2] for r in res if r[0] == 200)
            row = dict(
                req_per_s=round(n / wall, 1),
                decisions_per_s=round(4 * n / wall, 1),
                p50_ms=round(lat[len(lat) // 2]),
                p90_ms=round(lat[int(len(lat) * 0.9)]),
                errors=sum(r[0] != 200 for r in res),
            )
            out[f"{be['name']} 4q c={conc}"] = row
            print(f"LAT {be['name']:>8} 4q c={conc:<3} {row}", flush=True)
    return out


# =============================================================================== CLI
def parse_backend(spec):
    name, _, rest = spec.partition("=")
    url, *flags = rest.split(",")
    url = url.rstrip("/")
    if not url.endswith("/v1/systemone"):
        url += "/v1/systemone"
    return dict(name=name, url=url, basic="basic" in flags, serial="serial" in flags)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="systemone bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--backend", action="append", required=True, help="name=url[,basic][,serial]"
    )
    ap.add_argument("--suite", default="features,public,typed,latency")
    ap.add_argument("--n", type=int, default=300, help="examples per public dataset")
    ap.add_argument(
        "--typed-n",
        type=int,
        default=0,
        help="typed-decisions test cases (0 = all 400)",
    )
    ap.add_argument(
        "--calib-n",
        type=int,
        default=300,
        help="typed-decisions train cases to fit temperatures on (0 = skip)",
    )
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument(
        "--calibration-dir",
        default=None,
        help="write fitted temperatures here, one file per backend",
    )
    ap.add_argument(
        "--out", default="results.json", help="merged into, per backend and suite"
    )
    ap.add_argument(
        "--think",
        type=int,
        default=0,
        help="add think: N to public/typed/latency requests",
    )
    a = ap.parse_args(argv)
    if a.think:
        EXTRA["think"] = a.think
    backends = [parse_backend(b) for b in a.backend]
    suites = a.suite.split(",")
    try:
        res = json.load(open(a.out))
    except (OSError, ValueError):
        res = {}
    if "features" in suites:
        saved = dict(EXTRA)
        EXTRA.clear()  # the feature cases set their own options
        rows = run_features(backends)
        EXTRA.update(saved)
        for be in backends:
            mine = [r for r in rows if r["backend"] == be["name"]]
            res.setdefault("features", {})[be["name"]] = mine
            scored = [r for r in mine if r["ok"] is not None]
            na = len(mine) - len(scored)
            print(
                f"FEATURES {be['name']}: {sum(r['ok'] for r in scored)}/{len(scored)}"
                + (f" ({na} n/a)" if na else ""),
                flush=True,
            )
    if "public" in suites:
        res.setdefault("public", {}).update(run_public(backends, a.n, a.conc))
    if "typed" in suites:
        res.setdefault("typed", {}).update(
            run_typed(backends, a.typed_n, a.calib_n, a.conc, a.calibration_dir)
        )
    if "latency" in suites:
        res.setdefault("latency", {}).update(run_latency(backends))
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()


# =============================================================================== report
JEV_PUBLISHED = {  # third-party published figures, not measured here (see the README)
    "typed": dict(
        accuracy=0.727, soft_accuracy=0.580, brier=0.148, ece=0.144, score_mae=0.391
    ),
    "ag_news": 0.910,
    "emotion": 0.480,
    "typed_case_p50_ms": 710,
}


def report(path):
    """Markdown tables from a results file written by `systemone bench`."""
    r = json.load(open(path))
    names = list(dict.fromkeys(list(r.get("typed", {})) + list(r.get("features", {}))))
    f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
    out = [
        "| backend | features | typed acc | soft acc | Brier (cal) | ECE raw → cal | score MAE (cal) "
        "| AG News | Emotion | SST-2 noul | SST-2 choice | 1q p50 | 4q p50 | typed case p50 | 4q req/s @32 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        feats = r.get("features", {}).get(n)
        fs = "–"
        if feats:
            sc = [x for x in feats if x["ok"] is not None]
            fs = f"{sum(x['ok'] for x in sc)}/{len(sc)}"
        t = r.get("typed", {}).get(n, {})
        raw, cal = t.get("raw", {}), t.get("calibrated") or {}
        pub = lambda d: (r.get("public", {}).get(f"{d}/{n}") or {}).get("acc")  # noqa: E731
        lat = lambda k: (r.get("latency", {}).get(f"{n} {k}") or {})  # noqa: E731
        out.append(
            f"| {n} | {fs} | {f(raw.get('accuracy'))} | {f(raw.get('soft_accuracy'))} | {f(cal.get('brier'))} "
            f"| {f(raw.get('ece'))} → {f(cal.get('ece'))} | {f(cal.get('score_mae'))} | {f(pub('ag_news'))} "
            f"| {f(pub('emotion'))} | {f(pub('sst2'))} | {f(pub('sst2_choice'))} | {f(lat('1q').get('p50_ms'), 0)} ms "
            f"| {f(lat('4q').get('p50_ms'), 0)} ms | {f(lat('typed-case').get('p50_ms'), 0)} ms "
            f"| {f(lat('4q c=32').get('req_per_s'), 1)} |"
        )
    j = JEV_PUBLISHED
    out.append(
        f"| *Jev 1.13 (published)* | – | *{j['typed']['accuracy']}* | *{j['typed']['soft_accuracy']}* "
        f"| *{j['typed']['brier']}* | *{j['typed']['ece']}* | *{j['typed']['score_mae']}* | *{j['ag_news']}* "
        f"| *{j['emotion']}* | – | – | – | – | *{j['typed_case_p50_ms']} ms* | – |"
    )
    return "\n".join(out)
