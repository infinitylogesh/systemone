"""Accuracy and latency of image, video and audio classification through /v1/systemone.

  python -m systemone.demo.media_eval --url http://127.0.0.1:8012 --models gemma4,gemma4-e4b --n 200

Suites (each one `choice` question per item, sent one at a time and in parallel):
  image   CIFAR-10 test images (10 classes), upscaled to 224 px
  video   synthetic 8-frame clips of a square moving left/right/up/down: the answer is
          only in the motion, not in any single frame (sent as video_frames)
  audio   ESC-10 (the 10-class subset of ESC-50 environmental sounds), 5 s clips at 16 kHz
Needs: datasets, numpy, pillow (image/video), soundfile + librosa (audio).
"""

import argparse
import base64
import io
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body):
    t = time.perf_counter()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        return None, (time.perf_counter() - t) * 1000, json.loads(e.read() or b"{}").get("error", {}).get("message", str(e))
    return d, (time.perf_counter() - t) * 1000, None


def jpeg_url(img, q=85):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def image_items(n):
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", split="test").shuffle(seed=0).select(range(n))
    names = ds.features["label"].names
    q = {"type": "choice", "instructions": "What is the main object in this photo?", "criteria": {k: None for k in names}}
    for ex in ds:
        yield {"state": "A small photo.", "images": [jpeg_url(ex["img"].convert("RGB").resize((224, 224)))],
               "questions": {"label": q}}, names[ex["label"]]


def video_items(n):
    from PIL import Image, ImageDraw

    dirs = ["left", "right", "up", "down"]
    q = {"type": "choice", "instructions": "Which way does the red square move during the clip (as seen in the video)?",
         "criteria": {d: None for d in dirs}}
    rng = random.Random(0)
    for i in range(n):
        d = dirs[i % 4]
        x0, y0 = rng.randint(70, 110), rng.randint(70, 110)
        bg = tuple(rng.randint(200, 255) for _ in range(3))
        frames = []
        for t in range(8):
            dx = {"left": -1, "right": 1}.get(d, 0) * t * 14
            dy = {"up": -1, "down": 1}.get(d, 0) * t * 14
            im = Image.new("RGB", (224, 224), bg)
            ImageDraw.Draw(im).rectangle([x0 + dx - 40 + 20, y0 + dy - 20, x0 + dx + 20, y0 + dy + 20], fill=(220, 40, 40))
            frames.append(jpeg_url(im))
        yield {"state": "A short clip; the frames are in time order.", "video_frames": frames, "fps": 8,
               "questions": {"motion": q}}, d


def audio_items(n):
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.filter(lambda ex: ex["esc10"]).cast_column("audio", Audio(sampling_rate=16000)).shuffle(seed=0)
    ds = ds.select(range(min(n, len(ds))))
    names = sorted(set(ds["category"]))
    q = {"type": "choice", "instructions": "Which sound is this?", "criteria": {k.replace("_", " "): None for k in names}}
    for ex in ds:
        buf = io.BytesIO()
        sf.write(buf, np.asarray(ex["audio"]["array"], dtype="float32"), 16000, format="WAV", subtype="PCM_16")
        yield {"state": "A 5-second sound recording.", "audio": ["data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()],
               "questions": {"label": q}}, ex["category"].replace("_", " ")


def run(url, model, suite, items, parallel):
    def one(item):
        body, gold = item
        d, ms, err = post(url, dict(body, model=model))
        if err:
            return None, gold, ms, None, err
        a = next(iter(d["answers"].values()))
        return a["choice"], gold, ms, d["diagnostics"]["timing"]["total_ms"], None

    items = list(items)
    post(url, dict(items[0][0], model=model))  # warm up
    seq = [one(it) for it in items[:15]]  # one at a time: latency
    with ThreadPoolExecutor(parallel) as pool:  # in parallel: throughput and accuracy
        t = time.perf_counter()
        res = list(pool.map(one, items))
        wall = time.perf_counter() - t
    errs = [r[4] for r in res if r[4]]
    ok = [r for r in res if not r[4]]
    out = dict(model=model, suite=suite, n=len(items),
               accuracy=round(sum(p == g for p, g, *_ in ok) / len(items), 3),
               errors=len(errs), first_error=errs[0] if errs else None,
               round_trip_ms_p50=round(statistics.median(r[2] for r in seq)),
               model_ms_p50=round(statistics.median(r[3] for r in seq if r[3] is not None)) if any(r[3] for r in seq) else None,
               items_per_s=round(len(items) / wall, 1), parallel=parallel)
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8012")
    ap.add_argument("--models", default="gemma4,gemma4-e4b")
    ap.add_argument("--suites", default="image,video,audio")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    make = {"image": image_items, "video": video_items, "audio": audio_items}
    results = []
    for suite in a.suites.split(","):
        items = list(make[suite](a.n))
        for model in a.models.split(","):
            results.append(run(a.url, model, suite, items, a.parallel))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
