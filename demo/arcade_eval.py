"""Score the arcade games headlessly: the same boards and prompts as web/arcade.html.

  python -m systemone.demo.arcade_eval --url http://127.0.0.1:8012 --model gemma4 --games 10

For each input mode it reports maze solve rate / moves and snake scores, next to two
baselines that see exactly the options the model sees:
  random     uniform over the offered moves
  heuristic  uses only what the hints spell out (maze: unvisited first; snake: towards
             the food, then most free space), i.e. what the scaffolding alone achieves
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from systemone.demo import arcade
from systemone.demo.arcade import Maze, Snake

TOKEN = ""


def ask(url, model, body):
    body = dict(body, model=model)
    headers = {"content-type": "application/json"}
    if TOKEN:
        headers["authorization"] = f"Bearer {TOKEN}"
    t = time.perf_counter()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), headers)
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d["answers"]["move"]["choice"], (time.perf_counter() - t) * 1000


def play(kind, seed, policy, args):
    rng = random.Random(seed)
    g = Maze(args.maze, rng) if kind == "maze" else Snake(args.snake, rng)
    prng, lat, calls = random.Random(seed + 1), [], 0
    while not g.done:
        opts = g.legal() if kind == "maze" else g.safe()
        if not opts:
            g.done, g.why = True, "trapped"
            break
        if len(opts) == 1:
            choice = opts[0]
        elif policy == "random":
            choice = prng.choice(opts)
        elif policy == "heuristic":
            choice = g.heuristic(prng)
        else:
            mode, hints = policy.split(":")
            choice, ms = ask(args.url, args.model, g.request(mode, hints == "hints"))
            lat.append(ms)
            calls += 1
        g.move(choice)
    return dict(won=g.won, moves=g.moves, score=getattr(g, "score", None), calls=calls,
                ms=statistics.median(lat) if lat else None)


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8012")
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--maze", type=int, default=7)
    ap.add_argument("--snake", type=int, default=10)
    ap.add_argument("--token", default="")
    ap.add_argument("--policies", default="random,heuristic,text:hints,text:plain,both:hints,image:hints")
    ap.add_argument("--out", default="")
    ap.add_argument("--parallel", type=int, default=0, help="games at once (default: all)")
    ap.add_argument("--layout", choices=["cached", "legacy"], default="cached")
    a = ap.parse_args()
    TOKEN = a.token
    arcade.LAYOUT = a.layout
    res = {}
    for kind in ("maze", "snake"):
        for pol in a.policies.split(","):
            with ThreadPoolExecutor(a.parallel or a.games) as pool:
                runs = list(pool.map(lambda s: play(kind, 1000 + s, pol, a), range(a.games)))
            if kind == "maze":
                row = dict(solved=f"{sum(r['won'] for r in runs)}/{len(runs)}",
                           moves_when_solved=statistics.median([r["moves"] for r in runs if r["won"]] or [0]))
            else:
                sc = [r["score"] for r in runs]
                row = dict(score_mean=round(statistics.mean(sc), 1), score_max=max(sc), moves_mean=round(statistics.mean(r["moves"] for r in runs)))
            ms = [r["ms"] for r in runs if r["ms"]]
            row["model_calls"] = sum(r["calls"] for r in runs)
            row["ms_per_move_p50"] = round(statistics.median(ms)) if ms else None
            res[f"{kind} {pol}"] = row
            print(f"{kind:5} {pol:12} {row}", flush=True)
    if a.out:
        json.dump(dict(model=a.model, maze_cells=a.maze, snake_board=a.snake, games=a.games, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
