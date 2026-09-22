"""POST /arcade/play: run maze or snake next to the model and stream one SSE per move."""

import json
import random
import time

from systemone.engine import decide

from . import arcade


def play(handler, req, registry):
    """Play one arcade game here, next to the model, streaming a server-sent event
    per move. Moves run at model speed; the viewer's distance only delays display."""
    game = req.get("game", "maze")
    maze = game == "maze"
    size = max(3, min(int(req.get("size") or (7 if maze else 10)), 12 if maze else 16))
    mode = (
        req.get("mode", "text")
        if req.get("mode") in ("text", "both", "image")
        else "text"
    )
    hints, think = bool(req.get("hints", True)), int(req.get("think") or 0)
    pace = max(0, min(int(req.get("pace_ms") or 0), 5000)) / 1000
    seed = int(req.get("seed") or random.randrange(1 << 30))
    model, calibration = registry.get(req.get("model"))
    g = (
        arcade.Maze(size, random.Random(seed))
        if maze
        else arcade.Snake(size, random.Random(seed))
    )
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-store")
    handler.send_header("x-accel-buffering", "no")
    handler.send_header("connection", "close")
    handler.end_headers()
    handler.close_connection = True

    def emit(obj):
        handler.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        handler.wfile.flush()

    try:
        start = {
            "type": "start",
            "game": game,
            "size": size,
            "seed": seed,
            "model": model.model,
            "engine": model.be.name
            + (f" {model.be.version}" if model.be.version else ""),
        }
        start.update(
            {"grid": ["".join(r) for r in g.g], "goal": g.goal}
            if maze
            else {"n": size}
        )
        emit(start)
        t_game = time.perf_counter()
        while not g.done:
            t0 = time.perf_counter()
            opts = g.legal() if maze else g.safe()
            if not opts:
                g.done, g.why = True, "trapped: no safe move"
                break
            if len(opts) == 1:
                choice, probs, ms = opts[0], {opts[0]: 1.0}, 0.0
            else:
                body = g.request(mode, hints)
                if think and mode == "text":
                    body["think"] = think
                res = decide(model, body, calibration)
                a = res["answers"]["move"]
                choice, probs, ms = (
                    a["choice"],
                    a["probabilities"],
                    res["diagnostics"]["timing"]["total_ms"],
                )
            g.move(choice)
            ev = {
                "type": "move",
                "i": g.moves,
                "choice": choice,
                "probs": probs,
                "ms": round(ms, 1),
                "forced": len(opts) == 1,
                "elapsed_s": round(time.perf_counter() - t_game, 3),
            }
            ev.update(
                {"pos": g.pos}
                if maze
                else {"body": g.body, "food": g.food, "score": g.score}
            )
            emit(ev)
            rest = pace - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
        emit(
            {
                "type": "end",
                "won": g.won,
                "moves": g.moves,
                "score": getattr(g, "score", None),
                "why": getattr(g, "why", "") or ("solved" if g.won else "move limit"),
                "elapsed_s": round(time.perf_counter() - t_game, 3),
            }
        )
    except (BrokenPipeError, ConnectionResetError):
        return  # the viewer stopped the game
    except Exception as e:
        try:
            emit({"type": "error", "message": str(e)})
        except OSError:
            pass
