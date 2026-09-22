"""The HTTP front: POST /v1/systemone, GET /health, GET /v1/systemone/info."""

import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import Calibration, decide
from .model import Model, Upstream, UpstreamError
from .schema import SchemaError


class Registry:
    """Every model the upstream servers serve (a base, its LoRA adapters, another vLLM
    with a different model...), probed on first use. A request's `model` picks one; an
    unknown name such as Jev's "jev-latest" gets the default."""

    def __init__(
        self,
        upstreams,
        default: str = None,
        calibration: str = None,
        calibration_dir: str = None,
    ):
        self.ups, self.lock = upstreams, threading.Lock()
        self.cal_path, self.cal_dir = calibration, calibration_dir
        self.cals, self.models, self.known = {}, {}, set()
        self.refresh()
        first = default or next(iter(self.served))
        if first not in self.served:
            raise UpstreamError(
                f"no upstream serves {first!r}; they serve {list(self.served)}"
            )
        self.default = Model(self.served[first], first)
        self.default.probe()
        self.models[first] = self.default

    def refresh(self):
        self.served = {}
        for up in self.ups:
            try:
                for m in up.get("/v1/models")["data"]:
                    self.served.setdefault(m["id"], up)
                    self.known.add(m["id"])
            except (UpstreamError, OSError):
                pass  # that server is down or still starting; try again on the next unknown name

    def calibration(self, name):
        if name not in self.cals:
            path = os.path.join(self.cal_dir, f"{name}.json") if self.cal_dir else None
            if not (path and os.path.exists(path)):
                path = self.cal_path if name == self.default.model else None
            self.cals[name] = Calibration(path)
        return self.cals[name]

    def get(self, name):
        if not name or name in self.models:
            m = self.models.get(name, self.default)
            return m, self.calibration(
                m.model
            )  # a probed model's server failing surfaces as a 502 on the request
        if name not in self.served:
            self.refresh()
            if name not in self.served:
                if (
                    name in self.known
                ):  # served before, its server is down or restarting: say so, don't substitute
                    raise UpstreamError(
                        f"model {name!r} is temporarily unavailable (its server is not answering)"
                    )
                return self.default, self.calibration(self.default.model)
        with self.lock:
            if name not in self.models:
                m = Model(self.served[name], name)
                m.probe()
                self.models[name] = m
        return self.models[name], self.calibration(name)

    def listing(self):
        self.refresh()
        return list(self.served)


def make_handler(registry: Registry, api_key: str, demo: bool = False):
    try_get = play = None
    if demo:
        from systemone.demo import try_get
        from systemone.demo.play import play

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if try_get and try_get(self):
                return
            if self.path == "/health":
                return self._json(200, {"ok": True})
            if self.path == "/v1/systemone/info":
                d = registry.default
                return self._json(
                    200,
                    dict(
                        d.info,
                        calibration=registry.calibration(d.model).path,
                        models=registry.listing(),
                    ),
                )
            self._json(404, {"error": {"message": "unknown route"}})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(n) if n else b""
            if self.path not in ("/v1/systemone", "/arcade/play") or (
                self.path == "/arcade/play" and not demo
            ):
                return self._json(404, {"error": {"message": "unknown route"}})
            if api_key and not hmac.compare_digest(
                self.headers.get("authorization", ""), f"Bearer {api_key}"
            ):
                return self._json(
                    401,
                    {
                        "error": {
                            "message": "missing or wrong API key",
                            "type": "authentication_error",
                        }
                    },
                )
            if self.path == "/arcade/play":
                return play(self, json.loads(raw or b"{}"), registry)
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise SchemaError("the body must be a JSON object")
                model, calibration = registry.get(body.get("model"))
                return self._json(200, decide(model, body, calibration))
            except json.JSONDecodeError as e:
                return self._json(
                    400,
                    {
                        "error": {
                            "message": f"invalid JSON: {e}",
                            "type": "invalid_request_error",
                        }
                    },
                )
            except SchemaError as e:
                return self._json(
                    422, {"error": {"message": str(e), "type": "validation_error"}}
                )
            except UpstreamError as e:
                return self._json(
                    502, {"error": {"message": str(e), "type": "upstream_error"}}
                )
            except Exception as e:  # keep serving
                return self._json(
                    500, {"error": {"message": repr(e), "type": "server_error"}}
                )

    return Handler


def serve(
    upstream,
    host: str = "0.0.0.0",
    port: int = 8011,
    model_name=None,
    calibration=None,
    upstream_key=None,
    api_key=None,
    calibration_dir=None,
    demo=False,
):
    urls = [upstream] if isinstance(upstream, str) else list(upstream)
    registry = Registry(
        [Upstream(u, upstream_key) for u in urls],
        model_name,
        calibration,
        calibration_dir,
    )
    info = registry.default.info
    api_key = (
        api_key if api_key is not None else os.environ.get("SYSTEMONE_API_KEY", "")
    )
    print("systemone:", json.dumps(info, ensure_ascii=False), flush=True)
    ThreadingHTTPServer.request_queue_size = 1024
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((host, port), make_handler(registry, api_key, demo))
    print(
        f"systemone: POST http://{host}:{port}/v1/systemone -> {', '.join(urls)} (default {info['model']}; "
        f"all of {list(registry.served)} selectable by the request's model field)",
        flush=True,
    )
    if demo:
        print(
            f"systemone: demos at http://{host}:{port}/arcade (maze, snake) and /studio (camera, motion, audio)",
            flush=True,
        )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
