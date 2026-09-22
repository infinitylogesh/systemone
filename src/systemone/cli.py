"""systemone serve | launch | bench"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from . import __version__


def _wait_health(url, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"vllm exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(url + "/health", timeout=5)
            return
        except Exception:
            time.sleep(3)
    sys.exit(f"vllm not healthy after {timeout}s")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="systemone",
        description="Jev's /v1/systemone on any LLM served by vLLM or SGLang.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "serve", help="serve /v1/systemone in front of running vLLM or SGLang servers"
    )
    s.add_argument(
        "--upstream",
        action="append",
        default=None,
        help="a vLLM or SGLang server's base URL (engine auto-detected); repeat to serve the models of several "
        "servers (default http://127.0.0.1:8000)",
    )
    s.add_argument(
        "--model",
        default=None,
        help="served model name (default: the upstream's first)",
    )
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8011)
    s.add_argument(
        "--calibration",
        default=None,
        help="temperatures JSON written by `systemone bench`",
    )
    s.add_argument(
        "--calibration-dir",
        default=None,
        help="a <served model name>.json per model (base, LoRAs)",
    )
    s.add_argument(
        "--demo",
        action="store_true",
        help="also serve /arcade and /studio (see demo/README.md)",
    )
    s.add_argument(
        "--upstream-api-key", default=os.environ.get("SYSTEMONE_UPSTREAM_API_KEY")
    )

    l_ = sub.add_parser(
        "launch",
        help="start `vllm serve MODEL` and the systemone server together",
        epilog="Arguments after `--` go to vllm serve, e.g. -- --gpu-memory-utilization 0.8",
    )
    l_.add_argument("model")
    l_.add_argument("--host", default="0.0.0.0")
    l_.add_argument("--port", type=int, default=8011)
    l_.add_argument("--vllm-port", type=int, default=8000)
    l_.add_argument("--calibration", default=None)

    sub.add_parser(
        "bench", help="conformance, quality and latency benchmark", add_help=False
    )
    sub.add_parser(
        "rlcd",
        help="RLCD fine-tuning: `rlcd prepare` then `rlcd train`",
        add_help=False,
    )
    rp = sub.add_parser(
        "report", help="markdown comparison table from a bench results file"
    )
    rp.add_argument("results")

    if argv is None:
        argv = sys.argv[1:]
    
    if argv[:1] == ["rlcd"]:
        sub_cmd = argv[1] if len(argv) > 1 else ""
        if sub_cmd == "prepare":
            from .rlcd.prepare import main as prepare_main

            return prepare_main(argv[2:])
        if sub_cmd == "train":
            from .rlcd.train import (
                main as train_main,
            )  # needs: pip install "systemone[rlcd]"

            return train_main(argv[2:])
        sys.exit("usage: systemone rlcd prepare|train ...")

    if argv[:1] == ["bench"]:
        from .bench import main as bench_main

        return bench_main(argv[1:])

    # everything after a bare `--` goes to vllm serve untouched
    vllm_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, vllm_args = argv[:i], argv[i + 1 :]
    a = ap.parse_args(argv)
    if a.cmd == "report":
        from .bench import report

        return print(report(a.results))
    from .server import serve

    if a.cmd == "serve":
        return serve(
            a.upstream or ["http://127.0.0.1:8000"],
            a.host,
            a.port,
            a.model,
            a.calibration,
            a.upstream_api_key,
            calibration_dir=a.calibration_dir,
            demo=a.demo,
        )

    # launch
    vllm = shutil.which("vllm") or sys.exit("vllm is not installed (pip install vllm)")

    cmd = [
        vllm,
        "serve",
        a.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(a.vllm_port),
        "--enable-prefix-caching",
        "--max-logprobs",
        "32",
    ] + vllm_args

    print("systemone: " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    signal.signal(signal.SIGTERM, lambda *_: (proc.terminate(), sys.exit(0)))

    try:
        _wait_health(f"http://127.0.0.1:{a.vllm_port}", proc, 3600)
        serve(f"http://127.0.0.1:{a.vllm_port}", a.host, a.port, None, a.calibration)
    finally:
        proc.terminate()
        proc.wait(60)


if __name__ == "__main__":
    main()
