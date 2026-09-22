"""Arcade and studio pages served by `systemone serve --demo`.

See README.md in this folder for setup.
"""

import os

WEB = os.path.join(os.path.dirname(__file__), "web")
PAGES = {"/arcade": "arcade.html", "/studio": "studio.html"}


def try_get(handler) -> bool:
    """Serve /arcade or /studio. True if this request was a demo page."""
    page = PAGES.get(handler.path.split("?")[0].rstrip("/"))
    if not page:
        return False
    body = open(os.path.join(WEB, page), "rb").read()
    handler.send_response(200)
    handler.send_header("content-type", "text/html; charset=utf-8")
    handler.send_header("content-length", str(len(body)))
    handler.send_header("cache-control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)
    return True
