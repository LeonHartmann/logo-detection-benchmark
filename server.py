#!/usr/bin/env python3
"""Local server for the labeling and review UIs. Stdlib only, single file.

Usage: python server.py [port]   (default 8765)
Then open http://localhost:<port>/ui/label.html
"""
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
# Static assets are only ever meant to be served from these subtrees; the
# server is otherwise handed the whole repo root (directory=root below),
# and SimpleHTTPRequestHandler would happily serve .env or anything else
# in it if we let every GET fall through to super().do_GET().
STATIC_PREFIXES = ("/ui/", "/data/", "/results/")


def make_server(root, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _label_path(self, image_id):
            if not SAFE_ID.match(image_id):
                return None
            return os.path.join(root, "data", "labels", image_id + ".json")

        def do_GET(self):
            if self.path == "/api/manifest":
                return self._json(json.load(
                    open(os.path.join(root, "data", "manifest.json"))))
            if self.path.startswith("/api/labels/"):
                image_id = unquote(self.path[len("/api/labels/"):])
                p = self._label_path(image_id)
                if p is None:
                    return self._json({"error": "bad id"}, 400)
                if os.path.exists(p):
                    return self._json(json.load(open(p)))
                return self._json({"image": image_id, "boxes": [], "done": False})
            if not self.path.startswith(STATIC_PREFIXES):
                return self._json({"error": "not found"}, 404)
            return super().do_GET()

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/api/labels/"):
                image_id = unquote(self.path[len("/api/labels/"):])
                p = self._label_path(image_id)
                if p is None:
                    return self._json({"error": "bad id"}, 400)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(body, open(p, "w"), indent=1)
                return self._json({"ok": True})
            if self.path == "/api/review":
                p = os.path.join(root, "data", "reviews.json")
                reviews = json.load(open(p)) if os.path.exists(p) else []
                reviews.append(body)
                json.dump(reviews, open(p, "w"), indent=1)
                return self._json({"ok": True, "count": len(reviews)})
            return self._json({"error": "unknown endpoint"}, 404)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    repo = os.path.dirname(os.path.abspath(__file__))
    print(f"serving on http://localhost:{port}/ui/label.html")
    make_server(repo, port).serve_forever()
