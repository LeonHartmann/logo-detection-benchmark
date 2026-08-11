import json
import threading
import urllib.error
import urllib.request

import server as srv


def _start(tmp_path):
    (tmp_path / "data" / "labels").mkdir(parents=True)
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(
        {"images": [{"id": "a.jpg", "native": [854, 480], "stratum": "n", "source": {}}]}))
    httpd = srv.make_server(str(tmp_path), 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def _post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_manifest_labels_roundtrip_and_review(tmp_path):
    httpd, base = _start(tmp_path)
    try:
        assert _get(base + "/api/manifest")["images"][0]["id"] == "a.jpg"
        empty = _get(base + "/api/labels/a.jpg")
        assert empty == {"image": "a.jpg", "boxes": [], "done": False}
        label = {"image": "a.jpg", "labeler": "leon", "done": True,
                 "boxes": [{"brand": "adidas", "box": [1, 2, 3, 4], "size": "small",
                            "placement": "foreground", "location": "chest"}]}
        assert _post(base + "/api/labels/a.jpg", label)["ok"]
        on_disk = json.load(open(tmp_path / "data" / "labels" / "a.jpg.json"))
        assert on_disk == label
        assert _get(base + "/api/labels/a.jpg") == label
        _post(base + "/api/review", {"entry": {"model": "m1"}, "verdict": "model_right"})
        _post(base + "/api/review", {"entry": {"model": "m2"}, "verdict": "truth_wrong"})
        reviews = json.load(open(tmp_path / "data" / "reviews.json"))
        assert len(reviews) == 2 and reviews[1]["verdict"] == "truth_wrong"
    finally:
        httpd.shutdown()


def test_path_traversal_rejected(tmp_path):
    httpd, base = _start(tmp_path)
    try:
        try:
            _get(base + "/api/labels/..%2F..%2Fetc%2Fpasswd")
            assert False, "should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()
