"""Batched teacher client v2: raw candidate as question, state carries the judgment."""
import json, urllib.request, time, os

TSAFE = "https://api.typesafe.ai/v1/systemone"
TOKEN_FILE = os.path.expanduser("~/.config/typesafe/token")

def _post(payload, timeout=120):
    tok = open(TOKEN_FILE).read().strip()
    data = json.dumps(payload).encode()
    req = urllib.request.Request(TSAFE, data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + tok},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def grade_batch(state, items, criteria, batch_size=20, pause=1.0):
    out = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i+batch_size]
        questions = {k: {"type": "score", "question": q, "criteria": criteria}
                     for k, q in chunk}
        try:
            r = _post({"model": "jev-preview", "state": state, "questions": questions})
            for k, _ in chunk:
                a = r["answers"][k]
                p = a["probabilities"]
                out[k] = ([p["0"], p["1"], p["2"]], a["score"])
        except Exception as e:
            print(f"batch {i} failed: {e}", flush=True)
        time.sleep(pause)
    return out
