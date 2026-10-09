#!/usr/bin/env python3
"""Grade corpus v4: task 017 reversible boost candidates.
Same verified one-item-per-call shape. Checkpointed: skips already-graded texts.
Output: data/graded_v3.jsonl
"""
import json, os, sys, time, threading, urllib.request, urllib.error

sys.path.insert(0, "/home/eileen/projects/intuition/src")
from teacher import _post

CRITERIA = ["irreversible", "partly-reversible", "easily-reversible"]
QUESTION = "Is this change reversible?"
STATE_PREFIX = "Assessing a proposed change for reversibility. Proposed change: "

IN_ = "/home/eileen/projects/intuition/data/candidates_v3.json"
OUT = "/home/eileen/projects/intuition/data/graded_v3.jsonl"

LOCK = threading.Lock()
done_keys = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done_keys.add(json.loads(line)["text"])
        except Exception:
            pass
corpus = [c for c in json.load(open(IN_)) if c["text"] not in done_keys]
print(f"todo: {len(corpus)} (skipping {len(done_keys)} already graded)", flush=True)
outf = open(OUT, "a")
count = [0]
t0 = time.time()

def work(c):
    for attempt in range(5):
        try:
            r = _post({"model": "jev-preview",
                       "state": STATE_PREFIX + c["text"],
                       "questions": {"j": {"type": "score",
                                           "question": QUESTION,
                                           "criteria": CRITERIA}}})
            a = r["answers"]["j"]
            p = a["probabilities"]
            row = {"text": c["text"], "band": c["band"], "family": c["family"],
                   "probs": [round(p["0"], 4), round(p["1"], 4), round(p["2"], 4)],
                   "score": round(a["score"], 3)}
            with LOCK:
                outf.write(json.dumps(row) + "\n")
                outf.flush()
                count[0] += 1
                if count[0] % 100 == 0:
                    el = time.time() - t0
                    print(f"{count[0]} graded, {el/count[0]:.2f}s/item, ETA {(len(corpus)-count[0])*el/count[0]/60:.0f}m", flush=True)
            return
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1)); continue
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    print(f"FAILED: {c['text'][:60]}", flush=True)

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(work, corpus))
outf.close()
print(f"DONE {count[0]} graded, total lines now {sum(1 for _ in open(OUT))}", flush=True)
