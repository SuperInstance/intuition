#!/usr/bin/env python3
"""Grade corpus v2: one jev call per item (state carries the item).

Why: every batched variant tested (question-text items, array state,
shared question) collapsed to near-identical distributions regardless of
item content. The only verified-discriminating call shape is one item in
`state` + one question (the 003 battery pattern: jokes <=0.45, design
questions >=1.34). So we pay 1 call per item, 4 workers, checkpointed.
"""
import json, os, sys, time, threading, urllib.request, urllib.error

sys.path.insert(0, "/home/eileen/projects/intuition/src")
from teacher import _post

CRITERIA = {
    "load-bearing": ["not load-bearing", "neutral", "load-bearing"],
    "coherence": ["incoherent", "unclear", "coherent"],
    "go-no-go": ["no-go", "wait", "go"],
}
QUESTION = {
    "load-bearing": "How load-bearing is this question for the design task?",
    "coherence": "Does this statement cohere logically?",
    "go-no-go": "Should this action proceed?",
}
STATE_PREFIX = {
    "load-bearing": "Design task: a discriminator agent judges a builder agent's work across rounds. Candidate question: ",
    "coherence": "Reviewing a candidate statement for logical soundness. Statement: ",
    "go-no-go": "Triaging a proposed action for a team. Proposed action: ",
}

IN_ = "/home/eileen/projects/intuition/data/candidates.json"
OUT = "/home/eileen/projects/intuition/data/graded.jsonl"
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
                       "state": STATE_PREFIX[c["family"]] + c["text"],
                       "questions": {"j": {"type": "score",
                                           "question": QUESTION[c["family"]],
                                           "criteria": CRITERIA[c["family"]]}}})
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
