#!/usr/bin/env python3
"""Grade the candidate corpus with the batched teacher, per family.

Family-specific judgment cells (jev-preview, score type over 3 levels):
- load-bearing: [not load-bearing, neutral, load-bearing]
- coherence:    [incoherent, unclear, coherent]
- go-no-go:     [no-go, wait, go]

Output: data/graded.jsonl lines {"text","band","family","probs":[p0,p1,p2],"score"}
"""
import json, sys, time
sys.path.insert(0, "/home/eileen/projects/intuition/src")
from teacher import grade_batch

CRITERIA = {
    "load-bearing": ["not load-bearing", "neutral", "load-bearing"],
    "coherence": ["incoherent", "unclear", "coherent"],
    "go-no-go": ["no-go", "wait", "go"],
}
QUESTION = {
    "load-bearing": "Is this item load-bearing for the design decision at hand?",
    "coherence": "Does this statement cohere logically?",
    "go-no-go": "Should this action proceed?",
}
STATE = {
    "load-bearing": "You are triaging candidate questions for a design review of a multi-agent system. For each question, judge how load-bearing it is for the design.",
    "coherence": "You are reviewing candidate statements for logical soundness. For each statement, judge whether it coheres.",
    "go-no-go": "You are triaging proposed actions for a team. For each action with its context, judge whether it should proceed.",
}

def main(inp, outp):
    corpus = json.load(open(inp))
    graded = []
    for fam in CRITERIA:
        items = [(i, c) for i, c in enumerate(corpus) if c["family"] == fam]
        # each question must carry its ITEM text — v1 sent the same generic
        # question 20x, so jev graded the family, not the items
        qitems = [(f"q{i}", f"{QUESTION[fam]}\nItem: {c['text']}")
                  for i, c in items]
        print(f"{fam}: grading {len(items)}...", flush=True)
        res = grade_batch(STATE[fam], qitems, CRITERIA[fam], batch_size=20, pause=1.0)
        misses = 0
        for i, c in items:
            r = res.get(f"q{i}")
            if r is None:
                misses += 1
                continue
            probs, score = r
            graded.append({"text": c["text"], "band": c["band"], "family": fam,
                           "probs": [round(p, 4) for p in probs], "score": round(score, 3)})
        print(f"{fam}: {len(items)-misses} graded, {misses} missed", flush=True)
    with open(outp, "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    print(f"wrote {len(graded)} -> {outp}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/eileen/projects/intuition/data/candidates.json",
         sys.argv[2] if len(sys.argv) > 2 else "/home/eileen/projects/intuition/data/graded.jsonl")
