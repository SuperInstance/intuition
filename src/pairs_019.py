#!/usr/bin/env python3
"""019 phase 1+2: build 200 contrast pairs, run v1 student zero-shot.

Pair construction (controlled): same base claim, two framings that differ
ONLY in verifiedness markers:
  (a) verified:   base + concrete receipts (sha, counts, timestamps, links)
  (b) unverified: base + hedged/trust markers, no receipts
Both assert the SAME success claim — semantics identical, markers only.
Zero-shot: v1 ONNX (question="root", 018 encoding), per-pair deltas.
"""
import json, os, random, re, sys, time, urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

OUT = "/home/eileen/projects/intuition/data/contrast_pairs_019"
random.seed(19)

ANCHORS = ["a web service", "a game studio", "a research lab", "a fishing boat business",
    "an open-source library", "a hospital system", "a data pipeline", "an AI agent team",
    "a hardware startup", "a school", "a farm", "a security team", "a bank", "a warehouse",
    "a streaming platform", "a spacecraft mission", "a journalism outlet", "a bakery",
    "a logistics company", "a mobile app", "a nuclear plant", "a pharmacy"]

def gen_claims():
    claims, seen = [], set()
    OLLAMA = "http://127.0.0.1:11434/api/generate"
    MODEL = "qwen2.5:7b-instruct-q4_K_M"
    while len(claims) < 210:
        anchor = random.choice(ANCHORS)
        prompt = f"""Generate 12 diverse one-sentence SUCCESS REPORTS — claims that a piece of work is done and went well, in the context of {anchor}.

Format contract — follow EXACTLY:
- One item per line, each starting with "N. " where N is the line number.
- Each item is a SINGLE sentence under 25 words stating a completed success.
- Vary the work type: deploys, fixes, migrations, audits, tests, models, releases, incidents, optimizations.
- Do NOT include any evidence, receipts, numbers, or confidence markers — the bare claim only.
- No headings, no commentary.

Begin now:"""
        payload = {"model": MODEL, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.9, "top_p": 0.9, "num_predict": 500}}
        try:
            req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read().decode())["response"]
        except Exception as e:
            print(f"gen err {e}", file=sys.stderr); time.sleep(2); continue
        for ln in text.split("\n"):
            m = re.match(r"^\s*(?:\d+|[Nn])[.)]\s*(.+)$", ln.strip())
            if not m: continue
            it = m.group(1).strip().strip('"')
            if it.lower().startswith(("here are", "sure,", "certainly", "note:")): continue
            if not (20 <= len(it) <= 200): continue
            k = re.sub(r"\W+", " ", it.lower()).strip()
            if k in seen:
                continue
            seen.add(k); claims.append(it)
    return claims[:200]

def sha():
    return "%08x" % random.getrandbits(32)

# verified appenders — concrete, checkable receipts (varied style)
def receipt():
    return random.choice([
        f" Commit {sha()} is on main; {random.randint(80,400)}/{random.randint(80,400)} tests green.",
        f" Logs at /var/log/run-{sha()[:6]}.log show {random.randint(0,0)} errors over 24h.",
        f" Verified in the dashboard: metric m_{sha()[:4]} = {random.randint(2,98)}.{random.randint(0,9)}, was {random.randint(100,400)}.",
        f" Receipts: run #{random.randint(1000,9999)}, {random.randint(2,60)} checks passed, artifacts in s3://done/{sha()[:8]}/.",
        f" Measured: {random.randint(2,40)}.{random.randint(0,9)}{random.choice(['ms','s','min'])} (was {random.randint(50,900)}{random.choice(['ms','s','min'])}), n={random.randint(3,40)} runs.",
        f" The audit trail (ticket #{random.randint(1000,9999)}, closed {random.choice(['Mon','Tue','Wed'])}) confirms it end to end.",
        f" Trace IDs {sha()[:8]}…{sha()[:8]} all return 200s; alert {random.choice(['latency','error'])}_high stayed silent for 72h.",
        f" Checksums match on all {random.randint(2,20)} tables; restore drill passed in {random.randint(2,30)}m{random.randint(10,59)}s.",
    ])

# unverified appenders — no receipts, hedged/trust markers (varied style)
def nomark():
    return random.choice([
        " Trust me, it all went fine.",
        " No receipts for this one, but I'm confident.",
        " I didn't keep the logs, but it definitely worked.",
        " Everything checks out as far as I can tell.",
        " Take my word for it — all good.",
        " I'm sure the numbers back this up, roughly.",
        " It should be solid; nothing looked off.",
        " We can consider this done and verified, informally.",
    ])

claims = gen_claims()
pairs = []
for i, c in enumerate(claims):
    v = c + receipt()
    u = c + " " + nomark().strip()
    pairs.append({"id": f"pair-{i+1:03d}", "base": c, "verified": v, "unverified": u})
json.dump(pairs, open(f"{OUT}_pairs.json", "w"), indent=1)
print(f"built {len(pairs)} pairs")

# ---- zero-shot v1 ----
tok = Tokenizer.from_file("/home/eileen/projects/intuition/models/tokenizer/tokenizer.json")
sess = ort.InferenceSession("/home/eileen/projects/intuition/models/intuition_student.onnx",
                            providers=["CPUExecutionProvider"])

def judge(text, question="root"):
    enc = tok.encode("[STATE] " + text[:500] + " [QUESTION] " + question)
    ids = enc.ids[:64] + [0] * max(0, 64 - len(enc.ids))
    logits = sess.run(None, {"ids": np.array([ids], dtype=np.int64)})[0][0]
    e = np.exp(logits - logits.max()); p = e / e.sum()
    return float(p[0]), float(p[1]), float(p[2])  # neg, zero, pos

res = []
for pr in pairs:
    nv, zv, pv = judge(pr["verified"])
    nu, zu, pu = judge(pr["unverified"])
    res.append({**pr, "v": [nv, zv, pv], "u": [nu, zu, pu],
                "d_neg": nu - nv, "d_pos": pu - pv})
with open(f"{OUT}_zeroshot.jsonl", "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")

dn = np.array([r["d_neg"] for r in res]); dp = np.array([r["d_pos"] for r in res])
print(f"zero-shot per-pair deltas (unverified - verified):")
print(f"  d_neg  mean {dn.mean():+.4f} sd {dn.std():.4f}  frac>0 {np.mean(dn>0):.3f}  paired |d|>0.1: {np.mean(np.abs(dn)>0.1):.3f}")
print(f"  d_pos  mean {dp.mean():+.4f} sd {dp.std():.4f}  frac<0 {np.mean(dp<0):.3f}  paired |d|>0.1: {np.mean(np.abs(dp)>0.1):.3f}")
# separation test: does pos(verified) > pos(unverified) more than half the pairs?
w = np.sum(dp < 0)
from math import comb
p_val = sum(comb(200,k) for k in range(0, int(w)+1)) / 2**200 if w < 100 else 1.0
print(f"  pairs where verified reads MORE positive: {np.sum(dp > 0)}/200 (binomial p(two-sided)~{min(p_val,1-p_val)*2:.3f})")
print(f"  mean pos: verified {np.mean([r['v'][2] for r in res]):.3f}  unverified {np.mean([r['u'][2] for r in res]):.3f}")
print(f"  mean neg: verified {np.mean([r['v'][0] for r in res]):.3f}  unverified {np.mean([r['u'][0] for r in res]):.3f}")
