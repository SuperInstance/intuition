#!/usr/bin/env python3
"""017 fair eval + ONNX export.

1. Reconstruct v2's exact train/test split (replicating train_student_v2_ep8.py
   seed/order logic on old data only) -> identify rows v2 trained on.
2. Evaluate v2 and v3 per-family on the v3 test split, reporting:
   (a) raw same-split table (v2 inflated on old-family rows it trained on)
   (b) FAIR table: only rows v2 never trained on
3. Export v3 ONNX (single file) + bit-parity check vs torch on sample rows.
"""
import json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

ROOT = "/home/eileen/projects/intuition"
SEED = 7
MAX_LEN = 64
DEV = "cpu"

def H(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1)
    return float(-(p * np.log2(p)).sum())

# --- reconstruct v2 train membership ---
rows_old = []
for path in (f"{ROOT}/data/graded.jsonl", f"{ROOT}/data/graded_v2.jsonl"):
    rows_old += [json.loads(l) for l in open(path)]
seen = set(); dedup = []
for r in rows_old:
    if r["text"] not in seen:
        seen.add(r["text"]); dedup.append(r)
rows_old = dedup
random.seed(SEED)
random.shuffle(rows_old)
by_strat = defaultdict(list)
for r in rows_old:
    by_strat[(r["family"], r["band"])].append(r)
v2_train_texts = set()
for k, v in by_strat.items():
    random.shuffle(v)  # NOTE: same RNG stream position as original script
    n_test = max(1, int(0.15 * len(v)))
    v2_train_texts |= {r["text"] for r in v[n_test:]}
print(f"reconstructed v2 train set: {len(v2_train_texts)} rows", flush=True)

# hmm: original script shuffled rows THEN built by_strat from shuffled rows;
# replicate exactly: by_strat built from shuffled rows_old, then each v shuffled again.
# The loop above shuffles rows_old once (seed 7), appends to by_strat, then shuffles
# each stratum — matching the original's call sequence.

STATE_PREFIX = {
    "load-bearing": "Design task: a discriminator agent judges a builder agent's work across rounds. Candidate question: ",
    "coherence": "Reviewing a candidate statement for logical soundness. Statement: ",
    "go-no-go": "Triaging a proposed action for a team. Proposed action: ",
    "reversible": "Assessing a proposed change for reversibility. Proposed change: ",
    "risk": "Assessing a proposed action's downside risk. Proposed action: ",
}
QUESTION = {
    "load-bearing": "How load-bearing is this question for the design task?",
    "coherence": "Does this statement cohere logically?",
    "go-no-go": "Should this action proceed?",
    "reversible": "Is this change reversible?",
    "risk": "What is the risk level of this action?",
}

test = json.load(open(f"{ROOT}/data/test_split_v3.json"))
for r in test:
    r["input"] = "[STATE] " + STATE_PREFIX[r["family"]] + r["text"] + " [QUESTION] " + QUESTION[r["family"]]
    r["H"] = H(r["probs"])

from tokenizers import Tokenizer as T
tok2 = T.from_file(f"{ROOT}/models/tokenizer_v2/tokenizer.json")
tok3 = T.from_file(f"{ROOT}/models/tokenizer_v3/tokenizer.json")

class Student(torch.nn.Module):
    def __init__(self, vocab, d=320, L=8, h=8, PAD=0):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = torch.nn.Embedding(MAX_LEN, d)
        enc = torch.nn.TransformerEncoderLayer(d, h, dim_feedforward=1280,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
        self.enc = torch.nn.TransformerEncoder(enc, L)
        self.head = torch.nn.Linear(d, 3)
    def forward(self, x):
        m = (x == 0)
        h = self.emb(x) + self.pos(torch.arange(x.size(1)))
        h = self.enc(h, src_key_padding_mask=m)
        g = (h * (~m).unsqueeze(-1)).sum(1) / (~m).sum(1, keepdim=True).clamp(min=1)
        return self.head(g)

v2 = Student(tok2.get_vocab_size()); v2.load_state_dict(torch.load(f"{ROOT}/models/intuition_student_v2.pt", map_location=DEV)); v2.eval()
v3 = Student(tok3.get_vocab_size()); v3.load_state_dict(torch.load(f"{ROOT}/models/intuition_student_v3.pt", map_location=DEV)); v3.eval()

def kl_rows(model, tokr, rows):
    pad = tokr.token_to_id("[PAD]")
    out = []
    with torch.no_grad():
        for r in rows:
            e = tokr.encode(r["input"], add_special_tokens=False).ids[:MAX_LEN]
            ids = torch.tensor([e + [pad]*(MAX_LEN-len(e))])
            p = torch.tensor(r["probs"])
            logp = F.log_softmax(model(ids), -1)
            out.append(F.kl_div(logp, p, reduction="none").sum(-1).item())
    return out

def table(rows, tag):
    print(f"--- {tag} (n={len(rows)}) ---", flush=True)
    res = {}
    for fam in sorted(set(r["family"] for r in rows)):
        sub = [r for r in rows if r["family"] == fam]
        k2 = kl_rows(v2, tok2, sub); k3 = kl_rows(v3, tok3, sub)
        res[fam] = (sum(k2)/len(k2), sum(k3)/len(k3), len(sub))
        print(f"{fam}: v2 {res[fam][0]:.4f}  v3 {res[fam][1]:.4f}  (n={len(sub)})", flush=True)
    return res

table(test, "RAW same-split (v2 trained on most old rows — v2 inflated)")
fair = [r for r in test if r["text"] not in v2_train_texts]
table(fair, "FAIR (rows v2 never trained on)")

# soft-point check on fair reversible rows
fair_rev_soft = [r for r in fair if r["family"] == "reversible" and r["H"] > 1.4]
if fair_rev_soft:
    k2 = kl_rows(v2, tok2, fair_rev_soft); k3 = kl_rows(v3, tok3, fair_rev_soft)
    print(f"fair reversible soft(H>1.4): v2 {sum(k2)/len(k2):.4f}  v3 {sum(k3)/len(k3):.4f}  (n={len(fair_rev_soft)})", flush=True)

# --- ONNX export ---
dummy = None
for r in test:
    e = tok3.encode(r["input"], add_special_tokens=False).ids[:MAX_LEN]
    if e: dummy = torch.tensor([e + [tok3.token_to_id('[PAD]')]*(MAX_LEN-len(e))]); break
torch.onnx.export(v3, dummy, f"{ROOT}/models/intuition_student_v3.onnx",
                  input_names=["ids"], output_names=["logits"],
                  dynamo=False, opset_version=17,
                  dynamic_axes={"ids": {0: "batch"}, "logits": {0: "batch"}})
import onnx
m = onnx.load(f"{ROOT}/models/intuition_student_v3.onnx")
onnx.checker.check_model(m)
exts = [e.name for e in m.graph.initializer if e.data_location == 1]
print(f"ONNX ok, external data files: {exts or 'NONE (single file)'}", flush=True)

import onnxruntime as ort, time as T
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(f"{ROOT}/models/intuition_student_v3.onnx", so)
ids_np = dummy.numpy()
t1 = T.time()
for _ in range(50): sess.run(None, {"ids": ids_np})
print(f"ONNX CPU latency (1 thread): {(T.time()-t1)/50*1000:.1f} ms", flush=True)
# bit-parity spot check
with torch.no_grad():
    tl = F.log_softmax(v3(dummy), -1).numpy()
ol = sess.run(None, {"ids": ids_np})[0]
print(f"parity max|Δlogit|: {abs(tl - ol).max():.2e} (softmax-space {'OK' if abs(tl-ol).max() < 5e-3 else 'DRIFT'})", flush=True)
