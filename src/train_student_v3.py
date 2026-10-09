#!/usr/bin/env python3
"""Train intuition student v3 — task 017 (reversible boost).

Deltas vs v2:
- data: + graded_v3.jsonl (~1k new murky-heavy reversible items)
- model: same d320/8L/8H cascade shape (12.45M) — no reason to change
- 8 epochs (booked knee at this corpus size)
- tokenizer v3 retrained on merged corpus
Honest comparison: v2 AND v3 evaluated on the SAME fresh stratified split.
"""
import json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = "/home/eileen/projects/intuition"
DATA = [f"{ROOT}/data/graded.jsonl", f"{ROOT}/data/graded_v2.jsonl", f"{ROOT}/data/graded_v3.jsonl"]
TOKDIR = f"{ROOT}/models/tokenizer_v3"
MODEL_OUT = f"{ROOT}/models/intuition_student_v3.pt"
V2_CKPT = f"{ROOT}/models/intuition_student_v2.pt"
TOKDIR_V2 = f"{ROOT}/models/tokenizer_v2"
SEED = 7
MAX_LEN = 64
BATCH = 64
EPOCHS = 8
LR = 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def H(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1)
    return float(-(p * np.log2(p)).sum())

rows = []
for path in DATA:
    if os.path.exists(path):
        rows += [json.loads(l) for l in open(path)]
seen = set(); dedup = []
for r in rows:
    k = r["text"]
    if k not in seen:
        seen.add(k); dedup.append(r)
rows = dedup
random.shuffle(rows)

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
for r in rows:
    r["input"] = "[STATE] " + STATE_PREFIX[r["family"]] + r["text"] + " [QUESTION] " + QUESTION[r["family"]]
    r["H"] = H(r["probs"])

from collections import defaultdict, Counter
by_strat = defaultdict(list)
for r in rows:
    by_strat[(r["family"], r["band"])].append(r)
train, test = [], []
for k, v in by_strat.items():
    random.shuffle(v)
    n_test = max(1, int(0.15 * len(v)))
    test += v[:n_test]; train += v[n_test:]
print(f"train {len(train)} test {len(test)} soft-test {sum(1 for r in test if r['H']>1.4)}", flush=True)
print("family counts:", dict(Counter(r['family'] for r in rows)), flush=True)
json.dump([{k: r[k] for k in ("text","family","band","probs")} for r in test],
          open(f"{ROOT}/data/test_split_v3.json","w"))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
os.makedirs(TOKDIR, exist_ok=True)
TOKFILE = f"{TOKDIR}/tokenizer.json"
if not os.path.exists(TOKFILE):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=[
        "[PAD]", "[UNK]", "[STATE]", "[QUESTION]"], show_progress=False)
    tok.train_from_iterator([r["input"] for r in rows], trainer)
    tok.save(TOKFILE)
from tokenizers import Tokenizer as T
tok = T.from_file(TOKFILE)
PAD = tok.token_to_id("[PAD]")
def encode(s):
    return tok.encode(s, add_special_tokens=False).ids[:MAX_LEN]

class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ids = encode(r["input"])
        ids = ids + [PAD] * (MAX_LEN - len(ids))
        w = 1.0 + 2.0 * (r["H"] > 1.0)
        return (torch.tensor(ids), torch.tensor(r["probs"], dtype=torch.float32),
                torch.tensor(w))

train_dl = DataLoader(DS(train), batch_size=BATCH, shuffle=True)
test_dl = DataLoader(DS(test), batch_size=128)

class Student(nn.Module):
    def __init__(self, vocab, d=320, L=8, h=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d)
        enc = nn.TransformerEncoderLayer(d, h, dim_feedforward=1280,
                                         batch_first=True, norm_first=True,
                                         activation="gelu")
        self.enc = nn.TransformerEncoder(enc, L)
        self.head = nn.Linear(d, 3)
    def forward(self, x):
        m = (x == PAD)
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.enc(h, src_key_padding_mask=m)
        g = (h * (~m).unsqueeze(-1)).sum(1) / (~m).sum(1, keepdim=True).clamp(min=1)
        return self.head(g)

def eval_families(model, tokr, test_rows, tag):
    pad = tokr.token_to_id("[PAD]")
    fam_kl = defaultdict(list); soft_kl = []
    model.eval()
    with torch.no_grad():
        for r in test_rows:
            e = tokr.encode(r["input"], add_special_tokens=False).ids[:MAX_LEN]
            ids = torch.tensor([e + [pad]*(MAX_LEN-len(e))]).to(DEV)
            p = torch.tensor(r["probs"]).to(DEV)
            logp = F.log_softmax(model(ids), -1)
            kl = F.kl_div(logp, p, reduction="none").sum(-1).item()
            fam_kl[r["family"]].append(kl)
            if r["H"] > 1.4: soft_kl.append(kl)
    print(f"--- {tag} per-family test KL (same split) ---", flush=True)
    for f, v in sorted(fam_kl.items()):
        print(f"family {f}: {sum(v)/len(v):.4f} (n={len(v)})", flush=True)
    print(f"overall: {sum(sum(v) for v in fam_kl.values())/sum(len(v) for v in fam_kl.values()):.4f}  soft(H>1.4): {sum(soft_kl)/len(soft_kl) if soft_kl else float('nan'):.4f}", flush=True)
    return {f: sum(v)/len(v) for f, v in fam_kl.items()}

# ---- v2 baseline on the SAME split (v2's own tokenizer) ----
if os.path.exists(V2_CKPT):
    tok2 = T.from_file(f"{TOKDIR_V2}/tokenizer.json")
    v2 = Student(tok2.get_vocab_size()).to(DEV)
    v2.load_state_dict(torch.load(V2_CKPT, map_location=DEV))
    eval_families(v2, tok2, test, "v2 baseline")

# ---- train v3 ----
model = Student(tok.get_vocab_size()).to(DEV)
nparams = sum(p.numel() for p in model.parameters())
print(f"params: {nparams/1e6:.2f}M device {DEV}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
                                            steps_per_epoch=len(train_dl))

def run_eval(dl):
    model.eval(); tot = 0.0; n = 0; soft_kl = []
    with torch.no_grad():
        for ids, probs, _ in dl:
            ids, probs = ids.to(DEV), probs.to(DEV)
            logp = F.log_softmax(model(ids), -1)
            kl = F.kl_div(logp, probs, reduction="none").sum(-1)
            tot += kl.sum().item(); n += len(kl)
            for i in range(len(kl)):
                if H(probs[i].cpu().numpy()) > 1.4:
                    soft_kl.append(kl[i].item())
    return tot / n, (sum(soft_kl) / len(soft_kl) if soft_kl else float("nan"))

t0 = time.time()
for ep in range(EPOCHS):
    model.train(); tl = 0.0
    for ids, probs, w in train_dl:
        ids, probs, w = ids.to(DEV), probs.to(DEV), w.to(DEV)
        logp = F.log_softmax(model(ids), -1)
        kl = F.kl_div(logp, probs, reduction="none").sum(-1)
        loss = (kl * w).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl += loss.item() * len(ids)
    kls, soft = run_eval(test_dl)
    print(f"ep {ep}: train_wKL {tl/len(train):.4f} test_KL {kls:.4f} softKL {soft:.4f} ({time.time()-t0:.0f}s)", flush=True)

eval_families(model, tok, test, "v3")

torch.save(model.state_dict(), MODEL_OUT)

dummy = torch.tensor([encode(test[0]["input"]) + [PAD]*(MAX_LEN-len(encode(test[0]["input"])))]).to(DEV)
torch.onnx.export(model, dummy, f"{ROOT}/models/intuition_student_v3.onnx",
                  input_names=["ids"], output_names=["logits"],
                  dynamo=False,
                  opset_version=17,
                  dynamic_axes={"ids": {0: "batch"}, "logits": {0: "batch"}})
print(f"saved {MODEL_OUT} + ONNX", flush=True)

# CPU latency probe
model_cpu = model.to("cpu").eval()
import time as T
ids = torch.tensor([encode(test[0]["input"]) + [PAD]*(MAX_LEN-len(encode(test[0]["input"])))])
for _ in range(5): model_cpu(ids)
t1 = T.time()
for _ in range(50): model_cpu(ids)
print(f"CPU latency (d320/8L, 1 thread): {(T.time()-t1)/50*1000:.1f} ms", flush=True)
