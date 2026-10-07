#!/usr/bin/env python3
"""Train the intuition student — a laser rangefinder, not a classifier.

Teacher: graded.jsonl (3100 examples, soft ternary distributions).
Student: micro transformer encoder (d256 / 6L / 8H, ~8M params), CPU-ms infer.
Loss: KL vs teacher soft probs, weighted toward soft points (H>1.0).
Primary eval: KL on soft points (H>1.4) of a stratified held-out split.
"""
import json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = "/home/eileen/projects/intuition"
DATA = f"{ROOT}/data/graded.jsonl"
TOKDIR = f"{ROOT}/models/tokenizer"
MODEL_OUT = f"{ROOT}/models/intuition_student.pt"
SEED = 7
MAX_LEN = 64
BATCH = 64
EPOCHS = 12
LR = 3e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def H(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1)
    return float(-(p * np.log2(p)).sum())

# ---------- data ----------
rows = [json.loads(l) for l in open(DATA)]
random.shuffle(rows)
STATE_PREFIX = {
    "load-bearing": "Design task: a discriminator agent judges a builder agent's work across rounds. Candidate question: ",
    "coherence": "Reviewing a candidate statement for logical soundness. Statement: ",
    "go-no-go": "Triaging a proposed action for a team. Proposed action: ",
}
QUESTION = {
    "load-bearing": "How load-bearing is this question for the design task?",
    "coherence": "Does this statement cohere logically?",
    "go-no-go": "Should this action proceed?",
}
for r in rows:
    r["input"] = "[STATE] " + STATE_PREFIX[r["family"]] + r["text"] + " [QUESTION] " + QUESTION[r["family"]]
    r["H"] = H(r["probs"])

# stratified split by family x band
from collections import defaultdict
by_strat = defaultdict(list)
for r in rows:
    by_strat[(r["family"], r["band"])].append(r)
train, test = [], []
for k, v in by_strat.items():
    random.shuffle(v)
    n_test = max(1, int(0.15 * len(v)))
    test += v[:n_test]; train += v[n_test:]
print(f"train {len(train)} test {len(test)} soft-test {sum(1 for r in test if r['H']>1.4)}")

# ---------- tokenizer (BPE, ~8k) ----------
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
        w = 1.0 + 2.0 * (r["H"] > 1.0)   # soft points count 3x
        return (torch.tensor(ids), torch.tensor(r["probs"], dtype=torch.float32),
                torch.tensor(w))

train_dl = DataLoader(DS(train), batch_size=BATCH, shuffle=True)
test_dl = DataLoader(DS(test), batch_size=128)

# ---------- model ----------
class Student(nn.Module):
    def __init__(self, vocab, d=256, L=6, h=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d)
        enc = nn.TransformerEncoderLayer(d, h, dim_feedforward=1024,
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

model = Student(tok.get_vocab_size()).to(DEV)
nparams = sum(p.numel() for p in model.parameters())
print(f"params: {nparams/1e6:.2f}M device {DEV}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
                                            steps_per_epoch=len(train_dl))

def run_eval(dl):
    model.eval(); tot = 0.0; n = 0; soft_kl = []; hs_t, hs_s = [], []
    with torch.no_grad():
        for ids, probs, _ in dl:
            ids, probs = ids.to(DEV), probs.to(DEV)
            logp = F.log_softmax(model(ids), -1)
            kl = F.kl_div(logp, probs, reduction="none").sum(-1)
            tot += kl.sum().item(); n += len(kl)
            for i in range(len(kl)):
                p = probs[i].cpu().numpy()
                hp = H(p)
                hs_t.append(hp)
                q = F.softmax(model(ids[i:i+1]), -1)[0].cpu().numpy()
                hs_s.append(H(q))
                if hp > 1.4:
                    soft_kl.append(kl[i].item())
    corr = float(np.corrcoef(hs_t, hs_s)[0, 1]) if len(hs_t) > 2 else float("nan")
    return tot / n, (sum(soft_kl) / len(soft_kl) if soft_kl else float("nan")), corr

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
    kls, soft, corr = run_eval(test_dl)
    print(f"ep {ep}: train_wKL {tl/len(train):.4f} test_KL {kls:.4f} softKL {soft:.4f} Hcorr {corr:.3f} ({time.time()-t0:.0f}s)", flush=True)

# per-family eval
from collections import Counter
fam_kl = defaultdict(list)
model.eval()
with torch.no_grad():
    for ids, probs, _ in test_dl:
        ids, probs = ids.to(DEV), probs.to(DEV)
        logp = F.log_softmax(model(ids), -1)
        kl = F.kl_div(logp, probs, reduction="none").sum(-1)
        for i in range(len(kl)):
            pass
for r in test:
    with torch.no_grad():
        ids = torch.tensor([encode(r["input"]) + [PAD]*(MAX_LEN-len(encode(r["input"])))]).to(DEV)
        p = torch.tensor(r["probs"]).to(DEV)
        logp = F.log_softmax(model(ids), -1)
        fam_kl[r["family"]].append(F.kl_div(logp, p, reduction="none").sum(-1).item())
for f, v in fam_kl.items():
    print(f"family {f}: test KL {sum(v)/len(v):.4f} (n={len(v)})")

torch.save(model.state_dict(), MODEL_OUT)
# CPU latency probe (Oracle target: milliseconds)
model_cpu = model.to("cpu").eval()
import time as T
ids = torch.tensor([encode(test[0]["input"]) + [PAD]*(MAX_LEN-len(encode(test[0]["input"])))])
for _ in range(5): model_cpu(ids)  # warm
t1 = T.time()
for _ in range(50): model_cpu(ids)
print(f"CPU latency (d256/6L, 1 thread): {(T.time()-t1)/50*1000:.1f} ms")
print(f"saved {MODEL_OUT}")
