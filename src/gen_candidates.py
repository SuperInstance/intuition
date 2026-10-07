#!/usr/bin/env python3
"""Candidate generator v2 — robust, diverse, three families x three bands.

Fixes over v1 (which returned 0 candidates for 2 of 3 bands):
- qwen2.5:7b-instruct instead of qwen3.5:0.8b (0.8b went off-format)
- few-shot examples + explicit format contract
- topic-anchor rotation for diversity (each call seeds a different domain)
- robust parse: keeps the question text intact, strips only leading
  numbering via regex (v1's char-loop ate real text and could destroy
  lines so nothing survived the >15-char filter)
- retries with re-prompt on empty/short output
- global dedupe (case-folded)
"""
import json, re, sys, time, urllib.request, random

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

FAMILIES = {
    "load-bearing": {
        "role": "candidate QUESTIONS about designing agents, systems, or projects",
        "criterion": "whether the question is load-bearing for the design task",
        "example": "How does the discriminator avoid reward hacking by the builder?",
    },
    "coherence": {
        "role": "candidate STATEMENTS or claims (may contain subtle flaws)",
        "criterion": "whether the statement coheres logically",
        "example": "The cache was added to reduce latency, and latency increased, so the cache worked.",
    },
    "go-no-go": {
        "role": "candidate DECISIONS or proposed actions with context",
        "criterion": "whether the action should proceed, wait, or stop",
        "example": "Deploy the refactor Friday evening; tests pass locally but CI is down.",
    },
}

# topic anchors rotated per call to force diversity across the corpus
ANCHORS = ["a robot fleet", "a web service", "a game studio", "a research lab",
    "a fishing boat business", "an open-source library", "a hospital system",
    "a school", "a farm", "a music app", "a security team", "a data pipeline",
    "an AI agent team", "a hardware startup", "a city transit authority",
    "a journalism outlet", "a weather station network", "a bakery",
    "a spacecraft mission", "a warehouse", "a film production", "a bank",
    "an energy grid", "a mobile app", "a crypto exchange", "a museum",
    "a logistics company", "a climate research project", "a gym",
    "a publishing house", "a water utility", "an esports team",
    "a car repair shop", "a streaming platform", "a nuclear plant",
    "a small farm drone program", "a coding bootcamp", "a pharmacy",
    "a space telescope", "a rescue service"]

BAND_SPECS = {
    "clear_yes": "OBVIOUSLY on the positive end for {criterion}. Any expert would instantly agree.",
    "clear_no":  "OBVIOUSLY on the negative end for {criterion}. Trivial, irrelevant, silly, or clearly flawed.",
    "murky":     "DELIBERATELY BORDERLINE for {criterion}. Reasonable experts could disagree. Subtle, in-between, arguable. These are the most valuable — make most of your output like this if unsure.",
}

def build_prompt(fam, band, anchor, n):
    f = FAMILIES[fam]
    spec = BAND_SPECS[band].format(criterion=f["criterion"])
    return f"""Generate {n} diverse one-sentence items.

Setting for variety: {anchor}.
Item type: {f['role']}.
Target: each item should be {spec}

Format contract — follow EXACTLY:
- One item per line, each starting with "N. " where N is the line number.
- Each item is a SINGLE sentence under 30 words.
- No headings, no commentary, no blank lines between items.

Example of the right kind of item for this family:
"{f['example']}"

Begin now:"""

def _gen(prompt):
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.95, "top_p": 0.92, "num_predict": 600}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["response"]

def parse(text):
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^\s*(?:\d+|[Nn])[.)]\s*(.+)$", ln.strip())
        if not m:
            continue
        item = m.group(1).strip().strip('"')
        # skip meta-lines the model sometimes adds
        if item.lower().startswith(("here are", "sure,", "certainly", "note:")):
            continue
        if 25 <= len(item) <= 220:
            out.append(item)
    return out

def gen_batch(fam, band, n=12, tries=3):
    anchor = random.choice(ANCHORS)
    for attempt in range(tries):
        try:
            items = parse(_gen(build_prompt(fam, band, anchor, n)))
        except Exception as e:
            print(f"  gen err {e}", file=sys.stderr); time.sleep(2); continue
        if items:
            return items
        anchor = random.choice(ANCHORS)  # re-roll anchor on empty
    return []

if __name__ == "__main__":
    target_per_combo = int(sys.argv[1]) if len(sys.argv) > 1 else 340
    seen = set()
    corpus = []
    for fam in FAMILIES:
        for band in ("clear_yes", "clear_no", "murky"):
            got, calls = 0, 0
            while got < target_per_combo and calls < target_per_combo * 2:
                items = gen_batch(fam, band)
                calls += 1
                for it in items:
                    k = re.sub(r"\W+", " ", it.lower()).strip()
                    if k not in seen:
                        seen.add(k)
                        corpus.append({"text": it, "family": fam, "band": band})
                        got += 1
            print(f"{fam}/{band}: {got} unique ({calls} calls)", flush=True)
    random.shuffle(corpus)
    json.dump(corpus, open("/home/eileen/projects/intuition/data/candidates.json", "w"),
              indent=1)
    print(f"total {len(corpus)} -> data/candidates.json")
