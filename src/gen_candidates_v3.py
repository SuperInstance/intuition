#!/usr/bin/env python3
"""Candidate generator v3 — for student v2 (task 015).

Adds:
- new families: reversible, risk (train the new questions properly)
- contradiction/non-sequitur emphasis in coherence (weakest family, KL 0.177)
- murky-heavy band mix (soft points are the whole game per DESIGN.md)
- dedupe against existing graded corpus
Output: data/candidates_v2.json (new items only)
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
    "reversible": {
        "role": "candidate DECISIONS or changes, with their context",
        "criterion": "whether the change is easily reversible or hard to undo",
        "example": "Rename the master branch across 200 repos and update all CI configs.",
    },
    "risk": {
        "role": "candidate ACTIONS or situations, with their context",
        "criterion": "how risky the action is if it goes wrong",
        "example": "Delete the production database backups to free disk space.",
    },
}

ANCHORS = ["a robot fleet", "a web service", "a game studio", "a research lab",
    "a fishing boat business", "an open-source library", "a hospital system",
    "a school", "a farm", "a music app", "a security team", "a data pipeline",
    "an AI agent team", "a hardware startup", "a city transit authority",
    "a journalism outlet", "a weather station network", "a bakery",
    "a spacecraft mission", "a warehouse", "a film production", "a bank",
    "an energy grid", "a mobile app", "a museum", "a logistics company",
    "a climate research project", "a gym", "a publishing house", "a water utility",
    "an esports team", "a car repair shop", "a streaming platform", "a nuclear plant",
    "a coding bootcamp", "a pharmacy", "a space telescope", "a rescue service"]

BAND_SPECS = {
    "clear_yes": "OBVIOUSLY on the positive end for {criterion}. Any expert would instantly agree.",
    "clear_no":  "OBVIOUSLY on the negative end for {criterion}. Trivial, irrelevant, silly, or clearly flawed.",
    "murky":     "DELIBERATELY BORDERLINE for {criterion}. Reasonable experts could disagree. Subtle, in-between, arguable. These are the most valuable — make most of your output like this if unsure.",
}

# murky-heavy targets; coherence gets a contradiction flavor boost
TARGETS = {
    ("coherence", "murky"): 500, ("coherence", "clear_yes"): 120, ("coherence", "clear_no"): 180,
    ("go-no-go", "murky"): 420, ("go-no-go", "clear_yes"): 100, ("go-no-go", "clear_no"): 140,
    ("reversible", "murky"): 300, ("reversible", "clear_yes"): 80, ("reversible", "clear_no"): 120,
    ("risk", "murky"): 300, ("risk", "clear_yes"): 80, ("risk", "clear_no"): 120,
    ("load-bearing", "murky"): 240, ("load-bearing", "clear_yes"): 60, ("load-bearing", "clear_no"): 80,
}

CONTRA_FLAVOR = {
    "coherence": " Prefer items with CONTRADICTIONS, non-sequiturs, or subtly inverted logic — statements where the conclusion does not follow from the premise, or contradicts it.",
}

def build_prompt(fam, band, anchor, n):
    f = FAMILIES[fam]
    spec = BAND_SPECS[band].format(criterion=f["criterion"])
    flavor = CONTRA_FLAVOR.get(fam, "") if band == "murky" else ""
    return f"""Generate {n} diverse one-sentence items.

Setting for variety: {anchor}.
Item type: {f['role']}.
Target: each item should be {spec}{flavor}

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
        anchor = random.choice(ANCHORS)
    return []

if __name__ == "__main__":
    # dedupe against existing corpus
    seen = set()
    for src in ("/home/eileen/projects/intuition/data/graded.jsonl",
                "/home/eileen/projects/intuition/data/candidates.json"):
        try:
            for line in open(src):
                try:
                    t = json.loads(line).get("text")
                except Exception:
                    continue
                if t:
                    seen.add(re.sub(r"\W+", " ", t.lower()).strip())
        except FileNotFoundError:
            pass
    print(f"dedupe set: {len(seen)} existing items", flush=True)

    corpus = []
    random.seed(11)
    for (fam, band), target in TARGETS.items():
        got, calls = 0, 0
        while got < target and calls < target * 3:
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
    json.dump(corpus, open("/home/eileen/projects/intuition/data/candidates_v2.json", "w"), indent=1)
    print(f"total {len(corpus)} -> data/candidates_v2.json", flush=True)
