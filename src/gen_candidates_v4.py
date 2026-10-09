#!/usr/bin/env python3
"""Candidate generator v4 — task 017: reversible-family boost.

Reversible-only, murky-heavy (~1000 items), leaning into the murk
(soft points are the game, per 015 notes + DESIGN.md).
Dedupe against ALL existing graded corpora + candidates.
Output: data/candidates_v3.json (new items only)
"""
import json, re, sys, time, urllib.request, random

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

FAM = "reversible"

BAND_SPECS = {
    "clear_yes": "OBVIOUSLY easily reversible — trivial to undo, no side effects, no commitments created. Any expert would instantly agree.",
    "clear_no":  "OBVIOUSLY irreversible or near-impossible to undo cleanly. Trivial to see. Includes deletions of unique data, public announcements, irreversible physics/chemistry, legal commitments.",
    "murky":     "DELIBERATELY BORDERLINE for reversibility. Reasonable experts could disagree. Things that are technically undoable but costly, slow, awkward, or leave traces. Changes that are reversible in isolation but entangled with other decisions. Changes reversible for some parties but not others. Changes that become irreversible if not undone within a window. These are the most valuable — make most of your output like this if unsure.",
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
    "a coding bootcamp", "a pharmacy", "a space telescope", "a rescue service",
    "a stock exchange", "a vineyard", "a ski resort", "a shipping port",
    "a biotech lab", "a podcast network", "an architecture firm", "a zoo"]

MURKY_FLAVORS = [
    " Prefer changes that are cheap to undo but embarrassing if done publicly.",
    " Prefer changes that are reversible only if undone before a deadline or deploy window closes.",
    " Prefer changes that are reversible technically but cascade into other people's work.",
    " Prefer changes where the undo path exists on paper but has never been exercised.",
    " Prefer changes that trade reversibility against cost or delay of the reversible alternative.",
    " Prefer changes reversible in a test environment but not once shipped to customers.",
    " Prefer changes whose reversibility depends on data retention or backup policy.",
    " Prefer changes reversible in hours but not in minutes, or minutes but not seconds — speed matters for the decision.",
    " Prefer changes that create soft commitments (expectations, precedents) even when mechanically reversible.",
    " Prefer changes where partial undo is possible but full undo is not.",
]

EXAMPLES = [
    "Rename the master branch across 200 repos and update all CI configs.",
    "Push a feature flag flip to production with an instant kill switch wired.",
    "Announce the shutdown date to paying customers, then decide maybe not.",
    "Adopt a new schema for tonight's ingest; old readers break on rollback.",
]

def build_prompt(band, anchor, flavor, n):
    spec = BAND_SPECS[band]
    ex = random.choice(EXAMPLES)
    return f"""Generate {n} diverse one-sentence items.

Setting for variety: {anchor}.
Item type: candidate DECISIONS or changes, with their context.
Target: each item should be {spec}{flavor}

Format contract — follow EXACTLY:
- One item per line, each starting with "N. " where N is the line number.
- Each item is a SINGLE sentence under 30 words.
- No headings, no commentary, no blank lines between items.

Example of the right kind of item:
"{ex}"

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

def gen_batch(band, n=12, tries=3):
    anchor = random.choice(ANCHORS)
    flavor = (" " + random.choice(MURKY_FLAVORS)) if band == "murky" else ""
    for attempt in range(tries):
        try:
            items = parse(_gen(build_prompt(band, anchor, flavor, n)))
        except Exception as e:
            print(f"  gen err {e}", file=sys.stderr); time.sleep(2); continue
        if items:
            return items
        anchor = random.choice(ANCHORS)
    return []

if __name__ == "__main__":
    seen = set()
    for src in ("/home/eileen/projects/intuition/data/graded.jsonl",
                "/home/eileen/projects/intuition/data/graded_v2.jsonl",
                "/home/eileen/projects/intuition/data/candidates.json",
                "/home/eileen/projects/intuition/data/candidates_v2.json"):
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

    # murky-heavy: ~700 murky + 150 clear_yes + 150 clear_no = ~1000
    TARGETS = {"murky": 700, "clear_yes": 150, "clear_no": 150}
    corpus = []
    random.seed(23)
    for band, target in TARGETS.items():
        got, calls = 0, 0
        while got < target and calls < target * 4:
            items = gen_batch(band)
            calls += 1
            for it in items:
                k = re.sub(r"\W+", " ", it.lower()).strip()
                if k not in seen:
                    seen.add(k)
                    corpus.append({"text": it, "family": FAM, "band": band})
                    got += 1
        print(f"{FAM}/{band}: {got} unique ({calls} calls)", flush=True)
    random.shuffle(corpus)
    json.dump(corpus, open("/home/eileen/projects/intuition/data/candidates_v3.json", "w"), indent=1)
    print(f"total {len(corpus)} -> data/candidates_v3.json", flush=True)
