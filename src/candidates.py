"""Generate candidate questions across the difficulty gradient via local Ollama."""
import json, urllib.request

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3.5:0.8b"

PROMPTS = {
    "clear_yes": ("Generate 10 questions that are OBVIOUSLY load-bearing for building "
        "a discriminator agent that evaluates a builder agent. These should be "
        "questions any expert would agree are critical. One per line, numbered."),
    "clear_no": ("Generate 10 questions that are OBVIOUSLY NOT load-bearing for building "
        "a discriminator agent that evaluates a builder agent. Trivial, irrelevant, "
        "or silly questions. One per line, numbered."),
    "murky": ("Generate 10 questions that are ARGUABLY load-bearing for building "
        "a discriminator agent that evaluates a builder agent. Borderline cases — "
        "reasonable people could disagree. Subtle, debatable, in-between. "
        "One per line, numbered."),
}

def _gen(prompt, n=10):
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.8, "think": False}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read().decode())["response"]
    lines = []
    for ln in text.strip().split("\n"):
        ln = ln.strip()
        # strip leading "1. " / "1) " numbering
        while ln and (ln[0].isdigit() or ln[0] in ".)- "):
            ln = ln[1:].strip()
        if len(ln) > 15:
            lines.append(ln)
    return lines[:n]

if __name__ == "__main__":
    all_cands = {}
    for band, prompt in PROMPTS.items():
        cands = _gen(prompt)
        all_cands[band] = cands
        print(f"== {band}: {len(cands)}")
        for c in cands[:3]:
            print("  -", c[:80])
    json.dump(all_cands, open("/home/eileen/projects/intuition/data/candidates.json", "w"), indent=1)
    print("saved")
