# Intuition: design note (2026-10-06, Casey)

Not distillation for its own sake. The target is the specific intuition
soft point that is hard to compose from logic.

- The teacher's CERTAINTY is just logic (clear yes / clear no). Not the target.
- The teacher's SPREAD (high-entropy distributions, e.g. 0.4/0.35/0.25)
  is the intuition — the gut sense that can't be derived from rules.
- Training: generate candidates across a deliberate difficulty gradient
  (clear yes, clear no, murky middle). Teacher grades all. Loss is
  weighted toward high-entropy teacher distributions — the soft points.
- Evaluation is on the soft points specifically, not the easy cases.
  Matching the argmax on clear cases proves nothing; matching the
  *shape of uncertainty* on hard cases is the whole game.
- The student never sees rules. It only ever sees: text in, distribution
  out. What transfers is taste, not logic.

Quilt patches (each a soft-point family):
1. load-bearing — is this question/judgment carrying weight?
2. coherence — does this follow, or is something off?
3. go/no-go — should this proceed, stop, or wait?

## Use shape (2026-10-06, Casey)

Flashlight, not a wire. Waveform projection, not a bullet.

The intuition is not a pointwise classifier (is THIS load-bearing?
yes/no — that's a bullet at a target, a wire to a point). It is a field
instrument: run it across many candidates, read the *pattern* of its
responses. The spread of scores across the batch is the illumination —
it shows you where to look, not what to conclude.

- Deployment = batch + spectrum, not single-shot verdict.
- Score N candidates; the shape of the distribution is the tell.
- The human (or next agent) reads the lit landscape and decides.
- A bullet is right or wrong. A flashlight can't miss — it can only
  show you what's there.
