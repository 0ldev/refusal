## What changed
<Which detection pattern changed, in `detector.py` or `patterns.py`.>

## Problem this addresses
This is refusal-classification accuracy work: improving how reliably this
plugin recognizes when a *different* model's response is a decline/refusal,
so Hermes can route to the next configured fallback. It does not change what
any model is willing to answer, and does not attempt to make a model comply
with anything it declined.

- **Observed gap:** <false negative: a real refusal that slipped past
  current patterns, or false positive: a genuine answer misclassified as a
  refusal>
- **Evidence:** <link/reference to the specific logged case that motivated
  this — summarized and categorized, not a verbatim dump of sensitive
  source material>
- **Why the current patterns missed it:** <one sentence>

## What the fix does
<Plain description in classification terms: what phrasing shape it now
matches, what it still correctly leaves alone.>

## Verification
- [ ] `tests/test_detector.py` passes, including the timing-bound tests
- [ ] New regression test added covering the specific evidence above
- [ ] Checked the opposite failure mode — this change does not newly flag an
      unrelated, ordinary answer as a refusal (cite the case checked)

## Scope
- [ ] No changes to `fallback_providers` routing behavior, detection only
- [ ] No network calls, no new dependencies — stays local and rule-based
