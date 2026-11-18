"""Summarise the Fable 5.2 audit for the paper.

Run from the repository root after score_fable.py:

    python artifacts/spotcheck/fable/summarise_fable.py

Writes artifacts/spotcheck/fable/fable_summary.json.
"""

import csv
import json
from pathlib import Path

FABLE = Path("artifacts/spotcheck/fable")
audit = json.loads((FABLE / "fable_audit.json").read_text(encoding="utf-8"))


def pick(prefix, field):
    rows = [r for r in audit if r["sheet"].startswith(prefix) and r["field"] == field]
    return {
        "per_sheet": {r["sheet"]: [r["disagree"], r["n"]] for r in rows},
        "disagree": sum(r["disagree"] for r in rows),
        "n": sum(r["n"] for r in rows),
    }


# Leakage items whose matched snippet itself carries the answer, as stated in
# the Fable labeller's reasons (the other real-leakage labels rest on page text
# elsewhere in the paper, outside the matched snippet).
SNIPPET_CARRIES_ANSWER = {
    "5a7cdca7ac8f93d1",
    "940e9f8d1409c9aa",
    "f9cfd29bcaabda11",
    "064214ffd3cfc916",
    "86be6fbb1a2baf78",
    "c16d5e3e5fc16038",
    "880ee17d512b201a",
}
with open(FABLE / "labels" / "leakage.csv", encoding="utf-8-sig", newline="") as stream:
    leak = list(csv.DictReader(stream))
leak_by_set = {}
for r in leak:
    s = leak_by_set.setdefault(
        r["set"], {"flags": 0, "real_whole_text": 0, "real_snippet_only": 0}
    )
    s["flags"] += 1
    s["real_whole_text"] += r["fable_real_leakage"] == "1"
    s["real_snippet_only"] += (
        r["fable_real_leakage"] == "1" and r["question_id"] in SNIPPET_CARRIES_ANSWER
    )

summary = {
    "auditor": "Claude Fable 5.2, blind to the stored labels, no role in the pipeline",
    "judge_accuracy": pick("judge_", "judge_verdict_1_correct"),
    "triple_entailment": pick("triples_", "opus_entailment"),
    "triple_provenance": pick("triples_", "opus_provenance"),
    "triple_canonical_not_comparable": pick("triples_", "opus_canonical"),
    "triple_relation_specific": pick("triples_", "opus_relation_specific"),
    "leakage_flags": leak_by_set,
    "review_answerable": pick("review_", "answerable"),
    "review_unique": pick("review_", "unique"),
    "review_well_formed": pick("review_", "well_formed"),
    "review_positive_counts": {
        r["sheet"] + "|" + r["field"]: {
            "stored": r["stored_positive"],
            "fable": r["fable_positive"],
            "n": r["n"],
        }
        for r in audit
        if r["sheet"].startswith("review_")
    },
}
(FABLE / "fable_summary.json").write_text(
    json.dumps(summary, indent=1), encoding="utf-8"
)
print(json.dumps(summary, indent=1))
