"""Compare the Fable 5.2 audit labels with the stored model verdicts.

Run from the repository root:

    python artifacts/spotcheck/fable/score_fable.py

Writes artifacts/spotcheck/fable/fable_audit.json with, per sheet and field,
the number of items, how many Fable labels disagree with the stored verdict,
the disagreement share with a Wilson 95% interval, and both labels' positive
rates. This is a model audit, not an author check.
"""

import csv
import json
import math
from pathlib import Path

SPOT = Path("artifacts/spotcheck")
LABELS = SPOT / "fable" / "labels"


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def compare(results, sheet, stored_rows, label_rows, key, pairs):
    labels = {key(r): r for r in label_rows}
    for stored_field, fable_field in pairs:
        n = disagree = stored_pos = fable_pos = 0
        missing = 0
        for r in stored_rows:
            lab = labels.get(key(r))
            s = (r.get(stored_field) or "").strip()
            if lab is None:
                missing += 1
                continue
            f = (lab.get(fable_field) or "").strip()
            if s not in ("0", "1") or f not in ("0", "1"):
                continue
            n += 1
            disagree += s != f
            stored_pos += s == "1"
            fable_pos += f == "1"
        results.append(
            {
                "sheet": sheet,
                "field": stored_field,
                "n": n,
                "disagree": disagree,
                "disagree_share": round(disagree / n, 4) if n else None,
                "wilson95": wilson(disagree, n),
                "stored_positive": stored_pos,
                "fable_positive": fable_pos,
                "missing_labels": missing,
            }
        )


def main():
    results = []
    judge_labels = []
    for name in (
        "judge_hotpotqa.csv",
        "judge_spiqa_pixels.csv",
        "judge_captions_publaynet.csv",
    ):
        judge_labels += read(LABELS / name)
    for path in sorted(SPOT.glob("judge_verdicts_*.csv")):
        rows = read(path)
        compare(
            results,
            path.stem.replace("judge_verdicts_", "judge_"),
            rows,
            judge_labels,
            lambda r: (r["set"], r["row"]),
            [("judge_verdict_1_correct", "fable_verdict")],
        )
    triple_rows = read(SPOT / "graph_triples.csv")
    triple_labels = read(LABELS / "graph_triples.csv")
    for corpus in ("hotpotqa", "spiqa", "publaynet"):
        compare(
            results,
            f"triples_{corpus}",
            [r for r in triple_rows if r["corpus"] == corpus],
            triple_labels,
            lambda r: (r["corpus"], r["row"]),
            [
                ("opus_entailment", "fable_entailment"),
                ("opus_canonical", "fable_canonical"),
                ("opus_relation_specific", "fable_relation_specific"),
                ("opus_provenance", "fable_provenance"),
            ],
        )
    leak_rows = [dict(r, flagged="1") for r in read(SPOT / "leakage_flagged.csv")]
    compare(
        results,
        "leakage_flagged",
        leak_rows,
        read(LABELS / "leakage.csv"),
        lambda r: (r["set"], r["question_id"]),
        [("flagged", "fable_real_leakage")],
    )
    review_rows = read(SPOT / "reviewed_questions.csv")
    review_labels = read(LABELS / "reviewed_questions.csv")
    for set_name in ("spiqa_multihop_cross", "publaynet_figures"):
        compare(
            results,
            f"review_{set_name}",
            [r for r in review_rows if r["set"] == set_name],
            review_labels,
            lambda r: (r["set"], r["question_id"]),
            [
                ("answerable", "fable_answerable"),
                ("unique", "fable_unique"),
                ("well_formed", "fable_well_formed"),
                ("leakage", "fable_leakage"),
            ],
        )
    out = SPOT / "fable" / "fable_audit.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for r in results:
        if r["n"]:
            print(
                f"{r['sheet']:34} {r['field']:24} disagree {r['disagree']:3}/{r['n']:<3}"
                f" {r['disagree_share']:.3f}  stored+ {r['stored_positive']:3} fable+ {r['fable_positive']:3}"
                f"  missing {r['missing_labels']}"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
