"""Score exported spot-check labels against the stored verdicts.

Run from the repository root with one or more CSVs exported by labeller.html
(or written in the same long format by any labeller):

    python artifacts/spotcheck/score_labels.py artifacts/spotcheck/labels/*.csv

For each labeller, sheet and field it reports how many items were labelled,
how many the labeller would overturn, and the overturn share with a Wilson 95%
interval. "Unsure" labels are counted separately and left out of the share.
Results are printed and written to artifacts/spotcheck/spotcheck_results.json.
"""

import collections
import csv
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "spotcheck_results.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def main(paths):
    cells = collections.defaultdict(lambda: {"n": 0, "overturn": 0, "unsure": 0})
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["labeller"], row["sheet"], row["field"])
                if row["own_label"] == "?":
                    cells[key]["unsure"] += 1
                    continue
                if row["stored_label"] == "":
                    continue
                cells[key]["n"] += 1
                cells[key]["overturn"] += int(row["own_label"] != row["stored_label"])
    results = []
    for (who, sheet, field), c in sorted(cells.items()):
        share = c["overturn"] / c["n"] if c["n"] else None
        results.append(
            {
                "labeller": who,
                "sheet": sheet,
                "field": field,
                "labelled": c["n"],
                "overturned": c["overturn"],
                "overturn_share": None if share is None else round(share, 4),
                "wilson95": wilson(c["overturn"], c["n"]),
                "unsure": c["unsure"],
            }
        )
        rate = "-" if share is None else f"{share:.3f}"
        print(
            f"{who:10} {sheet:32} {field:18} {c['overturn']:3}/{c['n']:<3} {rate}  unsure {c['unsure']}"
        )
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
