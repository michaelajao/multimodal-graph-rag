"""Build labeller.html, an offline page for the author spot-check.

Run from the repository root:

    python artifacts/spotcheck/build_labeller.py

It reads the spot-check CSVs in this folder and writes labeller.html next to
them, with the items embedded. Crop images are linked by relative path, so
open the page from this folder inside the repository (double-click it). The
page asks for the labeller's own judgement before it shows the stored model
verdict, keeps progress in the browser, and exports a CSV of labels for
score_labels.py.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_FROM_PAGE = "../../"  # labeller.html sits in artifacts/spotcheck/


def read(name):
    with open(HERE / name, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def crops(value):
    parts = [p.strip() for p in value.replace("|", ";").split(";")]
    return [ROOT_FROM_PAGE + p.replace("\\", "/") for p in parts if p]


def judge_items(name):
    items = []
    for r in read(name):
        items.append(
            {
                "key": f"{r['set']}|{r['row']}",
                "title": f"{r['set']} row {r['row']} ({r['generator']}, {r['system']})",
                "show": [
                    ["Question", r["question"]],
                    ["Reference answer", r["reference_answer"]],
                    ["Generated answer", r["generated_answer"]],
                ],
                "extra": [
                    ["Evidence the generator saw", r["evidence_seen_by_generator"]],
                    ["Gold received", r["gold_units_received"]],
                ],
                "images": crops(r["crops_sent_paths"]),
                "fields": [["correct", "Is the generated answer correct?"]],
                "stored": {"correct": r["judge_verdict_1_correct"]},
                "reveal": [
                    ["Judge", r["judge"]],
                    ["Justification", r["judge_justification"]],
                ],
            }
        )
    return items


def triple_items():
    items = []
    for r in read("graph_triples.csv"):
        items.append(
            {
                "key": f"{r['corpus']}|{r['row']}",
                "title": f"{r['corpus']} triple {r['row']}",
                "show": [
                    ["Triple", f"({r['head']})  --[{r['relation']}]-->  ({r['tail']})"],
                    ["Source chunk", r["source_chunk_text"]],
                ],
                "extra": [["Source file", r["source_file"]]],
                "images": [],
                "fields": [
                    [
                        "entailment",
                        "Does the chunk state or directly entail the triple?",
                    ],
                    ["canonical", "Are head and tail specific, canonical entities?"],
                    ["relation_specific", "Is the relation specific and informative?"],
                    ["provenance", "Is the triple drawn from this chunk?"],
                ],
                "stored": {
                    "entailment": r["opus_entailment"],
                    "canonical": r["opus_canonical"],
                    "relation_specific": r["opus_relation_specific"],
                    "provenance": r["opus_provenance"],
                },
                "reveal": [
                    ["Judge", "Claude Opus 5.5"],
                    ["Reasons", r["opus_reasons"]],
                ],
            }
        )
    return items


def leakage_items():
    items = []
    for r in read("leakage_flagged.csv"):
        items.append(
            {
                "key": f"{r['set']}|{r['question_id']}",
                "title": f"{r['set']} {r['question_id']}",
                "show": [
                    ["Question", r["question"]],
                    ["Reference answer", r["reference_answer"]],
                    ["Flagged in", r["where_flagged"]],
                    ["Rule", r["rule_fired"]],
                    ["Matching text", r["matching_snippet"]],
                ],
                "extra": [],
                "images": crops(r["crop_path"]),
                "fields": [
                    [
                        "real_leakage",
                        "Could a reader get the answer from this text without the figure?",
                    ]
                ],
                # the screen flagged every item, so its claim is always "leaks"
                "stored": {"real_leakage": "1"},
                "reveal": [["Screen", "flagged as leakage"]],
            }
        )
    return items


def review_items():
    items = []
    for r in read("reviewed_questions.csv"):
        fields = [
            [
                "answerable",
                "Answerable from the evidence shown, with a correct reference?",
            ],
            ["unique", "Does it have one defensible answer?"],
            ["well_formed", "Is it specific and self-contained?"],
        ]
        stored = {k: r[k] for k, _ in fields}
        if r["set"] == "publaynet_figures":
            fields.append(
                [
                    "leakage",
                    "Can the answer be read from caption or page text? (1 = leaks)",
                ]
            )
            stored["leakage"] = r["leakage"]
        items.append(
            {
                "key": f"{r['set']}|{r['question_id']}",
                "title": f"{r['set']} {r['question_id']}",
                "show": [
                    ["Question", r["question"]],
                    ["Reference answer", r["reference_answer"]],
                    [
                        "Evidence shown to the reviewer",
                        r["gold_evidence_shown_to_reviewer"],
                    ],
                ],
                "extra": [["Gold sources", r["gold_sources"]]],
                "images": crops(r["crop_path"]),
                "fields": fields,
                "stored": stored,
                "reveal": [
                    ["Reviewer", r["reviewer"]],
                    ["Decision", r["decision"]],
                    ["Answerable", r["answerable_reason"]],
                    ["Unique", r["unique_reason"]],
                    ["Well formed", r["well_formed_reason"]],
                    ["Leakage", r.get("leakage_reason", "")],
                ],
            }
        )
    return items


SHEETS = [
    [
        "judge_hotpotqa_bridge",
        "Judge verdicts: HotpotQA bridge",
        judge_items("judge_verdicts_hotpotqa_bridge.csv"),
    ],
    [
        "judge_hotpotqa_comparison",
        "Judge verdicts: HotpotQA comparison",
        judge_items("judge_verdicts_hotpotqa_comparison.csv"),
    ],
    [
        "judge_spiqa_screened_pixels",
        "Judge verdicts: SPIQA screened, pixels",
        judge_items("judge_verdicts_spiqa_screened_pixels.csv"),
    ],
    [
        "judge_spiqa_flagged_pixels",
        "Judge verdicts: SPIQA flagged, pixels",
        judge_items("judge_verdicts_spiqa_flagged_pixels.csv"),
    ],
    [
        "judge_spiqa_screened_captions",
        "Judge verdicts: SPIQA screened, captions",
        judge_items("judge_verdicts_spiqa_screened_captions.csv"),
    ],
    [
        "judge_publaynet_pixels",
        "Judge verdicts: PubLayNet, pixels",
        judge_items("judge_verdicts_publaynet_pixels.csv"),
    ],
    ["graph_triples", "Graph triples", triple_items()],
    ["leakage_flagged", "Leakage-flagged items", leakage_items()],
    ["reviewed_questions", "Model-reviewed questions", review_items()],
]

data = json.dumps(
    [{"id": i, "name": n, "items": items} for i, n, items in SHEETS], ensure_ascii=False
).replace("</", "<\\/")
template = (HERE / "labeller_template.html").read_text(encoding="utf-8")
(HERE / "labeller.html").write_text(
    template.replace("/*DATA*/[]", data), encoding="utf-8"
)
print("wrote", HERE / "labeller.html", sum(len(s[2]) for s in SHEETS), "items")
