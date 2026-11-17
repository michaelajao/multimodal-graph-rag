"""Model-judged triple precision (plan section 7.6).

A stratified sample of extracted triples is read against the chunk each came
from, and a judge labels four properties, each with its reason:

    entailment         the chunk states this relation between these endpoints
    canonical          subject and object each name a specific entity or value
                       as the chunk does, not a pronoun, vague reference, or
                       fragment (both must pass)
    relation_specific  the relation is informative, not "is related to"
    provenance         the triple comes from this chunk's own text, so the
                       recorded source is where a reader finds its support

The share of triples judged entailed is reported as model-judged triple
precision. It is not human precision, and the grounding audit
(:mod:`.graph_grounding`) stays a separate, lexical measure.

The sample is the one ``sample_triple_audit`` draws, restricted to cache
entries whose chunk is still in the corpus. A second judge from a third family
re-labels a deterministic subset spread over the corpora, for agreement.
Each triple costs one call per judge: the reply gives a one-sentence reason for
each of the four criteria and then the four verdicts on the last four lines,
read by ``ModelClient.judge_labelled``. A reply missing any verdict is asked
again and, if still malformed, stops the run; no partial verdict is stored.
One entry per judge, corpus, and triple is appended to a JSON-lines file, so an
interrupted run resumes without repeating a call.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..clients import JUDGE_MAX_TOKENS, ModelClient
from ..corpora.loaders import iter_corpus_chunks
from ..errors import ConfigurationError
from ..retrieval.graph import chunk_key
from .audits import sample_triple_audit
from .posthoc import JudgementStore
from .review import family
from .statistics import cohen_kappa, wilson_interval

CRITERIA = ("entailment", "canonical", "relation_specific", "provenance")
HEADLINE = "entailment"
# Output of one call for a dry run: four one-sentence reasons and four verdict
# lines. Claude Opus 5.5 averaged about 290 output tokens on one-verdict calls
# (the 82 adjudication calls of 2026-10), so roughly 4 x 160 is allowed.
EXPECTED_OUTPUT_TOKENS = 650

HEADER = (
    "You are checking one fact that an automatic extractor read out of a "
    "passage of a document. The fact is a TRIPLE of subject, relation, and "
    "object.\n\n"
    "SOURCE FILE: {source}\n"
    'CHUNK: """{chunk}"""\n'
    'TRIPLE: subject="{subject}" | relation="{relation}" | object="{obj}"\n\n'
)
DEFINITIONS = {
    "entailment": (
        "Decide whether the CHUNK states, or directly entails, that the "
        "subject stands in the stated relation to the object. Use only the "
        "chunk, not outside knowledge; the same words appearing in the chunk "
        "is not enough if the chunk relates them differently or not at all."
    ),
    "canonical": (
        "Decide whether the subject and the object each name one specific "
        "entity, value, or concept as the chunk names it: a complete noun "
        "phrase that identifies one thing. It fails if either one is a "
        "pronoun, a vague reference ('it', 'the study', 'this method', "
        "'the results'), or a truncated or run-together fragment. Say which "
        "endpoint fails when one does."
    ),
    "relation_specific": (
        "Decide whether the relation is informative: judged together with its "
        "endpoints, it tells a reader what the particular relationship is. It "
        "fails if it is vacuous or generic enough to link almost any two "
        "things ('is related to', 'is associated with', 'involves', 'has', "
        "'mentions')."
    ),
    "provenance": (
        "Decide whether this triple comes from this chunk: the facts it "
        "states are in the chunk's own text, so a reader who opens this "
        "chunk finds the passage the triple was extracted from. It fails if "
        "the triple rests on outside knowledge, on text that lies outside the "
        "chunk, or on a different passage than the chunk's."
    ),
}
# What a 1 means for each criterion, in the reply instructions.
ONE = {
    "entailment": "the chunk states the relation (entailed)",
    "canonical": "both endpoints are specific",
    "relation_specific": "the relation is informative",
    "provenance": "the triple comes from this chunk",
}
PROMPT = (
    HEADER + "Judge the triple on four criteria. Judge each one on its own: failing "
    "one does not decide another.\n\n"
    + "\n\n".join(
        f"{number}. {criterion}\n{DEFINITIONS[criterion]}"
        for number, criterion in enumerate(CRITERIA, 1)
    )
    + "\n\nReply in two parts. First write one line for each criterion, in "
    "the order above, giving one sentence of reason:\n"
    + "\n".join(f"{c} reason: <one sentence>" for c in CRITERIA)
    + "\nThen, as the last four lines of your reply and with nothing after "
    "them, write the four verdicts, in this order and in exactly this form, "
    "with each X replaced by the digit 1 or 0:\n"
    + "\n".join(f"{c}: X" for c in CRITERIA)
    + "\nA 1 means "
    + "; ".join(f"for {c}, {ONE[c]}" for c in CRITERIA)
    + ". A 0 means the opposite."
)


def check_judge_families(primary: str, second: str | None, extractor: str) -> None:
    """The second judge must differ in family from both the primary judge and
    the model that extracted the triples (plan 7.1)."""
    if second is None:
        return
    for other, role in ((primary, "the primary judge"), (extractor, "the extractor")):
        if family(second) == family(other):
            raise ConfigurationError(
                f"second judge {second} is from the {family(other)} family of "
                f"{role} {other}; choose another family"
            )


@dataclass(frozen=True)
class TripleItem:
    corpus: str
    triple_id: str
    subject: str
    relation: str
    obj: str
    frequency: int
    source: str
    chunk: str

    @property
    def fields(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "triple_id": self.triple_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "extraction_frequency": self.frequency,
            "source": self.source,
        }


def triple_id(corpus: str, key: str, subject: str, relation: str, obj: str) -> str:
    """Stable id of a sampled triple, independent of the sample's size."""
    text = "\x1f".join((corpus, key, subject, relation, obj))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sample_corpus(
    corpus: str,
    corpus_dir: str | Path,
    cache_path: str | Path,
    count: int,
    seed: int,
) -> list[TripleItem]:
    """``count`` triples from one corpus's cache, each with its source chunk."""
    chunks: dict[str, tuple[str, str]] = {}
    for chunk in iter_corpus_chunks(corpus_dir):
        chunks.setdefault(chunk_key(chunk.text), (chunk.text, chunk.source))
    rows = sample_triple_audit(cache_path, corpus, count, seed, live_keys=set(chunks))
    items = []
    for row in rows:
        key = str(row["cache_key"])
        text, source = chunks[key]
        subject, relation, obj = (
            str(row[f]) for f in ("subject", "relation", "object")
        )
        items.append(
            TripleItem(
                corpus=corpus,
                triple_id=triple_id(corpus, key, subject, relation, obj),
                subject=subject,
                relation=relation,
                obj=obj,
                frequency=int(row["extraction_frequency"]),
                source=source,
                chunk=text,
            )
        )
    return items


def second_subset(
    items: Sequence[TripleItem], corpora: Sequence[str], count: int, seed: int
) -> list[TripleItem]:
    """A deterministic subset spread evenly over the corpora (the first
    ``count % len(corpora)`` corpora take one extra), ranked by a seeded hash."""
    if count < 0:
        raise ConfigurationError("second-judge count cannot be negative")
    chosen: list[TripleItem] = []
    for index, corpus in enumerate(corpora):
        quota = count // len(corpora) + (index < count % len(corpora))
        pool = sorted(
            (i for i in items if i.corpus == corpus),
            key=lambda i: hashlib.sha256(
                f"{seed}:second:{i.triple_id}".encode()
            ).hexdigest(),
        )
        if quota > len(pool):
            raise ConfigurationError(
                f"second-judge subset needs {quota} {corpus} triples; "
                f"the sample has {len(pool)}"
            )
        chosen.extend(pool[:quota])
    return chosen


@dataclass(frozen=True)
class TripleTask:
    """One call: a judge reading one triple against its chunk."""

    item: TripleItem
    judge: str

    @property
    def key(self) -> str:
        return f"{self.judge}|{self.item.corpus}|{self.item.triple_id}"

    @property
    def prompt(self) -> str:
        item = self.item
        return PROMPT.format(
            source=item.source,
            chunk=item.chunk,
            subject=item.subject,
            relation=item.relation,
            obj=item.obj,
        )


def plan_tasks(
    items: Sequence[TripleItem], judge: str, store: JudgementStore
) -> list[TripleTask]:
    """The calls a judge still has to make on these triples."""
    tasks = [TripleTask(item, judge) for item in items]
    return [task for task in tasks if task.key not in store]


def reasons(text: str) -> dict[str, str]:
    """The one-sentence reason written for each criterion; empty if absent.

    Reasons are kept for audit only and never decide a verdict.
    """
    found = {c: "" for c in CRITERIA}
    for line in text.splitlines():
        for criterion in CRITERIA:
            match = re.match(
                rf"^[\s>#*_`-]*{criterion}[\s*_`]+reason[\s*_`]*:\s*(.*)$",
                line.strip(),
                re.IGNORECASE,
            )
            if match and not found[criterion]:
                found[criterion] = match.group(1).strip(" *_`")
    return found


def run_task(task: TripleTask, client: ModelClient) -> dict[str, Any]:
    values, call = client.judge_labelled(task.judge, task.prompt, CRITERIA)
    return {
        "key": task.key,
        "pass": "judge-triples",
        **task.item.fields,
        "judge": task.judge,
        "values": values,
        "reasons": reasons(call.text),
        "response": call.text,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "max_tokens": JUDGE_MAX_TOKENS,
    }


def _verdicts(
    items: Sequence[TripleItem], judge: str, store: JudgementStore
) -> dict[str, dict[str, int]]:
    """Per triple id, the four verdicts the judge gave; triples it has not
    judged yet are absent."""
    found: dict[str, dict[str, int]] = {}
    for item in items:
        entry = store.entries.get(f"{judge}|{item.corpus}|{item.triple_id}")
        if entry:
            found[item.triple_id] = {c: int(entry["values"][c]) for c in CRITERIA}
    return found


def _rate(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"yes": 0, "n": 0, "rate": None, "wilson95": None}
    low, high = wilson_interval(sum(values), len(values))
    return {
        "yes": sum(values),
        "n": len(values),
        "rate": sum(values) / len(values),
        "wilson95": [low, high],
    }


def _rates(
    items: Sequence[TripleItem], verdicts: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    judged = [verdicts[i.triple_id] for i in items if i.triple_id in verdicts]
    table: dict[str, Any] = {c: _rate([v[c] for v in judged]) for c in CRITERIA}
    table["all_four"] = _rate([int(all(v.values())) for v in judged])
    return table


def summarise(
    items: Sequence[TripleItem],
    subset: Sequence[TripleItem],
    store: JudgementStore,
    *,
    corpora: Sequence[str],
    primary: str,
    second: str | None,
    extractor: str,
) -> dict[str, Any]:
    first = _verdicts(items, primary, store)
    summary: dict[str, Any] = {
        "primary_judge": primary,
        "second_judge": second,
        "extractor": extractor,
        "headline": "share of sampled triples the primary judge found entailed",
        "triples": len(items),
        "judged": len(first),
        "incomplete": len(items) - len(first),
        "per_corpus": {
            c: {
                "triples": sum(i.corpus == c for i in items),
                **_rates([i for i in items if i.corpus == c], first),
            }
            for c in corpora
        },
        "pooled": _rates(items, first),
        "by_extraction_frequency": {
            label: _rates([i for i in items if keep(i.frequency)], first)
            for label, keep in (
                ("1", lambda f: f == 1),
                ("2_or_more", lambda f: f >= 2),
            )
        },
    }
    if second is not None and subset:
        other = _verdicts(subset, second, store)
        both = [i for i in subset if i.triple_id in first and i.triple_id in other]
        agreement: dict[str, Any] = {}
        for c in CRITERIA:
            a = [first[i.triple_id][c] for i in both]
            b = [other[i.triple_id][c] for i in both]
            agreement[c] = {
                "n": len(both),
                "raw": sum(x == y for x, y in zip(a, b, strict=True)) / len(both)
                if both
                else None,
                "kappa": cohen_kappa(a, b) if both else None,
                f"{primary}_rate": sum(a) / len(a) if a else None,
                f"{second}_rate": sum(b) / len(b) if b else None,
            }
        summary["second_subset"] = {
            "triples": len(subset),
            "per_corpus": dict(Counter(i.corpus for i in subset)),
            "judged_by_both": len(both),
        }
        summary["agreement"] = agreement
    spent: dict[str, float] = {}
    for entry in store.entries.values():
        spent[entry["judge"]] = spent.get(entry["judge"], 0.0) + float(
            entry["cost_usd"]
        )
    summary["cost_usd"] = spent
    return summary


__all__ = [
    "CRITERIA",
    "TripleItem",
    "EXPECTED_OUTPUT_TOKENS",
    "PROMPT",
    "TripleTask",
    "check_judge_families",
    "plan_tasks",
    "run_task",
    "reasons",
    "sample_corpus",
    "second_subset",
    "summarise",
    "triple_id",
]
