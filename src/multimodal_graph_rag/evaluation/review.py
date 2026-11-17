"""Model review of model-authored question sets (plan section 7.3).

A reviewer model from a family other than the author's reads each item and
returns four binary verdicts, each with the reason it gave:

    answerable    the reference answer follows from the gold provenance alone
    unique        the question has one defensible answer
    well_formed   the question is specific, not a template fragment
    leakage       the answer, a paraphrase, or a numeric equivalent is in a
                  caption or in retrievable corpus text (pixel-only sets, so
                  asked of image-gold items only; 1 means it leaks)

An item is rejected when any of the first three is 0. Leakage is reported
separately and decides the item's pixel-only status, not its acceptance.

Each verdict is its own call, so it goes through ``ModelClient.judge``
unchanged: one justification sentence, then the digit alone on the last line,
temperature 0, asked again when a reply has no verdict, reasoning switched off
after a reply fills the cap, and the run stopped (never a guessed verdict) when
every attempt is malformed. Separate calls also keep one criterion from
colouring another. Verdicts are appended to a JSON-lines file keyed by set,
question, and criterion, so a stopped run resumes without repeating a call.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..clients import ModelClient
from ..corpora.loaders import chunk_text, load_file
from ..errors import ConfigurationError
from ..questions.cross_paper import chunk_containing
from ..schemas import IMAGE_SUFFIXES, QuestionRecord
from .answer_match import answer_tokens, normalise_answer
from .audits import corpus_text_for_image
from .judges import VERDICT_INSTRUCTION
from .posthoc import JudgementStore, judgement_entry

CRITERIA = ("answerable", "unique", "well_formed", "leakage")
REJECTING = ("answerable", "unique", "well_formed")
# Chunks of a gold document shown when the question records no linking entity.
EXCERPTS_PER_SOURCE = 2
# Longest caption-plus-corpus text shown to the leakage reviewer; a longer text
# is cut and says so in the prompt.
MAX_LEAKAGE_CHARS = 20_000

ANSWERABLE_PROMPT = (
    "You are reviewing a question written for a question-answering benchmark. "
    "Decide whether the REFERENCE answer follows from the GOLD EVIDENCE alone, "
    "without outside knowledge: every fact the reference states must be "
    "present in the evidence or directly derivable from it.\n\n"
    "QUESTION: {question}\nREFERENCE: {reference}\n\n"
    "GOLD EVIDENCE:\n{evidence}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (answerable)", zero="0 (not answerable)")
)
UNIQUE_PROMPT = (
    "You are reviewing a question written for a question-answering benchmark. "
    "Decide whether the QUESTION has one defensible answer given the GOLD "
    "EVIDENCE: a careful reader would not give a materially different answer, "
    "and the question does not allow several equally valid answers (another "
    "entity, value, or reading). The REFERENCE answer is shown so you can see "
    "which answer is intended; do not judge it for correctness.\n\n"
    "QUESTION: {question}\nREFERENCE: {reference}\n\n"
    "GOLD EVIDENCE:\n{evidence}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (unique)", zero="0 (not unique)")
)
WELL_FORMED_PROMPT = (
    "You are reviewing a question written for a question-answering benchmark. "
    "Decide whether the QUESTION is well formed: specific, self-contained, and "
    "asking for one identifiable thing. It is not well formed if it is a "
    "template fragment, if it depends on a referent it does not name (such as "
    "'the study' or 'the first paper'), or if it is vague, generic, or "
    "garbled.{figure_note}\n\n"
    "QUESTION: {question}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (well formed)", zero="0 (not well formed)")
)
LEAKAGE_PROMPT = (
    "A question was written to be answerable only from a figure image, so its "
    "answer must not be readable from the figure's CAPTION or from the paper "
    "TEXT. Decide whether the REFERENCE answer can be found in them: the "
    "answer itself, a paraphrase of it, or a numeric equivalent (the same "
    "value in another unit or format). Topic words alone do not count; the "
    "answer's content must be there.\n\n"
    "QUESTION: {question}\nREFERENCE: {reference}\n\n"
    "CAPTION:\n{caption}\n\nTEXT:\n{text}\n\n"
    + VERDICT_INSTRUCTION.format(one="1 (leaks)", zero="0 (does not leak)")
)
FIGURE_NOTE = " The question refers to the attached figure image."
IMAGE_EVIDENCE = "(the attached figure image)"


def family(name: str) -> str:
    """The vendor family of a model alias, such as ``Anthropic`` or ``OpenAI``;
    an alias outside the roster is a configuration error."""
    return ModelClient.spec(name).vendor.split("/")[0]


def check_reviewer_family(reviewer: str, author: str) -> None:
    """A reviewer is never from the family that authored the items (7.1)."""
    if family(reviewer) == family(author):
        raise ConfigurationError(
            f"reviewer {reviewer} is from the {family(author)} family, which "
            f"authored these questions through {author}; choose another family"
        )


def check_vision(reviewer: str) -> None:
    if not ModelClient.spec(reviewer).vision:
        raise ConfigurationError(f"reviewer {reviewer} cannot see images")


def gold_excerpts(
    question: QuestionRecord, corpus_dir: str | Path
) -> list[tuple[str, str]]:
    """The gold text the reviewer is shown, as ``(source, excerpt)`` pairs.

    A graph-seeded question names its linking entity, and its author saw the
    longest chunk of each gold paper that holds that entity, so the same chunk
    is the gold provenance here. Any other text question gets the chunks of
    each gold document that share the most words with the reference answer.
    """
    corpus = Path(corpus_dir)
    entity = str(question.metadata.get("entity", "")).strip()
    excerpts: list[tuple[str, str]] = []
    for source in question.text_gold_sources:
        path = corpus / source
        if not path.is_file():
            raise ConfigurationError(f"gold source does not exist: {path}")
        chunk = chunk_containing(entity, source, corpus) if entity else None
        if chunk:
            excerpts.append((source, chunk))
            continue
        wanted = set(answer_tokens(question.answer))
        chunks = chunk_text(load_file(path))
        ranked = sorted(
            range(len(chunks)),
            key=lambda i: (-len(wanted & set(answer_tokens(chunks[i]))), i),
        )[:EXCERPTS_PER_SOURCE]
        excerpts.extend((source, chunks[i]) for i in sorted(ranked))
    return excerpts


def gold_images(question: QuestionRecord, image_dir: str | Path | None) -> list[Path]:
    sources = [s for s in question.gold_sources if s.lower().endswith(IMAGE_SUFFIXES)]
    if sources and image_dir is None:
        raise ConfigurationError(
            f"question {question.id} has an image gold source; pass --image-dir"
        )
    images = [Path(image_dir or ".") / source for source in sources]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise ConfigurationError(f"gold image does not exist: {missing}")
    return images


def leakage_texts(
    question: QuestionRecord,
    captions: Mapping[str, str],
    corpus_dir: str | Path,
) -> tuple[str, str, bool]:
    """Caption and retrievable corpus text for the question's gold image."""
    source = question.gold_sources[0]
    caption = str(captions.get(source, "")).strip()
    text = corpus_text_for_image(corpus_dir, source).strip()
    truncated = len(text) + len(caption) > MAX_LEAKAGE_CHARS
    if truncated:
        text = text[: max(0, MAX_LEAKAGE_CHARS - len(caption))] + " [text cut here]"
    return (
        caption or "(no caption)",
        text or "(no corpus text for this page)",
        truncated,
    )


@dataclass(frozen=True)
class ReviewItem:
    set_name: str
    question: QuestionRecord
    criterion: str
    prompt: str
    images: tuple[Path, ...] = ()
    truncated: bool = False

    @property
    def key(self) -> str:
        return f"{self.set_name}|{self.question.id}|{self.criterion}"


def build_items(
    questions: Sequence[QuestionRecord],
    *,
    set_name: str,
    corpus_dir: str | Path,
    image_dir: str | Path | None = None,
    captions: Mapping[str, str] | None = None,
) -> list[ReviewItem]:
    """Every review call for the set, in question order (done ones included)."""
    items: list[ReviewItem] = []
    for question in questions:
        images = gold_images(question, image_dir)
        evidence = "\n\n".join(
            ([IMAGE_EVIDENCE] if images else [])
            + [
                f"[{source}]\n{text}"
                for source, text in gold_excerpts(question, corpus_dir)
            ]
        )
        shared = {
            "question": question.question,
            "reference": question.answer,
            "evidence": evidence,
        }
        prompts = {
            "answerable": ANSWERABLE_PROMPT.format(**shared),
            "unique": UNIQUE_PROMPT.format(**shared),
            "well_formed": WELL_FORMED_PROMPT.format(
                question=question.question,
                figure_note=FIGURE_NOTE if images else "",
            ),
        }
        truncated = False
        if question.is_visual:
            caption, text, truncated = leakage_texts(
                question, captions or {}, corpus_dir
            )
            prompts["leakage"] = LEAKAGE_PROMPT.format(
                question=question.question,
                reference=question.answer,
                caption=caption,
                text=text,
            )
        for criterion in CRITERIA:
            if criterion in prompts:
                items.append(
                    ReviewItem(
                        set_name,
                        question,
                        criterion,
                        prompts[criterion],
                        # leakage is judged from text; the image is not the channel
                        images=tuple(images) if criterion != "leakage" else (),
                        truncated=truncated and criterion == "leakage",
                    )
                )
    return items


def pending(items: Sequence[ReviewItem], store: JudgementStore) -> list[ReviewItem]:
    return [item for item in items if item.key not in store]


def run_item(
    item: ReviewItem, client: ModelClient, reviewer: str, author: str
) -> dict[str, Any]:
    value, call = client.judge(reviewer, item.prompt, images=item.images)
    return {
        "key": item.key,
        "pass": "review-questions",
        "set": item.set_name,
        "question_id": item.question.id,
        "criterion": item.criterion,
        "author": author,
        "context_truncated": item.truncated,
        **judgement_entry(value, call, reviewer),
    }


def item_verdicts(
    questions: Sequence[QuestionRecord], set_name: str, store: JudgementStore
) -> list[dict[str, Any]]:
    """One row per question: each criterion's verdict (``None`` if not yet
    reviewed or not asked) and the decision they imply."""
    rows = []
    for question in questions:
        row: dict[str, Any] = {"question_id": question.id}
        for criterion in CRITERIA:
            entry = store.entries.get(f"{set_name}|{question.id}|{criterion}")
            row[criterion] = None if entry is None else int(entry["value"])
            row[f"{criterion}_reason"] = "" if entry is None else entry["justification"]
        verdicts = [row[c] for c in REJECTING]
        if 0 in verdicts:
            row["decision"] = "rejected"
        elif None in verdicts:
            row["decision"] = "pending"
        else:
            row["decision"] = "accepted"
        row["has_reference"] = bool(normalise_answer(question.answer))
        rows.append(row)
    return rows


def summarise_review(
    questions: Sequence[QuestionRecord],
    set_name: str,
    store: JudgementStore,
    *,
    reviewer: str,
    author: str,
) -> dict[str, Any]:
    rows = item_verdicts(questions, set_name, store)
    decisions = Counter(row["decision"] for row in rows)
    decided = decisions["rejected"] + decisions["accepted"]
    asked = {c: [r[c] for r in rows if r[c] is not None] for c in CRITERIA}
    placeholders = [r for r in rows if not r["has_reference"]]
    kept = [r for r in rows if r["has_reference"]]
    kept_decided = [r for r in kept if r["decision"] != "pending"]
    entries = [e for e in store.entries.values() if e.get("set") == set_name]
    return {
        "set": set_name,
        "reviewer": reviewer,
        "author": author,
        "items": len(rows),
        "accepted": decisions["accepted"],
        "rejected": decisions["rejected"],
        "pending": decisions["pending"],
        "rejection_rate": decisions["rejected"] / decided if decided else None,
        "failed": {c: sum(v == 0 for v in asked[c]) for c in REJECTING},
        "reviewed": {c: len(asked[c]) for c in CRITERIA},
        "leakage": {
            "reviewed": len(asked["leakage"]),
            "leaks": sum(v == 1 for v in asked["leakage"]),
            "item_ids": [r["question_id"] for r in rows if r["leakage"] == 1],
        },
        "placeholder_reference": {
            "items": len(placeholders),
            "item_ids": [r["question_id"] for r in placeholders],
            "rejected": sum(r["decision"] == "rejected" for r in placeholders),
            "items_excluding": len(kept),
            "rejected_excluding": sum(r["decision"] == "rejected" for r in kept),
            "rejection_rate_excluding": (
                sum(r["decision"] == "rejected" for r in kept) / len(kept_decided)
                if kept_decided
                else None
            ),
        },
        "calls": len(entries),
        "cost_usd": sum(float(e["cost_usd"]) for e in entries),
    }


def write_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=1), encoding="utf-8")


ITEM_FIELDS = (
    "question_id",
    "decision",
    "answerable",
    "unique",
    "well_formed",
    "leakage",
    "has_reference",
    "answerable_reason",
    "unique_reason",
    "well_formed_reason",
    "leakage_reason",
)


__all__ = [
    "CRITERIA",
    "ITEM_FIELDS",
    "REJECTING",
    "ReviewItem",
    "build_items",
    "check_reviewer_family",
    "check_vision",
    "family",
    "gold_excerpts",
    "item_verdicts",
    "pending",
    "run_item",
    "summarise_review",
    "write_summary",
]
