"""Model review of model-authored question sets (plan 7.3)."""

import json
from pathlib import Path

import pytest
from conftest import QUESTIONS, ScriptedJudgeClient
from PIL import Image

from multimodal_graph_rag.cli import main
from multimodal_graph_rag.clients import JUDGE_ATTEMPTS
from multimodal_graph_rag.errors import ConfigurationError, InvalidModelOutputError
from multimodal_graph_rag.evaluation.posthoc import JudgementStore
from multimodal_graph_rag.evaluation.review import (
    build_items,
    check_reviewer_family,
    check_vision,
    gold_excerpts,
    item_verdicts,
    pending,
    run_item,
    summarise_review,
)
from multimodal_graph_rag.schemas import QuestionRecord, load_questions


def _ok(model, prompt):
    return "It holds.\n1"


def test_reviewer_is_never_from_the_authors_family():
    check_reviewer_family("claude-opus-5.5", "deepseek")
    check_reviewer_family("gpt-6.1-sol", "claude-haiku")
    with pytest.raises(ConfigurationError):
        check_reviewer_family("deepseek-v4-pro", "deepseek")
    with pytest.raises(ConfigurationError):
        check_reviewer_family("claude-opus-5.5", "claude-haiku")
    with pytest.raises(ConfigurationError):
        check_reviewer_family("claude-opus-5.5", "not-a-model")
    check_vision("gpt-6.1-sol")
    with pytest.raises(ConfigurationError):
        check_vision("deepseek-v4-pro")


def test_text_items_carry_the_gold_chunks_and_three_verdicts(corpus_dir, question_file):
    questions = load_questions(question_file)
    items = build_items(questions, set_name="demo", corpus_dir=corpus_dir)
    assert [i.criterion for i in items] == ["answerable", "unique", "well_formed"] * 3
    cross = next(q for q in questions if len(q.gold_sources) == 2)
    # the linking entity picks the chunk of each gold paper that holds it
    sources = {source for source, _ in gold_excerpts(cross, corpus_dir)}
    assert sources == {"alpha.txt", "beta.txt"}
    prompt = next(i for i in items if i.question == cross).prompt
    assert "Compound Zeta binds the kappa receptor" in prompt
    assert cross.answer in prompt
    # a question without an entity gets the chunks closest to its answer
    plain = QuestionRecord(
        id="p",
        question="q?",
        answer="12 percent",
        gold_sources=("gamma.txt",),
        question_type="text",
    )
    assert (
        "reduced fracture incidence by 12 percent"
        in gold_excerpts(plain, corpus_dir)[0][1]
    )
    assert not any(i.images for i in items)
    with pytest.raises(ConfigurationError):
        gold_excerpts(
            QuestionRecord(
                id="m",
                question="q?",
                answer="a",
                gold_sources=("none.txt",),
                question_type="text",
            ),
            corpus_dir,
        )


@pytest.fixture
def figure_set(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (8, 8), "white").save(images / "PMC1_00001_figure0.png")
    (images / "captions.json").write_text(
        json.dumps({"PMC1_00001_figure0.png": "Survival curve of both arms."})
    )
    corpus = tmp_path / "pages"
    corpus.mkdir()
    (corpus / "PMC1_00001.txt").write_text(
        "Median survival was 17.3 months in the benefit arm.", encoding="utf-8"
    )
    question = QuestionRecord(
        id="f1",
        question="What was the median survival of the benefit arm?",
        answer="about 17 months",
        gold_sources=("PMC1_00001_figure0.png",),
        question_type="figure",
    )
    return question, images, corpus


def test_figure_items_attach_the_crop_and_add_a_leakage_call(figure_set):
    question, images, corpus = figure_set
    items = build_items(
        [question],
        set_name="figs",
        corpus_dir=corpus,
        image_dir=images,
        captions={"PMC1_00001_figure0.png": "Survival curve of both arms."},
    )
    assert [i.criterion for i in items] == [
        "answerable",
        "unique",
        "well_formed",
        "leakage",
    ]
    crop = images / "PMC1_00001_figure0.png"
    assert [i.images for i in items] == [(crop,), (crop,), (crop,), ()]
    leakage = items[-1].prompt
    # the page text is found for a PubLayNet crop name, and the caption is shown
    assert "Median survival was 17.3 months" in leakage
    assert "Survival curve of both arms." in leakage
    with pytest.raises(ConfigurationError):
        build_items([question], set_name="figs", corpus_dir=corpus)


def test_run_item_logs_the_full_call_and_resume_skips_done_keys(
    corpus_dir, question_file, tmp_path
):
    items = build_items(
        load_questions(question_file), set_name="demo", corpus_dir=corpus_dir
    )
    client = ScriptedJudgeClient(_ok)
    store = JudgementStore(tmp_path / "review.jsonl")
    for item in items[:4]:
        store.add(run_item(item, client, "claude-opus-5.5", "deepseek"))
    entry = store.entries[items[0].key]
    assert entry["value"] == 1
    assert entry["justification"] == "It holds."
    assert entry["response"] == "It holds.\n1"
    assert entry["judge"] == "claude-opus-5.5" and entry["author"] == "deepseek"
    assert entry["input_tokens"] > 0 and entry["cost_usd"] > 0
    reopened = JudgementStore(tmp_path / "review.jsonl")
    assert pending(items, reopened) == items[4:]
    before = len(client.calls)
    assert all(i.key in reopened for i in items[:4]) and len(client.calls) == before


def test_a_reply_without_a_final_verdict_line_is_never_guessed(
    corpus_dir, question_file, tmp_path
):
    items = build_items(
        load_questions(question_file), set_name="demo", corpus_dir=corpus_dir
    )
    store = JudgementStore(tmp_path / "review.jsonl")
    for reply in ("It holds, so I would say yes.", "The answer is 1 overall.", ""):
        client = ScriptedJudgeClient(lambda model, prompt, reply=reply: reply)
        with pytest.raises(InvalidModelOutputError):
            run_item(items[0], client, "claude-opus-5.5", "deepseek")
        assert len(client.calls) == JUDGE_ATTEMPTS
    assert store.entries == {}
    assert not Path(store.path).exists()


def test_summary_counts_rejections_leakage_and_placeholder_references(
    corpus_dir, tmp_path
):
    raw = [*QUESTIONS, {**QUESTIONS[2], "q": "A second question?", "answer": "..."}]
    path = tmp_path / "set.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    questions = load_questions(path)
    ids = [q.id for q in questions]

    def reply(model, prompt):
        if "1 (unique)" in prompt and "Which receptor" in prompt:
            return "Two readings.\n0"
        if "1 (answerable)" in prompt and "A second question?" in prompt:
            return "No reference.\n0"
        return "Fine.\n1"

    items = build_items(questions, set_name="demo", corpus_dir=corpus_dir)
    client = ScriptedJudgeClient(reply)
    store = JudgementStore(tmp_path / "review.jsonl")
    for item in items:
        if item.question.id != ids[0] or item.criterion != "well_formed":
            store.add(run_item(item, client, "claude-opus-5.5", "deepseek"))
    rows = {r["question_id"]: r for r in item_verdicts(questions, "demo", store)}
    assert rows[ids[0]]["decision"] == "pending"  # a verdict is still missing
    assert rows[ids[1]]["decision"] == "rejected"
    assert rows[ids[3]]["decision"] == "rejected" and not rows[ids[3]]["has_reference"]
    summary = summarise_review(
        questions, "demo", store, reviewer="claude-opus-5.5", author="deepseek"
    )
    assert (summary["items"], summary["accepted"], summary["rejected"]) == (4, 1, 2)
    assert summary["pending"] == 1
    assert summary["rejection_rate"] == pytest.approx(2 / 3)
    assert summary["failed"] == {"answerable": 1, "unique": 1, "well_formed": 0}
    assert summary["reviewed"]["leakage"] == 0
    assert summary["leakage"] == {"reviewed": 0, "leaks": 0, "item_ids": []}
    placeholder = summary["placeholder_reference"]
    assert placeholder["item_ids"] == [ids[3]] and placeholder["rejected"] == 1
    assert placeholder["items_excluding"] == 3
    assert placeholder["rejected_excluding"] == 1
    assert summary["calls"] == 11


def test_leakage_verdicts_are_reported_apart_from_acceptance(figure_set, tmp_path):
    question, images, corpus = figure_set
    items = build_items(
        [question],
        set_name="figs",
        corpus_dir=corpus,
        image_dir=images,
        captions={"PMC1_00001_figure0.png": "Survival curve of both arms."},
    )
    client = ScriptedJudgeClient(_ok)  # every verdict is 1, so the answer leaks
    store = JudgementStore(tmp_path / "figs.jsonl")
    for item in items:
        store.add(run_item(item, client, "gpt-6.1-sol", "claude-haiku"))
    summary = summarise_review(
        [question], "figs", store, reviewer="gpt-6.1-sol", author="claude-haiku"
    )
    assert summary["accepted"] == 1 and summary["rejected"] == 0
    assert summary["leakage"]["leaks"] == 1 and summary["leakage"]["reviewed"] == 1
    image_calls = [c for c in client.calls if c[2]]
    assert len(image_calls) == 3


def test_cli_dry_run_prices_without_calling_or_writing(
    corpus_dir, question_file, tmp_path, capsys
):
    output = tmp_path / "out" / "review.jsonl"
    args = [
        "review-questions",
        str(question_file),
        "--author",
        "deepseek",
        "--corpus-dir",
        str(corpus_dir),
        "--output",
        str(output),
        "--dry-run",
    ]
    assert main([*args, "--reviewer", "claude-opus-5.5"]) == 0
    out = capsys.readouterr().out
    assert "9 of 9 calls" in out and "claude-opus-5.5" in out
    assert not output.exists()
    # the same family as the author is refused before anything is priced
    assert main([*args, "--reviewer", "deepseek-v4-pro"]) == 2
