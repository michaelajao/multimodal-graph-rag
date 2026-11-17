"""Model-judged triple precision (plan 7.6)."""

import json
import shutil
from collections import Counter

import pytest
from conftest import ScriptedJudgeClient

from multimodal_graph_rag.cli import main
from multimodal_graph_rag.clients import JUDGE_ATTEMPTS, JUDGE_MAX_TOKENS
from multimodal_graph_rag.errors import ConfigurationError, InvalidModelOutputError
from multimodal_graph_rag.evaluation.posthoc import JudgementStore
from multimodal_graph_rag.evaluation.triple_judging import (
    CRITERIA,
    TripleItem,
    check_judge_families,
    plan_tasks,
    run_task,
    sample_corpus,
    second_subset,
    summarise,
)

CORPORA = ["hotpotqa", "spiqa", "publaynet"]
PRIMARY, SECOND = "claude-opus-5.5", "deepseek-v4-pro"


def _item(corpus, index, frequency=1):
    return TripleItem(
        corpus=corpus,
        triple_id=f"{corpus}{index:02d}",
        subject=f"subject {index}",
        relation="measured",
        obj=f"object {index}",
        frequency=frequency,
        source=f"{corpus}.txt",
        chunk=f"subject {index} was measured against object {index}.",
    )


def _items(per_corpus=4):
    return [_item(c, i, 2 if i == 0 else 1) for c in CORPORA for i in range(per_corpus)]


def _reply(entailment=1, canonical=1, relation_specific=1, provenance=1):
    """A well-formed reply: four reasons, then the four verdict lines."""
    verdicts = {
        "entailment": entailment,
        "canonical": canonical,
        "relation_specific": relation_specific,
        "provenance": provenance,
    }
    reasons = "\n".join(f"{c} reason: Because {c}." for c in CRITERIA)
    return reasons + "\n" + "\n".join(f"{c}: {v}" for c, v in verdicts.items())


def test_second_judge_must_differ_from_primary_and_extractor():
    check_judge_families(PRIMARY, SECOND, "gpt4o-mini")
    check_judge_families(PRIMARY, None, "gpt4o-mini")
    with pytest.raises(ConfigurationError):  # OpenAI extracted the triples
        check_judge_families(PRIMARY, "gpt-6.1-sol", "gpt4o-mini")
    with pytest.raises(ConfigurationError):  # same family as the primary judge
        check_judge_families(PRIMARY, "claude-haiku", "gpt4o-mini")


def test_sample_is_deterministic_live_only_and_carries_its_chunk(
    corpus_dir, triple_cache, tmp_path
):
    cache = json.loads(triple_cache.read_text(encoding="utf-8"))
    cache["stale-chunk-key"] = [["old subject", "old relation", "old object"]] * 3
    cache["stale-chunk-key-2"] = [
        ["old a", "old b", "old c"],
        ["old d", "old e", "old f"],
    ]
    path = tmp_path / "with_stale.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    first = sample_corpus("demo", corpus_dir, path, 4, seed=7)
    assert first == sample_corpus("demo", corpus_dir, path, 4, seed=7)
    assert len({i.triple_id for i in first}) == 4
    for item in first:
        assert not item.subject.startswith("old")
        assert item.subject in item.chunk.lower()
        assert item.source.endswith(".txt")
    # the stale entries would have supplied more than the six live triples
    assert sample_corpus("demo", corpus_dir, path, 6, seed=7)
    with pytest.raises(ConfigurationError):
        sample_corpus("demo", corpus_dir, path, 7, seed=7)
    # an id does not change when the sample is made larger
    ids = {i.triple_id for i in sample_corpus("demo", corpus_dir, path, 6, seed=7)}
    assert {i.triple_id for i in first} <= ids


def test_second_subset_is_deterministic_and_spread_over_the_corpora():
    items = _items(per_corpus=20)
    subset = second_subset(items, CORPORA, 60, seed=1)
    assert subset == second_subset(items, CORPORA, 60, seed=1)
    assert Counter(i.corpus for i in subset) == {c: 20 for c in CORPORA}
    uneven = second_subset(items, CORPORA, 10, seed=1)
    assert Counter(i.corpus for i in uneven) == {
        "hotpotqa": 4,
        "spiqa": 3,
        "publaynet": 3,
    }
    assert set(uneven) <= set(items)
    assert uneven != second_subset(items, CORPORA, 10, seed=2)
    with pytest.raises(ConfigurationError):
        second_subset(_items(per_corpus=2), CORPORA, 10, seed=1)


def test_the_prompt_asks_for_all_four_criteria_in_one_call():
    task = plan_tasks(_items(per_corpus=1), PRIMARY, JudgementStore("none.jsonl"))[0]
    prompt = task.prompt
    assert "was measured against object 0" in prompt
    assert 'subject="subject 0" | relation="measured"' in prompt
    for criterion in CRITERIA:
        assert f"{criterion} reason: <one sentence>" in prompt
    form = prompt.splitlines()
    start = form.index("entailment: X")
    assert form[start : start + 4] == [f"{c}: X" for c in CRITERIA]
    assert prompt.rstrip().endswith("A 0 means the opposite.")


def test_each_triple_gets_one_call_and_resume_skips_it(tmp_path):
    items = _items(per_corpus=1)
    store = JudgementStore(tmp_path / "triples.jsonl")
    tasks = plan_tasks(items, PRIMARY, store)
    assert len(tasks) == 3
    client = ScriptedJudgeClient(lambda model, prompt: _reply(canonical=0))
    store.add(run_task(tasks[0], client))
    assert len(client.calls) == 1
    entry = store.entries[tasks[0].key]
    assert tasks[0].key == f"{PRIMARY}|hotpotqa|hotpotqa00"
    assert entry["values"] == {
        "entailment": 1,
        "canonical": 0,
        "relation_specific": 1,
        "provenance": 1,
    }
    assert entry["reasons"]["canonical"] == "Because canonical."
    assert entry["response"] == _reply(canonical=0)
    assert entry["cost_usd"] > 0 and entry["max_tokens"] == JUDGE_MAX_TOKENS
    assert entry["corpus"] == "hotpotqa" and entry["judge"] == PRIMARY
    assert entry["input_tokens"] > 0 and entry["output_tokens"] > 0
    again = plan_tasks(items, PRIMARY, JudgementStore(tmp_path / "triples.jsonl"))
    assert again == tasks[1:]
    # the second judge's entries are keyed apart from the primary's
    assert len(plan_tasks(items, SECOND, store)) == 3


def test_labels_are_read_with_markdown_and_spacing_tolerated(tmp_path):
    item = _items(per_corpus=1)[0]
    store = JudgementStore(tmp_path / "triples.jsonl")
    reply = (
        "Some reasoning.\n"
        "**entailment: 1**  \n"
        "- canonical: 0\n"
        "`relation_specific:` 1\n"
        "Provenance: 1."
    )
    client = ScriptedJudgeClient(lambda model, prompt: reply)
    entry = run_task(plan_tasks([item], PRIMARY, store)[0], client)
    assert entry["values"] == {
        "entailment": 1,
        "canonical": 0,
        "relation_specific": 1,
        "provenance": 1,
    }
    assert entry["reasons"] == {c: "" for c in CRITERIA}  # none were written


@pytest.mark.parametrize(
    "reply",
    [
        "entailment: 1\ncanonical: 1\nprovenance: 1",  # three lines only
        "Fine.\ncanonical: 1\nrelation_specific: 1\nprovenance: 1",  # label missing
        "entailment: 1\nentailment: 0\ncanonical: 1\nrelation_specific: 1\n"
        "provenance: 1",  # duplicated label
        "entailment: 1\ncanonical: 1\nprovenance: 1\nrelation_specific: 1",  # order
        "entailment: 1/0\ncanonical: 1\nrelation_specific: 1\nprovenance: 1",
        "entailment: 1\ncanonical: 1\nrelation_specific: 1\nprovenance: yes",
        "entailment: 2\ncanonical: 1\nrelation_specific: 1\nprovenance: 1",
        "entailment: -1\ncanonical: 1\nrelation_specific: 1\nprovenance: 1",
        "entailment: 1 as stated\ncanonical: 1\nrelation_specific: 1\nprovenance: 1",
        _reply() + "\nThat is my answer.",  # text after the verdicts
        "1\n0\n1\n1",  # digits without labels
        "",
    ],
)
def test_a_malformed_reply_is_never_turned_into_a_verdict(tmp_path, reply):
    item = _items(per_corpus=1)[0]
    store = JudgementStore(tmp_path / "triples.jsonl")
    client = ScriptedJudgeClient(lambda model, prompt: reply)
    with pytest.raises(InvalidModelOutputError):
        run_task(plan_tasks([item], PRIMARY, store)[0], client)
    assert len(client.calls) == JUDGE_ATTEMPTS
    assert store.entries == {} and not store.path.exists()


def test_a_malformed_reply_is_asked_again_and_its_cost_is_kept(tmp_path):
    item = _items(per_corpus=1)[0]
    replies = iter(["canonical: 1", _reply(entailment=0)])
    client = ScriptedJudgeClient(lambda model, prompt: next(replies))
    store = JudgementStore(tmp_path / "triples.jsonl")
    entry = run_task(plan_tasks([item], PRIMARY, store)[0], client)
    assert len(client.calls) == 2
    assert entry["values"]["entailment"] == 0
    assert entry["cost_usd"] > 0


def test_summary_reports_rates_counts_and_two_judge_agreement(tmp_path):
    items = _items(per_corpus=4)
    subset = second_subset(items, CORPORA, 6, seed=1)
    store = JudgementStore(tmp_path / "triples.jsonl")

    def reply(model, prompt):
        # the primary rejects the entailment of object 3 in every corpus; the
        # second judge also rejects object 1 and disagrees on specificity
        object_text = prompt.split('object="', 1)[1].split('"', 1)[0]
        if model == PRIMARY:
            return _reply(entailment=int(object_text != "object 3"))
        return _reply(
            entailment=int(object_text not in ("object 3", "object 1")),
            relation_specific=0,
        )

    client = ScriptedJudgeClient(reply)
    for judge, sample in ((PRIMARY, items), (SECOND, subset)):
        for task in plan_tasks(sample, judge, store):
            store.add(run_task(task, client))
    assert len(client.calls) == 12 + 6
    summary = summarise(
        items,
        subset,
        store,
        corpora=CORPORA,
        primary=PRIMARY,
        second=SECOND,
        extractor="gpt4o-mini",
    )
    assert summary["triples"] == 12 and summary["judged"] == 12
    assert summary["incomplete"] == 0
    hotpot = summary["per_corpus"]["hotpotqa"]
    assert hotpot["triples"] == 4
    assert hotpot["entailment"]["yes"] == 3 and hotpot["entailment"]["n"] == 4
    assert hotpot["entailment"]["rate"] == 0.75
    low, high = hotpot["entailment"]["wilson95"]
    assert low < 0.75 < high
    assert hotpot["relation_specific"]["rate"] == 1.0
    assert hotpot["all_four"]["yes"] == 3
    assert summary["pooled"]["entailment"] == {
        "yes": 9,
        "n": 12,
        "rate": 0.75,
        "wilson95": summary["pooled"]["entailment"]["wilson95"],
    }
    # frequency bands: index 0 of each corpus was extracted twice
    assert summary["by_extraction_frequency"]["2_or_more"]["entailment"]["n"] == 3
    assert summary["by_extraction_frequency"]["1"]["entailment"]["n"] == 9
    assert summary["second_subset"]["per_corpus"] == {c: 2 for c in CORPORA}
    agreement = summary["agreement"]
    assert agreement["entailment"]["n"] == 6
    both_entail = [
        (
            1 if i.obj != "object 3" else 0,
            1 if i.obj not in ("object 3", "object 1") else 0,
        )
        for i in subset
    ]
    raw = sum(a == b for a, b in both_entail) / 6
    assert agreement["entailment"]["raw"] == pytest.approx(raw)
    assert agreement["relation_specific"]["raw"] == 0.0  # primary 1, second 0
    assert agreement["relation_specific"]["kappa"] == pytest.approx(0.0)
    assert agreement["canonical"]["raw"] == 1.0
    assert agreement["canonical"]["kappa"] is None  # one label throughout: undefined
    assert set(summary["cost_usd"]) == {PRIMARY, SECOND}


def test_incomplete_sample_is_counted_not_filled(tmp_path):
    items = _items(per_corpus=2)
    store = JudgementStore(tmp_path / "triples.jsonl")
    client = ScriptedJudgeClient(lambda model, prompt: _reply())
    for task in plan_tasks(items, PRIMARY, store)[:3]:
        store.add(run_task(task, client))
    summary = summarise(
        items,
        [],
        store,
        corpora=CORPORA,
        primary=PRIMARY,
        second=None,
        extractor="gpt4o-mini",
    )
    assert summary["judged"] == 3 and summary["incomplete"] == 3
    assert summary["pooled"]["entailment"]["n"] == 3
    assert summary["pooled"]["all_four"]["n"] == 3
    assert "agreement" not in summary


def test_cli_dry_run_counts_calls_for_both_judges(
    corpus_dir, triple_cache, tmp_path, capsys
):
    root = tmp_path / "root"
    shutil.copytree(corpus_dir, root / "demo_corpus")
    graphs = tmp_path / "graphs2"
    graphs.mkdir()
    shutil.copy(triple_cache, graphs / "triples_cache_demo_corpus.json")
    output = tmp_path / "triples.jsonl"
    args = [
        "judge-triples",
        "--corpora",
        "demo",
        "--count",
        "4",
        "--second-count",
        "4",
        "--graphs-dir",
        str(graphs),
        "--corpus-root",
        str(root),
        "--output",
        str(output),
        "--summary",
        str(tmp_path / "summary.json"),
        "--dry-run",
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5.5: 4 calls" in out
    assert "deepseek-v4-pro: 4 calls" in out
    assert "total: 8 calls" in out
    assert not output.exists()
    assert main([*args, "--second-judge", "gpt-6.1-sol"]) == 2  # extractor's family
    capsys.readouterr()
    assert main([*args, "--second-count", "0"]) == 0
    assert "total: 4 calls" in capsys.readouterr().out
