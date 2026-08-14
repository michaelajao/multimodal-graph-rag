# When Do Multimodal and Graph-Augmented RAG Help?

This repository implements and evaluates retrieval-augmented generation (RAG)
systems for document question answering. The evaluation varies the evidence
configuration, the generator, and the corpus as independent factors: five
system configurations, four production multimodal generators, and three
corpora (PubLayNet, SPIQA, HotpotQA) with matched question-set controls. It
accompanies the paper *"When Do Multimodal and Graph-Augmented RAG Help? A
Controlled Evaluation for Document Question Answering."*

Every table and figure in the paper is regenerated from the released
per-question run logs by two scripts (`make_tables.py`, `make_figures.py`), so
every reported number is traceable to a recorded model response.

## Architecture

<p align="center">
  <img src="architecture.png" alt="Multimodal graph-RAG architecture" width="900">
</p>

Documents are processed into text passages, a knowledge graph of extracted
subject–relation–object triples, and figure/table crops. At query time the
evidence sources are retrieved independently. The knowledge graph is deployed
at two alternative stages: injected into the generation prompt under a
provenance constraint (+KG), or used to expand the retrieval candidate set
with chunks from entity-bridged documents (+KGret). The two are complementary
by construction—one can only reformulate retrieved evidence, the other can
only extend it—so their comparison isolates the pipeline stage at which graph
evidence acts.

## Systems evaluated

* `baseline` — text-only RAG (top-3 chunks)
* `+KG` — generation-stage graph augmentation (provenance-filtered facts in the prompt)
* `+KGret` — retrieval-stage graph augmentation (entity-bridged candidate expansion)
* `+multimodal` — CLIP-retrieved figures/tables (pixels or captions)
* `+both` — +KG and +multimodal combined

## Generators

The retrieval stack is fixed while the generator changes: GPT-4o-mini and
Gemini 3.1 Flash-Lite (closed-weight), Llama 4 Scout and Llama 4 Maverick
(open-weight, hosted inference). GPT-4o is a higher-cost diagnostic on
selected figure experiments. Question authoring (DeepSeek-Chat, Claude
Haiku 4.5) and judging (DeepSeek-Chat, Claude Haiku 4.5) use models outside
the four generator families.

## Corpora

One environment variable selects the corpus for every script:

```powershell
$env:RAG_CORPUS = "spiqa"      # publaynet (default) | spiqa | hotpotqa
```

| Corpus | Source | Units | Property varied |
|---|---|---|---|
| `publaynet` | `lhoestq/small-publaynet-wds` (1,000 pages) | page | disconnected, OCR text, obscure content |
| `spiqa` | SPIQA test-A (100 papers, 1,126 crops) | paper | cross-document terminology, clean TeX text, prominent papers |
| `hotpotqa` | HotpotQA distractor dev (2,964 paragraphs) | article | canonical entities, bridge/comparison controls, text-only |

Each corpus is rebuilt deterministically by its ingestion script
(`ingestion.py`, `ingest_spiqa.py`, `ingest_hotpotqa.py`); the corpus
directories themselves are not committed. Per-corpus caches
(`triples_cache_<corpus>_corpus.json`) are committed so the expensive triple
extraction never re-runs.

## Question sets

| File | n | Author | Notes |
|---|---|---|---|
| `questions_publaynet_text/multihop/figures[_caption].json` | 35/30/35/35 | DeepSeek / Claude Haiku | original sets |
| `questions_spiqa_text.json` | 35 | DeepSeek | |
| `questions_spiqa_multihop.json` | 30 | DeepSeek | within-paper control |
| `questions_spiqa_multihop_cross.json` | 50 | DeepSeek | seeded from the triple store; 50 entities, 50 paper pairs |
| `questions_spiqa_figures.json` | 35 | Claude Haiku (vision) | tiered by `verify_figure_integrity.py` |
| `questions_hotpotqa_bridge/comparison.json` | 50/50 | HotpotQA gold | multi-source gold lists |

All sets pass programmatic validators (no modality cues, no container
references, no source-id leaks, length caps) implemented in
`generate_questions.py`; `clean_question_sets.py` re-applies them to existing
files.

## Installation

```bash
conda env create -f environment.yml && conda activate rag
# or: pip install -r requirements.txt
```

Create a `.env` in the project root (never committed; see `.env.example`):

```text
OPENAI_API_KEY=...        # generation (GPT-4o-mini/GPT-4o), embeddings, captioning
DEEPSEEK_API_KEY=...      # question authoring + text judging
ANTHROPIC_API_KEY=...     # figure-question authoring + figure judging
GEMINI_API_KEY=...        # Gemini 3.1 Flash-Lite
DEEPINFRA_API_KEY=...     # Llama 4 Scout / Maverick hosted inference
```

## Reproduction path

```powershell
# 1. Build a corpus (once per corpus)
python ingestion.py --limit 1000                     # publaynet
python ingest_spiqa.py --papers 100                  # spiqa
python ingest_hotpotqa.py                            # hotpotqa

# 2. Run an evaluation (RAG_CORPUS selects everything consistently)
$env:RAG_CORPUS = "spiqa"
python compare_all.py questions_spiqa_multihop_cross.json --model gpt4o-mini --systems baseline,+KG,+KGret

$env:RAG_CORPUS = "hotpotqa"
python compare_all.py questions_hotpotqa_bridge.json --model gpt4o-mini --systems baseline,+KG,+KGret --hops 2

# 3. Regenerate every paper table and figure from results/
python make_tables.py          # -> paper_tables/
python make_figures.py         # -> figs/
```

`compare_all.py` guards against corpus/question mismatches (it aborts if no
gold source exists in the active corpus) and writes summary and per-question
CSVs to `results/`, which this repository includes for all reported runs.

Useful auxiliary analyses:

```powershell
python graph_retrieval.py questions_spiqa_multihop_cross.json          # retrieval-side A/B (+ bridge diagnostics)
python retrieval_dump.py questions_spiqa_multihop_cross.json           # per-question top-k with multi-gold hits
python verify_figure_integrity.py questions_spiqa_figures.json spiqa_images/captions.json spiqa_corpus
```

## Metrics

Retrieval: Recall@K, MRR, and completeness (all gold sources present, for
multi-gold questions). Answers: accuracy, faithfulness, and relevancy as
binary LLM-judge verdicts (RAGAS-style), plus accuracy conditioned on
evidence completeness, which separates retrieval-attributable performance
from parametric memory. `metrics.py` adds lexical/semantic measures (EM, F1,
BLEU, ROUGE-L, BERTScore).

## Explainability

`explainability.py` records, per answer: retrieved chunks and source pages
with scores, injected graph facts, retrieved crops with CLIP scores, bridged
chunks (for +KGret), and the final fused context.

## Project structure

```text
.
├── ingestion.py / ingest_spiqa.py / ingest_hotpotqa.py   # corpus builders
├── rag_basics.py / rag_multimodal.py / rag_full.py       # retrieval + fusion
├── graph_aware.py                                        # triples, graph, +KG
├── graph_retrieval.py                                    # +KGret bridging + A/B
├── compare_all.py                                        # main evaluation (5 systems)
├── generate_questions.py                                 # canonical question authoring
├── generate_questions_spiqa_cross.py                     # cross-paper set (triple-seeded)
├── generate_caption_questions_spiqa.py                   # matched caption protocol
├── clean_question_sets.py / verify_figure_integrity.py   # validators
├── retrieval_dump.py / metrics.py / evaluation.py / explainability.py
├── make_tables.py / make_figures.py                      # paper reproduction
├── questions_*.json                                      # evaluation sets
├── triples_cache_*_corpus.json                           # committed KG caches
├── results/                                              # all reported run logs
├── paper_tables/  figs/                                  # regenerated outputs
└── legacy/                                               # superseded scripts (do not run)
```

`legacy/` holds early question-generation scripts kept for provenance; running
them would overwrite canonical question files.
