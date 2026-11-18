# Author spot-check sheets

## Quick start with the labeller page

1. Open `labeller.html` from this folder: double-click it in File Explorer. It opens in your browser and
   works offline. Keep it inside the repository, because the crop images are linked from
   `spiqa_images/` and `publaynet_images/` two folders up.
2. Type your labeller name (`author1` or `author2`) and press Start. Pick a sheet from the list.
3. For each item, answer every question with Yes, No or Unsure before anything else. The stored model
   verdict and its reasons appear only after you have answered, with "agree" or "overturn" beside each one.
   If seeing the verdict changes your mind about a careless click, correct it, but do not change a
   considered judgement to match the model.
4. Write a note for every No or Unsure. Keys: `1`, `0`, `?` answer the first open question; `Enter` or
   `→` goes to the next item; `←` goes back; `u` jumps to the next unlabelled item. Click an image to open
   it full size.
5. Progress is saved in the browser as you go. Press **Export labels (CSV)** at the end of each session and
   keep the file; **Import** restores it in another browser or on another machine.
6. When both authors are done, put both exports in one folder and run, from the repository root:

       python artifacts/spotcheck/score_labels.py path/to/spotcheck_labels_author1_*.csv path/to/spotcheck_labels_author2_*.csv

   It prints, per author, sheet and question, how many items the author would overturn, with a Wilson 95%
   interval, and writes `spotcheck_results.json` next to it. Those overturn shares fill the Author
   supervision section of the paper.

Rough time: the judge sheets are quick (most items take under a minute), triples take longer, and the
PubLayNet questions need a look at each figure. Plan on several sittings; a whole pass is about 557 items.

If the sheets are rebuilt, rebuild the page as well with `python artifacts/spotcheck/build_labeller.py`.
It reads the CSVs in this folder and `labeller_template.html`.

A separate check by Fable 5.2, a third Claude model, is in `fable/`. Its blind inputs are in
`fable/blind/` and its labels are in `fable/labels/`. It is a model audit, so it is not author
supervision and must not be reported as such.

The sections below describe the CSV sheets themselves, which the page is built from.

Two authors check samples of the model judgements and model-built items used in the paper. Each author
labels every sheet independently. This is author supervision of the instruments, not independent
annotation, and is not used to compute inter-annotator agreement. The report is the share of items each
author would overturn.

## How to label

1. Work alone. Do not open the other author's copy of a sheet, and do not discuss items until both
   copies are finished. Make your own copy of each CSV (for example `judge_verdicts_hotpotqa_bridge_author1.csv`)
   and fill in only your own columns.
2. Form your own view of the item from the question, the reference answer, the answer or evidence, and the
   crop image where there is one, before you read the model's justification or reasons.
3. Enter `1` when you agree with the model and `0` when you would overturn it. Leave a cell empty only when
   you cannot decide, and say why in `note`. Use `note` for any `0`.
4. The files are UTF-8 with a byte-order mark, so Excel opens them directly. Image paths are relative to the
   repository root (for example `spiqa_images/1705.09296v2-Figure1-1.png`).

## What `agree` means for each sheet

| Sheet | You are checking | `1` | `0` |
|---|---|---|---|
| `judge_verdicts_*.csv` | The judge's accuracy verdict on one generated answer. The judge saw only the question, the reference answer and the generated answer, and graded whether the generated answer is factually correct given the reference, ignoring minor wording differences. | You would give the same verdict (`judge_verdict_1_correct`) | You would overturn it |
| `graph_triples.csv` | The four labels the model judge (Claude Opus 5.5) gave one extracted triple (head, relation, tail) against its source chunk. One agree column per label per author. | You agree with that label | You would overturn it |
| `leakage_flagged.csv` | A flag from the exact-string and numeric leakage screen: whether the reference answer can be read from a caption or the document text. | `1` = real leakage: the answer, or the numbers in it, can be read from the quoted text in a way that answers the question | `0` = false alarm: the match is a coincidence (a lone digit, a section number, a citation) |
| `reviewed_questions.csv` | The model reviewer's verdict on each criterion for one question. One agree column per criterion per author. | You agree with the reviewer's verdict on that criterion | You would overturn it |

Triple labels (1 means the property holds). `entailment`: the chunk states, or directly entails, the relation
between the two endpoints. `canonical`: head and tail each name one specific entity or value, not a pronoun,
vague reference or fragment. `relation_specific`: the relation is informative, not generic. `provenance`:
the triple comes from this chunk's own text.

Review criteria (1 means the property holds, except `leakage`, where 1 means the answer leaks). `answerable`:
the reference answer follows from the gold evidence alone. `unique`: the question has one defensible answer.
`well_formed`: the question is specific and self-contained. `leakage`: the answer is readable from the
caption or page text; asked only for the PubLayNet figure questions, so leave the leakage columns empty
for `spiqa_multihop_cross` rows. An item is rejected by the reviewer if any of the first three is 0; leakage
does not change the decision. For `reviewed_questions.csv`, `agree` is about the verdict on each criterion,
not about whether the item was accepted overall.

## Counts per sheet

| File | Rows | Population it is drawn from |
|---|---|---|
| `judge_verdicts_hotpotqa_bridge.csv` | 50 | 6,500 answers (5 generators x 13 systems x 100 questions) |
| `judge_verdicts_hotpotqa_comparison.csv` | 50 | 6,500 |
| `judge_verdicts_spiqa_screened_pixels.csv` | 50 | 2,400 (4 generators x 4 systems x 150 questions) |
| `judge_verdicts_spiqa_flagged_pixels.csv` | 50 | 2,112 (4 x 4 x 132) |
| `judge_verdicts_spiqa_screened_captions.csv` | 50 | 1,200 (4 generators x 2 systems x 150) |
| `judge_verdicts_publaynet_pixels.csv` | 50 | 560 (4 x 4 x 35) |
| `graph_triples.csv` | 150 (50 per corpus: hotpotqa, spiqa, publaynet) | the 100 triples per corpus judged by Claude Opus 5.5 |
| `leakage_flagged.csv` | 22 (15 SPIQA screened set, 7 PubLayNet) | every item flagged in `figure_integrity_spiqa_primary.csv` and `figure_integrity_publaynet.csv` |
| `reviewed_questions.csv` | 85 (50 `spiqa_multihop_cross`, 35 `publaynet_figures`) | every reviewed item |
| `leakage_flagged_spiqa_caption_set_OPTIONAL.csv` | 132 | the SPIQA caption-flagged contrast set, flagged by construction; optional, see below |

Judge verdicts: HotpotQA sets use the DeepSeek V4 Pro accuracy verdict stored in each trace record
(`raw_judges.accuracy`); the SPIQA and PubLayNet sets use the GPT-6.1 Sol accuracy verdict. All are the
primary verdict, before any adjudication. Records with a run error are excluded (there are none).

## Seeds and sampling

* Judge verdict sheets use seed `20261118`. Each sheet spreads its 50 answers over the generator x system
  cells: slot `i` takes generator `i mod G` and system `(i + i // lcm(G, S)) mod S`, with generator and system
  order set by a SHA-256 hash of the seed. The k-th visit to a cell takes the k-th record in a SHA-256 ranking
  of `seed:question_id:generator:system`, which is the ranking `stratified_sample` in
  `src/multimodal_graph_rag/evaluation/posthoc.py` uses with the generator added so generators do not share
  questions. HotpotQA: every generator contributes 10, every system 3 or 4, no cell twice. SPIQA screened
  and flagged pixels and PubLayNet pixels: 12 or 13 per generator, 12 or 13 per system, 3 or 4 per cell.
  SPIQA screened captions: 12 or 13 per generator, 25 per system, 6 or 7 per cell.
* Triples: the 100 triples per corpus that Claude Opus 5.5 judged were drawn with seed `20260903`. The sheet
  takes the first 50 per corpus after ordering the judged triple ids by SHA-256 of `20261118:triples:<id>`.
  The source chunk text is rebuilt from the corpus with the same sampler and matches all 300 judged triples.
* Leakage and reviewed questions are full lists, not samples.

## What each sheet shows

* `judge_verdicts_*`: `evidence_seen_by_generator` lists the text passages (cut to 500 characters) and graph
  facts the generator received. For image arms `crops_sent_paths` lists only the crops actually sent
  (one under pixels, up to three under captions); crops that were only scored by CLIP are not listed.
  Under pixels the generator sees the image, not its caption. Under captions it sees the caption text, shown in
  `evidence_seen_by_generator`. `gold_units_received` counts gold units that reached the generator. The
  closed-book arm has no evidence.
* `leakage_flagged.csv`: `where_flagged` is the generated caption, or the paper or page text. `matching_snippet`
  marks the matched span between `>>` and `<<`; numeric flags show each matched number in context. PubLayNet page
  text is OCR and is often garbled. One snippet in the main sheet (question `8c7783e516a4e70d`) and five in the optional sheet say
  the match was not located (the span could not be found in the raw text); judge those from the crop and the
  paper.
* `reviewed_questions.csv`: `gold_evidence_shown_to_reviewer` is the gold text excerpts (or the figure and its
  generated caption) the reviewer saw. One SPIQA cross-paper item (`81f48f16b77acf24`) has `...` as its
  reference answer and is excluded from the paper's analysis; it is kept here so the row count is the full 85.

## Optional sheet

`leakage_flagged_spiqa_caption_set_OPTIONAL.csv` holds the 132 questions of the SPIQA caption-flagged contrast
set. They were flagged by the generated caption or the author caption when the set was built, so the flag is the
point of the set. Check it only if the paper's "every leakage-flagged item" is meant to include them. It is not
part of the 22 in the main leakage sheet.
