# Project context for Claude sessions

This repository holds the code, configs, question sets, and manuscript for
"A Controlled Evaluation of Evidence Attribution in Graph-Augmented and
Multimodal RAG for Document Question Answering", a revision for Data &
Knowledge Engineering. The revision plan, every decision, and the run log
live in `docs/REVISION_PLAN.md` (section 17 is the most recent log). Read the
part of it that bears on a task before changing results or the paper.

## Machines and git

- Two Windows machines. The laptop has no GPU, and Smart App Control blocks
  PyTorch in the `rag` env there, so no `rag` command runs on it. The desktop
  (`ssh ajaoo@192.168.1.129`, DHCP address, may change) runs every `rag`
  command, conda env `rag`.
- Branch `consolidate-pipeline`. The user's fork is
  `github.com/michaelajao/multimodal-graph-rag` (public). Upstream is
  `sokistar24/multimodal-graph-rag`, which must never be pushed to. Remote
  names differ by machine: on the laptop the fork is `origin` and Soki's repo
  is `upstream`; on the desktop the fork is `michaelajao` and `origin` is
  Soki's repo. Check `git remote -v` before any push.
- Commit as the user. No `Co-Authored-By` or "Generated with Claude" lines in
  commits or PR bodies. Commit or push only when asked.
- Pull before starting work and push when stopping, so the two machines do
  not diverge on `docs/Graph_Rag/manuscript.tex`.

## Code rules

- Imports stay at the top of each module. Never move an import into a
  function to work around an environment problem.
- `ruff check .`, `ruff format --check .`, and `pytest` must pass (CI runs
  them). Tests use a deterministic fake client and make no paid calls.
- Edit files with the editor tools or a script file, not `sed -i` or bash
  heredocs with backslash escapes (they get mangled).

## Paid runs

- Every model call goes through OpenRouter with one key in `.env`
  (never commit or print it). Credit is scarce; check the balance before a
  paid run and never start one the user has not approved.
- Dry-run prices (`--dry-run`) run about 3x high for accuracy passes; price
  from measured spend in earlier traces instead. The account balance lags.
- Start parallel `conda run` jobs at least 15 s apart, or they clash on a
  temp file and exit with code 3.
- Traces and `judgements_*.jsonl` sit in `artifacts/runs/` (not tracked).
  Judge passes resume from the judgements file. `stratified_sample` is a
  deterministic ranking cut, so a larger fraction extends a sample without
  repeating questions.
- Trace trap: on image arms `metrics.complete` counts gold among the 5 crops
  CLIP scored, but pixels send only the top crop and captions the top 3. Use
  `gold_received` (`evaluation/posthoc.py`) or `gold_sent`
  (`evaluation/multimodal.py`) for what a generator actually saw.

## The manuscript

- Sources: `docs/Graph_Rag/manuscript.tex`, `references.bib`,
  `figures/*.tex`, `highlights.txt`. Build products are gitignored.
- VS Code rebuilds the manuscript on save. A shell build must go to a private
  folder or the two builds corrupt each other's `.aux`:
  `pdflatex -output-directory=<scratch>/build manuscript.tex`, copy
  `references.bib` into that folder, run `bibtex` there, then two more
  `pdflatex` passes. A first pass that stops while reading the `.aux` is a
  known transient; run again.
- Every number in the paper must come from an artifact in `artifacts/runs/`
  (for RQ3, `multimodal_rq3.json` and `multimodal_rq3_adjudicated.json`).
  Never type a number that was not read from a file in the same session.
- Plans that were not carried out are listed in Table `tab:deviations`. When
  a missing experiment is run, edit or remove its row.
- Writing style (the user reads each of these as a sign of AI writing):
  no `\paragraph{}` headings or bold/italic run-in labels; no bullet lists
  with bold lead phrases; no mid-sentence colons (colons before an equation
  or a real list are fine); titles without colons; captions say only what is
  shown and how to read it, with findings in the body; no engineering or
  budget talk (paid runs, checksums, manifests, config hashes, CLI names).
  Prefer specific, plainly hedged claims over promotional wording.

## Agent team

Subagents for this project are in `.claude/agents/`, and `/research-team`
(`.claude/commands/research-team.md`) starts a lead session that uses them.
