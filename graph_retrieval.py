"""
+KG-retrieval arm: the graph acts on RETRIEVAL instead of generation.

Mechanism. Entities matched in the question (same whole-word, min-length rules
as graph_facts_for_query) identify graph-connected pages. Chunks from those
pages that text similarity missed are ADDED to the candidate set: the standard
top-K text chunks, plus up to N_BRIDGE chunks drawn from bridge pages, selected
by their embedding similarity to the query. Bridged chunks are labelled in the
prompt so the evidence provenance is explicit.

This is the channel the generation-side +KG configuration structurally
excludes: there, the provenance filter restricts graph facts to pages text
retrieval already returned, so the graph can never contribute unretrieved
evidence. Here it can — which is the mechanism the graph-RAG literature's
positive results (e.g. HippoRAG) attribute their gains to.

Pre-registered primary outcome (leakage-proof, no generation needed):
    completeness — fraction of cross questions with BOTH gold papers in the
    candidate set — before vs after bridging.
Secondary outcomes (generation-side, wired into compare_all separately):
    faithfulness, and accuracy reported raw AND conditioned on evidence
    completeness, since parametric leakage puts a ~0.5-0.8 accuracy floor on
    this corpus regardless of evidence.

Usage (retrieval-side A/B, no API generation cost beyond query embeddings):
    $env:RAG_CORPUS = "spiqa"
    python graph_retrieval.py questions_spiqa_multihop_cross.json
"""
import os
import re
import sys
import json

from rag_basics import build_index, retrieve, embed, CORPUS_DIR
from graph_aware import build_triples, build_graph

K_TEXT    = 3      # standard text retrieval depth (unchanged from all configs)
N_BRIDGE  = 2      # max bridged chunks added to the candidate set
DEEP_K    = 300    # search depth used to rank chunks on bridge pages
MIN_LEN   = 5      # entity length filter — MUST match graph_facts_for_query


def match_entities(query, graph, min_len=MIN_LEN):
    """Graph nodes appearing as whole words in the question (runtime-identical
    matching to graph_facts_for_query)."""
    q = query.lower()
    return [n for n in graph.nodes
            if len(n) >= min_len and re.search(r"\b" + re.escape(n) + r"\b", q)]


def _pages_of(node, graph):
    pages = set()
    for _, _, data in graph.out_edges(node, data=True):
        if data.get("source"):
            pages.add(data["source"])
    for _, _, data in graph.in_edges(node, data=True):
        if data.get("source"):
            pages.add(data["source"])
    return pages


def bridge_pages(query, graph, hops=1, max_neighbors=20):
    """
    Source pages reachable from question-matched entities.

    hops=1: pages of edges incident to matched nodes. Sufficient when the
            linking entity appears verbatim in the question (the SPIQA cross
            protocol guarantees this).
    hops=2: additionally, pages of edges incident to the matched nodes'
            NEIGHBOURS. Needed for HotpotQA-style bridge questions, where the
            question names entity A (Inception) but the second gold page
            belongs to bridge entity B (Nolan) that the question never
            mentions — reachable only via the A->B edge. max_neighbors caps
            hub blow-up (a matched node with hundreds of neighbours would
            otherwise pull in most of the corpus).
    """
    matched = match_entities(query, graph)
    pages = set()
    for node in matched:
        pages |= _pages_of(node, graph)
    if hops >= 2:
        for node in matched:
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            for nb in sorted(neighbors)[:max_neighbors]:
                pages |= _pages_of(nb, graph)
    return pages


def retrieve_with_bridge(query, index, chunks, sources, graph,
                         k=K_TEXT, n_bridge=N_BRIDGE, hops=1):
    """
    Standard top-k text retrieval, plus up to n_bridge chunks from
    graph-bridged pages not already represented in the top-k.

    Returns (base_retrieved, bridged_retrieved):
      base_retrieved    — the usual [(score, source, chunk)] * k
      bridged_retrieved — [(score, source, chunk)] * <=n_bridge, from bridge
                          pages only, ranked by query similarity
    """
    base = retrieve(query, index, chunks, sources, k=k)
    base_sources = {src for _, src, _ in base}

    candidates = bridge_pages(query, graph, hops=hops) - base_sources
    if not candidates:
        return base, []

    # Rank chunks on bridge pages by query similarity via a deep index search.
    import faiss
    qv = embed([query])
    faiss.normalize_L2(qv)
    scores, idxs = index.search(qv, min(DEEP_K, index.ntotal))

    bridged, used_pages = [], set()
    for score, i in zip(scores[0], idxs[0]):
        src = sources[i]
        if src in candidates and src not in used_pages:
            bridged.append((float(score), src, chunks[i]))
            used_pages.add(src)              # one chunk per bridge page
            if len(bridged) >= n_bridge:
                break
    return base, bridged


def generate_with_bridge(query, base, bridged, model="gpt4o-mini"):
    """+KG-retrieval generation: bridged chunks are labelled so evidence
    provenance is explicit. Returns GenResult (for compare_all wiring)."""
    from llm_client import call
    ctx = "\n\n".join(f"[{src}] {chunk}" for _, src, chunk in base)
    if bridged:
        ctx += "\n\n" + "\n\n".join(
            f"[{src} | graph-bridged] {chunk}" for _, src, chunk in bridged)
    return call(
        model,
        system="Answer using only the provided context. "
               "If the answer isn't in it, say you don't know.",
        user=f"Context:\n{ctx}\n\nQuestion: {query}",
    )


# ─────────────────────── retrieval-side A/B (no generation) ──────────────────

def gold_list(item):
    """Gold sources as a list, handling both schemas:
    SPIQA cross: source (str) + source2 (str); HotpotQA: source (list)."""
    s = item["source"]
    golds = list(s) if isinstance(s, list) else [s]
    if item.get("source2"):
        golds.append(item["source2"])
    return golds


def main(qfile, hops=1):
    print(f"Corpus: {CORPUS_DIR}   bridge hops: {hops}")
    index, chunks, sources = build_index()
    graph = build_graph(build_triples(with_sources=True))

    with open(qfile, encoding="utf-8") as f:
        questions = json.load(f)

    out = []
    stats = dict(any_b=0, any_a=0, both_b=0, both_a=0, fired=0, rescued=0)

    for item in questions:
        q, golds = item["q"], gold_list(item)
        base, bridged = retrieve_with_bridge(q, index, chunks, sources, graph,
                                             hops=hops)
        base_srcs  = {src for _, src, _ in base}
        all_srcs   = base_srcs | {src for _, src, _ in bridged}
        # Diagnostic: the full candidate set, to classify failures as
        # slot-budget (gold was a candidate, lost the N_BRIDGE ranking) vs
        # mechanism-ceiling (gold never became a candidate at all).
        cand = bridge_pages(q, graph, hops=hops) - base_srcs

        b_hits = [g in base_srcs for g in golds]
        a_hits = [g in all_srcs  for g in golds]
        stats["any_b"]  += any(b_hits);  stats["any_a"]  += any(a_hits)
        stats["both_b"] += all(b_hits);  stats["both_a"] += all(a_hits)
        stats["fired"] += bool(bridged)
        rescued = all(a_hits) and not all(b_hits)
        stats["rescued"] += rescued

        missing = [g for g in golds if g not in all_srcs]
        out.append({"q": q, "golds": golds,
                    "base": sorted(base_srcs),
                    "bridged": [[round(s, 4), src] for s, src, _ in bridged],
                    "n_candidates": len(cand),
                    "missing_gold_status": {
                        g: ("slot_lost" if g in cand else "not_candidate")
                        for g in missing},
                    "complete_before": all(b_hits), "complete_after": all(a_hits),
                    "rescued": rescued})

    n = len(out)
    dump = (f"kg_retrieval_ab_{os.path.splitext(os.path.basename(qfile))[0]}"
            f"_h{hops}.json")
    with open(dump, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{n} questions -> {dump}")
    print(f"{'':26}{'top-3 only':>12}{'+bridge':>12}")
    print(f"{'ANY gold in candidates':<26}{stats['any_b']:>9}/{n}{stats['any_a']:>9}/{n}")
    print(f"{'ALL golds in candidates':<26}{stats['both_b']:>9}/{n}{stats['both_a']:>9}/{n}"
          f"   <- primary outcome")
    print(f"\nbridge fired on {stats['fired']}/{n} questions; "
          f"rescued (incomplete -> complete): {stats['rescued']}")
    slot = sum(1 for o in out for v in o["missing_gold_status"].values()
               if v == "slot_lost")
    nocand = sum(1 for o in out for v in o["missing_gold_status"].values()
                 if v == "not_candidate")
    print(f"still-missing golds: {slot} slot_lost (candidate, outran by "
          f"N_BRIDGE={N_BRIDGE} ranking) / {nocand} not_candidate "
          f"(mechanism ceiling)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("qfile", nargs="?",
                    default="questions_spiqa_multihop_cross.json")
    ap.add_argument("--hops", type=int, default=1, choices=[1, 2],
                    help="1: SPIQA cross (entity in question); "
                         "2: HotpotQA bridge (bridge entity NOT in question)")
    a = ap.parse_args()
    main(a.qfile, hops=a.hops)
