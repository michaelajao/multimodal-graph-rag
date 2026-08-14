# Runs every remaining evaluation. Safe to re-run a block if one fails —
# each compare_all invocation is independent. ~a few dollars total.
$ErrorActionPreference = "Stop"
$models = @("gemini-flash-lite", "llama4-maverick", "llama4-scout")

# ── Three-way graph comparison: remaining generators ─────────────────────────
$env:RAG_CORPUS = "spiqa"
foreach ($m in $models) {
  python compare_all.py questions_spiqa_multihop_cross.json --model $m --systems baseline,+KG,+KGret
}

$env:RAG_CORPUS = "hotpotqa"
# gpt4o-mini comparison set not run yet -> include all four models
foreach ($m in @("gpt4o-mini") + $models) {
  python compare_all.py questions_hotpotqa_comparison.json --model $m --systems baseline,+KG,+KGret --hops 2
}
foreach ($m in $models) {
  python compare_all.py questions_hotpotqa_bridge.json --model $m --systems baseline,+KG,+KGret --hops 2
}

# ── SPIQA figures: pixel + caption sets, baseline vs +multimodal ──────────────
$env:RAG_CORPUS = "spiqa"
foreach ($m in @("gpt4o-mini") + $models) {
  python compare_all.py questions_spiqa_figures.json --model $m --systems baseline,+multimodal
  python compare_all.py questions_spiqa_figures_caption.json --model $m --systems baseline,+multimodal
}

Remove-Item Env:RAG_CORPUS
Write-Host "`nAll remaining runs complete. Results in results\"
