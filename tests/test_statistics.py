import pytest

from multimodal_graph_rag.evaluation.statistics import (
    cohen_kappa,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_delta,
    wilson_interval,
)


def test_paired_bootstrap_reports_observed_delta():
    estimate = paired_bootstrap_delta(
        [0, 0, 1, 1], [0, 1, 1, 1], iterations=500, seed=3
    )
    assert estimate.delta == 0.25
    assert estimate.lower <= estimate.delta <= estimate.upper


def test_mcnemar_uses_only_discordant_pairs():
    result = mcnemar_exact([1, 0, 0, 1], [0, 1, 1, 1])
    assert result["baseline_only"] == 1
    assert result["treatment_only"] == 2
    assert result["p_value"] == 1.0


def test_holm_adjust_is_monotone_in_sorted_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_cohen_kappa_corrects_for_chance_and_is_undefined_without_variation():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])


def test_wilson_interval_brackets_the_rate_and_stays_in_range():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 5)[0] == 0.0 and wilson_interval(5, 5)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
