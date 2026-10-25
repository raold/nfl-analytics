"""Holm-Bonferroni must stop at the first p-value that fails its threshold.

market_benchmark.metrics.edge_threshold_sweep relies on apply_holm_bonferroni. The earlier
version compared every p-value with its own threshold independently, so a large p-value late
in the order could be flagged after a smaller one had already failed.
"""

import numpy as np
import pandas as pd

from nfl_analytics.analysis.statistical_tests import apply_holm_bonferroni


def _holm(pvalues, alpha=0.05):
    return apply_holm_bonferroni(pd.DataFrame({"p_value_one_sided": pvalues}), alpha=alpha)


def test_stops_at_first_failure():
    # Thresholds 0.0167, 0.025, 0.05: 0.03 fails, so 0.04 is retained although 0.04 < 0.05
    out = _holm([0.04, 0.001, 0.03])
    assert out["p_value_one_sided"].tolist() == [0.001, 0.03, 0.04]
    assert out["significant_holm"].tolist() == [True, False, False]


def test_rejects_nothing_when_smallest_fails():
    # The two counterexamples found against the old implementation
    assert not _holm([0.02, 0.03, 0.04, 0.045, 0.049])["significant_holm"].any()
    assert not _holm([0.03, 0.04])["significant_holm"].any()


def test_rejects_all_when_every_step_passes():
    assert _holm([0.001, 0.01, 0.04])["significant_holm"].all()


def test_adjusted_p_values_are_monotone_holm():
    # Holm-adjusted: cumulative max of (m - i + 1) * p_(i), capped at 1
    out = _holm([0.01, 0.04, 0.03, 0.6])
    np.testing.assert_allclose(out["p_holm"], [0.04, 0.09, 0.09, 0.6])
    assert (out["significant_holm"] == (out["p_holm"] < 0.05)).all()
