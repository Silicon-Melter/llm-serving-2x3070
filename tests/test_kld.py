"""Tests for scripts/07_kld.py (the quick quality screen). Pure maths, no server needed."""
import importlib.util
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kld", ROOT / "scripts/07_kld.py")
kld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kld)


def lp(ps):
    return [(t, math.log(p)) for t, p in ps]


def exact_kl(p, q):
    return sum(pi * math.log(pi / q[t]) for t, pi in p.items())


def test_identical_distributions_have_zero_kl():
    top = lp([(1, 0.6), (2, 0.3), (3, 0.1)])
    assert kld.kl_topk(top, top) == pytest.approx(0.0, abs=1e-9)


def test_kl_matches_exact_value_when_top_k_covers_everything():
    p = {1: 0.7, 2: 0.2, 3: 0.1}
    q = {1: 0.5, 2: 0.3, 3: 0.2}
    assert kld.kl_topk(lp(p.items()), lp(q.items())) == pytest.approx(exact_kl(p, q), rel=1e-6)


def test_kl_includes_mass_outside_the_top_k():
    # reference keeps 0.5 outside its listed tokens, test keeps 0.1: the bucket term must count that difference
    ref = lp([(1, 0.5)])
    test = lp([(1, 0.9)])
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert kld.kl_topk(ref, test) == pytest.approx(expected, rel=1e-6)


def test_compare_reports_agreement_and_perplexity():
    row = lambda real, top: {"real_lp": math.log(real), "top": lp(top)}  # noqa: E731
    ref = [[row(0.5, [(1, 0.5), (2, 0.4)]), row(0.5, [(3, 0.5), (4, 0.4)])]] * 3
    test = [[row(0.5, [(1, 0.5), (2, 0.4)]), row(0.25, [(4, 0.5), (3, 0.25)])]] * 3
    r = kld.compare(ref, test)
    assert r["positions"] == 6 and r["top1_agree"] == pytest.approx(0.5)
    assert r["ppl_ref"] == pytest.approx(2.0)
    assert r["ppl_test"] == pytest.approx(math.exp((math.log(2) + math.log(4)) / 2))
    assert r["kl_mean"] > 0


def test_compare_refuses_different_corpora():
    row = {"real_lp": -1.0, "top": [(1, -1.0)]}
    with pytest.raises(SystemExit):
        kld.compare([[row, row]], [[row]])
