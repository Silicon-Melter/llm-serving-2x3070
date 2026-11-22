"""Tests for scripts/08_quality_compare.py on synthetic lm-eval sample files."""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qc", ROOT / "scripts/08_quality_compare.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


def write(folder, task, scores, metric, flt=None, other_filter=True):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"samples_{task}_2026-11-22T19-00-00.jsonl", "w") as f:
        for i, s in enumerate(scores):
            row = {"doc_id": i, metric: s}
            if flt:
                row["filter"] = flt
            f.write(json.dumps(row) + "\n")
            if flt and other_filter:  # lm-eval also logs the flexible-extract filter; it must be ignored
                f.write(json.dumps({"doc_id": i, metric: 1.0, "filter": "flexible-extract"}) + "\n")


def test_mcnemar_known_values():
    assert qc.mcnemar_exact(0, 0) == 1.0
    assert qc.mcnemar_exact(5, 5) == 1.0
    assert qc.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_identical_runs_pass_with_zero_difference(tmp_path):
    s = [1, 0, 1, 1] * 50
    write(tmp_path / "ref", "gsm8k", s, "exact_match", "strict-match")
    write(tmp_path / "test", "gsm8k", s, "exact_match", "strict-match")
    r = qc.compare(tmp_path / "ref", tmp_path / "test")
    g = r["gsm8k"]
    assert g["n"] == 200 and g["diff_points"] == 0 and g["flip_rate"] == 0 and r["passes"]


def test_cancelling_flips_are_counted_even_when_accuracy_is_equal(tmp_path):
    ref = [1] * 20 + [0] * 20 + [1] * 160
    test = [0] * 20 + [1] * 20 + [1] * 160
    write(tmp_path / "ref", "gsm8k", ref, "exact_match", "strict-match")
    write(tmp_path / "test", "gsm8k", test, "exact_match", "strict-match")
    g = qc.compare(tmp_path / "ref", tmp_path / "test")["gsm8k"]
    assert g["diff_points"] == 0 and g["right_to_wrong"] == 20 and g["wrong_to_right"] == 20
    assert g["flip_rate"] == pytest.approx(20.0)


def test_a_three_point_drop_fails_the_rule(tmp_path):
    ref = [1] * 900 + [0] * 100
    test = [0] * 30 + [1] * 870 + [0] * 100
    write(tmp_path / "ref", "gsm8k", ref, "exact_match", "strict-match")
    write(tmp_path / "test", "gsm8k", test, "exact_match", "strict-match")
    r = qc.compare(tmp_path / "ref", tmp_path / "test")
    assert r["gsm8k"]["diff_points"] == pytest.approx(-3.0) and not r["passes"]
    assert r["gsm8k"]["mcnemar_p"] < 1e-6


def test_mmlu_subjects_are_pooled_and_strict_filter_is_used(tmp_path):
    for sub in ["mmlu_anatomy", "mmlu_astronomy"]:
        write(tmp_path / "ref", sub, [1, 0] * 10, "acc")
        write(tmp_path / "test", sub, [1, 0] * 10, "acc")
    write(tmp_path / "ref", "gsm8k", [0] * 10, "exact_match", "strict-match")
    write(tmp_path / "test", "gsm8k", [0] * 10, "exact_match", "strict-match")
    r = qc.compare(tmp_path / "ref", tmp_path / "test")
    assert r["mmlu"]["n"] == 40 and r["gsm8k"]["acc_ref"] == 0  # flexible-extract rows (all 1.0) were ignored


def test_runs_on_different_questions_are_refused(tmp_path):
    write(tmp_path / "ref", "gsm8k", [1] * 10, "exact_match", "strict-match")
    write(tmp_path / "test", "gsm8k", [1] * 8, "exact_match", "strict-match")
    with pytest.raises(SystemExit):
        qc.compare(tmp_path / "ref", tmp_path / "test")
