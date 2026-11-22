"""Tests for scripts/09_roofline.py: the ceiling must follow the physics it states."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("roof", ROOT / "scripts/09_roofline.py")
roof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roof)


def test_single_user_decode_ceiling_is_bandwidth_over_bytes():
    x = roof.model_numbers("8b", "awq")
    assert x["w_read"] == pytest.approx(4.93e9, rel=0.02)          # INT4 linears + BF16 output layer
    assert roof.decode_ceiling_single("8b", "awq", "1gpu") == pytest.approx(448e9 / x["w_read"])
    assert roof.decode_ceiling_single("8b", "awq", "tp2") == pytest.approx(2 * 448e9 / x["w_read"])


def test_weights_match_vllm_reported_size():
    assert roof.model_numbers("8b", "awq")["w_mem"] / 2 ** 30 == pytest.approx(5.71, rel=0.02)  # "Model loading took 5.71 GiB"


def test_two_copies_double_one_gpu_when_kv_is_not_the_limit():
    one, _ = roof.ceiling("4b", "awq", "1gpu", 4)
    two, _ = roof.ceiling("4b", "awq", "dp2", 8)
    assert two == pytest.approx(2 * one)


def test_kv_capacity_caps_the_batch():
    _, b = roof.ceiling("8b", "awq", "1gpu", 64)
    assert b == 21                                                  # (8 GiB - 5.68 GiB) / 144 KiB / 768 tokens
    _, b = roof.ceiling("8b", "awq", "tp2", 64)
    assert b == 64


def test_pipeline_with_one_user_is_no_faster_than_one_gpu():
    assert roof.ceiling("8b", "awq", "pp2", 1)[0] == pytest.approx(roof.ceiling("8b", "awq", "1gpu", 1)[0])


def test_model_that_does_not_fit_has_no_ceiling():
    c, b = roof.ceiling("8b", "bf16", "1gpu", 1)
    assert b == 0 and c != c                                        # NaN
