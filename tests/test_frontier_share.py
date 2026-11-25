"""Tests for scripts/12_frontier_share.py: each community result is judged against its own card."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fs", ROOT / "scripts/12_frontier_share.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


def test_single_stream_ceiling_is_cards_times_bandwidth_over_streamed_bytes():
    df = fs.community()
    r = df[(df.source.str.startswith("XiongjieDai"))].iloc[0]
    assert r.ceiling == pytest.approx(448e9 / (4_912_898_048 - 295_501_824))   # 97.0 tok/s
    tensor = df[df.setup.str.contains("tensor") & df.model.str.startswith("Qwen3-14B")].iloc[0]
    layer = df[df.setup.str.contains("layer") & df.model.str.startswith("Qwen3-14B")].iloc[0]
    assert tensor.ceiling == pytest.approx(2 * layer.ceiling)                  # layer split runs cards in turn


def test_batched_rows_use_the_source_card_not_the_3070():
    df = fs.community()
    r = df[df.source.str.startswith("arXiv") & (df.setup == "1gpu") & (df.users == 128)].iloc[0]
    card = fs.GPUS["RTX 5060 Ti 16GB"]
    on_16gb, b16 = fs.roof.ceiling("8b", "nvfp4", "1gpu", 128, 256, 256, card)
    on_8gb, b8 = fs.roof.ceiling("8b", "nvfp4", "1gpu", 128, 256, 256, dict(card, vram=8 * 2 ** 30))
    assert r.ceiling == pytest.approx(on_16gb)
    assert b16 == 128 and b8 < 128 and on_8gb < on_16gb                          # 16 GB holds the whole batch
    assert (df.share > 0).all() and (df.share < 1).all()


def test_ours_come_from_the_roofline_table(tmp_path):
    p = tmp_path / "roof.csv"
    p.write_text("config,users,measured,ceiling,batch_ceiling,share,decode_measured,decode_ceiling,decode_share\n"
                 "8b-awq_1gpu,1,70,85,1,0.82,74,92,0.80\n8b-awq_1gpu,64,364,705,21,0.52,,,\n")
    df = fs.ours(p)
    assert list(df.regime) == ["single stream", "batched", "batched"]
    assert df.share.tolist() == [0.80, 0.82, 0.52]
