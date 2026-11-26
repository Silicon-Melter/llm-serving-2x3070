"""Stage B quality check: paired comparison of two lm-eval runs (reference vs quantised) on the same questions.

A plain "accuracy went from 88.1 to 87.6" hides two things: the noise of one score (+-1 point on 1,319 GSM8K
questions) and questions that flip in both directions. This script pairs the runs question by question and reports
  - accuracy of each run and the difference (test - ref, in points),
  - flips: right->wrong and wrong->right counts,
  - an exact McNemar test on the flips (p-value),
  - a 95 % paired bootstrap interval on the difference,
  - the pass/fail of the pre-registered rule (drop <= 1.0 point and the interval's worst case <= 2.0 points).

  python scripts/08_quality_compare.py --ref results/quality/<ref>/lm_eval --test results/quality/<cfg>/lm_eval
Reads every samples_<task>_*.jsonl (or .jsonl.gz, as committed) under each folder (lm-eval --log_samples). Metric per task: exact_match for
gsm8k (strict-match filter), acc for mmlu_*.
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import re

METRIC = [(re.compile(r"^gsm8k"), "exact_match", "strict-match"), (re.compile(r"^mmlu"), "acc", None)]
MAX_DROP, MAX_DROP_CI = 1.0, 2.0  # points; pre-registered in the vault decision note


def load(folder):
    """{(group, task, doc_id): 0/1}; group = gsm8k or mmlu (all MMLU subjects pooled)."""
    out = {}
    files = glob.glob(os.path.join(folder, "**", "samples_*.jsonl"), recursive=True)
    files += [f for f in glob.glob(os.path.join(folder, "**", "samples_*.jsonl.gz"), recursive=True)
              if f[:-3] not in files]                       # gzipped copy; the plain file wins if both exist
    for f in files:
        name = os.path.basename(f).removesuffix(".gz")
        task = re.match(r"samples_(.+?)_\d{4}-\d\d-\d\dT", name)
        task = task.group(1) if task else name[8:-6]
        rule = next(((m, flt) for rx, m, flt in METRIC if rx.match(task)), None)
        if not rule:
            continue
        metric, flt = rule
        group = "gsm8k" if task.startswith("gsm8k") else "mmlu"
        for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
            s = json.loads(line)
            if flt and s.get("filter") != flt:
                continue
            out[(group, task, s["doc_id"])] = float(s[metric])
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b right->wrong and c wrong->right flips."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(ref, test, n_boot=5000, seed=0):
    keys = sorted(set(ref) & set(test))
    if not keys:
        raise SystemExit("no questions in common")
    if len(keys) != len(ref) or len(keys) != len(test):
        raise SystemExit(f"runs cover different questions ({len(ref)} vs {len(test)}, {len(keys)} shared)")
    r = [ref[k] for k in keys]; t = [test[k] for k in keys]
    n = len(keys)
    diff = [b - a for a, b in zip(r, t)]
    d = 100 * sum(diff) / n
    rng = random.Random(seed)
    boots = sorted(100 * sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    right_to_wrong = sum(1 for a, b in zip(r, t) if a > b)
    wrong_to_right = sum(1 for a, b in zip(r, t) if b > a)
    return {"n": n, "acc_ref": 100 * sum(r) / n, "acc_test": 100 * sum(t) / n, "diff_points": d,
            "diff_ci95": [lo, hi], "right_to_wrong": right_to_wrong, "wrong_to_right": wrong_to_right,
            "flip_rate": 100 * (right_to_wrong + wrong_to_right) / n,
            "mcnemar_p": mcnemar_exact(right_to_wrong, wrong_to_right),
            "passes": d >= -MAX_DROP and lo >= -MAX_DROP_CI}


def compare(ref_folder, test_folder):
    ref, test = load(ref_folder), load(test_folder)
    groups = sorted({k[0] for k in ref} | {k[0] for k in test})
    res = {g: paired({k: v for k, v in ref.items() if k[0] == g}, {k: v for k, v in test.items() if k[0] == g})
           for g in groups}
    res["passes"] = all(v["passes"] for v in res.values())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--test", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    res = compare(a.ref, a.test)
    res.update(ref=a.ref, test=a.test)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
