"""Stage A quality screen: how far a quantised model's next-token predictions move away from a reference.

Feeds one fixed token corpus (same token ids for every Qwen3 checkpoint: they share a tokenizer) through a running
vLLM server and keeps the top-K log-probabilities at every position (no text is generated, so this is prefill-only
and fast). `compare` then reports, against a reference run:
  - KL(P_ref || P_test) per position, mean and p99 (top-K approximation with one shared "rest of vocabulary" bucket),
  - top-1 agreement: share of positions where both models rank the same token first,
  - perplexity of the real next token, and its change,
with 95 % bootstrap intervals over passages (positions inside one passage are not independent).

  python scripts/07_kld.py corpus  --model Qwen/Qwen3-8B-AWQ --out results/quality/corpus.json
  python scripts/07_kld.py collect --corpus results/quality/corpus.json --model Qwen/Qwen3-8B-AWQ --out results/quality/<config>/logprobs.jsonl.gz
  python scripts/07_kld.py compare --ref results/quality/<ref>/logprobs.jsonl.gz --test results/quality/<cfg>/logprobs.jsonl.gz
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import math
import random
import sys
import urllib.request

TOPK = 20  # vLLM's default max_logprobs


def build_corpus(model, n_passages, length, seed):
    """n_passages windows of `length` tokens from WikiText-2 test, chosen with a fixed seed."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(ids) - length, length), n_passages))
    return {"source": "wikitext-2-raw-v1 test", "tokenizer": model, "seed": seed, "length": length,
            "passages": [ids[s:s + length] for s in starts]}


def query(base_url, model, ids):
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": TOPK}).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    plp = out["choices"][0]["prompt_logprobs"]  # one entry per prompt position; first is None
    rows = []
    for pos in range(1, len(ids)):
        d = plp[pos]  # {token_id: {"logprob": x, "rank": r, ...}}, holds the top-K plus the real token
        real = str(ids[pos])
        top = sorted(((int(t), v["logprob"]) for t, v in d.items() if v.get("rank", TOPK + 1) <= TOPK),
                     key=lambda x: -x[1])[:TOPK]
        rows.append({"real_lp": d[real]["logprob"], "top": top})
    return rows


def collect(corpus, base_url, model, concurrency):
    with cf.ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(lambda ids: query(base_url, model, ids), corpus["passages"]))


def kl_topk(ref_top, test_top):
    """KL(P||Q) over the reference's top-K tokens plus one bucket for the rest of the vocabulary.
    A token missing from the test's top-K gets the test's K-th log-probability (an upper bound on its probability,
    so the KL is if anything underestimated for that token)."""
    q = dict(test_top)
    q_floor = min(q.values()) if q else -30.0
    kl, p_sum, q_sum = 0.0, 0.0, 0.0
    for t, lp in ref_top:
        p = math.exp(lp)
        lq = q.get(t, q_floor)
        kl += p * (lp - lq)
        p_sum += p
        q_sum += math.exp(lq)
    p_rest, q_rest = max(1e-12, 1 - p_sum), max(1e-12, 1 - min(q_sum, 1 - 1e-12))
    return max(0.0, kl + p_rest * math.log(p_rest / q_rest))


def passage_stats(ref_rows, test_rows):
    kls = [kl_topk(r["top"], t["top"]) for r, t in zip(ref_rows, test_rows)]
    agree = [1.0 if r["top"] and t["top"] and r["top"][0][0] == t["top"][0][0] else 0.0
             for r, t in zip(ref_rows, test_rows)]
    nll_ref = [-r["real_lp"] for r in ref_rows]
    nll_test = [-t["real_lp"] for t in test_rows]
    return kls, agree, nll_ref, nll_test


def bootstrap(per_passage, stat, n=2000, seed=0):
    rng = random.Random(seed)
    vals = sorted(stat([per_passage[rng.randrange(len(per_passage))] for _ in per_passage]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def compare(ref, test):
    if len(ref) != len(test) or any(len(a) != len(b) for a, b in zip(ref, test)):
        raise SystemExit("reference and test were collected on different corpora")
    per = [passage_stats(r, t) for r, t in zip(ref, test)]
    flat = lambda i, P: [x for p in P for x in p[i]]  # noqa: E731
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    kl_mean = lambda P: mean(flat(0, P))  # noqa: E731
    agree = lambda P: mean(flat(1, P))  # noqa: E731
    ppl_ref = lambda P: math.exp(mean(flat(2, P)))  # noqa: E731
    ppl_test = lambda P: math.exp(mean(flat(3, P)))  # noqa: E731
    ppl_delta = lambda P: ppl_test(P) / ppl_ref(P) - 1  # noqa: E731
    kls = sorted(flat(0, per))
    return {
        "positions": len(kls), "passages": len(per),
        "kl_mean": kl_mean(per), "kl_mean_ci95": bootstrap(per, kl_mean),
        "kl_p99": kls[int(0.99 * len(kls)) - 1],
        "top1_agree": agree(per), "top1_agree_ci95": bootstrap(per, agree),
        "ppl_ref": ppl_ref(per), "ppl_test": ppl_test(per),
        "ppl_rel_change": ppl_delta(per), "ppl_rel_change_ci95": bootstrap(per, ppl_delta),
    }


def save_rows(path, rows):
    with gzip.open(path, "wt") as f:
        for p in rows:
            f.write(json.dumps(p) + "\n")


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus"); c.add_argument("--model", required=True); c.add_argument("--out", required=True)
    c.add_argument("--passages", type=int, default=128); c.add_argument("--length", type=int, default=512)
    c.add_argument("--seed", type=int, default=1234)
    g = sub.add_parser("collect"); g.add_argument("--corpus", required=True); g.add_argument("--model", required=True)
    g.add_argument("--out", required=True); g.add_argument("--base-url", default="http://127.0.0.1:8000")
    g.add_argument("--concurrency", type=int, default=8)
    m = sub.add_parser("compare"); m.add_argument("--ref", required=True); m.add_argument("--test", required=True)
    m.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "corpus":
        json.dump(build_corpus(a.model, a.passages, a.length, a.seed), open(a.out, "w"))
    elif a.cmd == "collect":
        save_rows(a.out, collect(json.load(open(a.corpus)), a.base_url, a.model, a.concurrency))
    else:
        res = compare(load_rows(a.ref), load_rows(a.test))
        res.update(ref=a.ref, test=a.test)
        txt = json.dumps(res, indent=1)
        print(txt)
        if a.out:
            open(a.out, "w").write(txt + "\n")
    sys.exit(0)
