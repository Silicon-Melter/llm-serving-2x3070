# H3 profile, 8B-AWQ TP=2, run h3-fa9897e (repo fa9897e)
- c=1 (4 requests, 512 -> 64 tokens) profiled with vLLM's torch profiler; raw traces (145 + 150 MB) kept on SSD2:
  /media/earth/SSD2/llm-serving/profile-tmp/h3-fa9897e/rank{0,1}.*.pt.trace.json.gz
- c1/breakdown-decode.json: `python scripts/11_trace_breakdown.py <rank0> <rank1> --decode-only` (4 prefill windows left out,
  197 decode steps). c1/breakdown-rank{0,1}.json: whole window including prefills.
- c=64 attempt failed: the profiler records a Python stack per op by default; a TP worker reached 27.6 GB RAM and the kernel
  OOM-killed it (journal 21:19:20). server-oom.log. 10_profile_tp.sh now disables stacks and profiles a fixed step window.
