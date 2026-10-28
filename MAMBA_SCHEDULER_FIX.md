# Mamba scheduler zero-chunk collapse — fix and install

## The bug

`_mamba_block_aligned_split` (`vllm/v1/core/sched/scheduler.py`) clips a prefill
chunk to the next Mamba-state block boundary. Under concurrent-load budget
contention, the per-tick token budget can shrink below one block; the old code
floored to the block boundary anyway whenever `block_size <= max_prefill_tokens`,
which can collapse the chunk end past its own start, returning `num_new_tokens
== 0`. Both callers of that result then do a bare `break` out of the
waiting-queue scheduling loop for that tick, stranding *every* waiting request
until something exogenous shifts the budget.

Symptom: `Running: 0 reqs, Waiting: N reqs, GPU KV cache usage: 0.0%` for
minutes, then self-recovering. Affects any hybrid Gated-DeltaNet / Mamba model
(e.g. Qwen3.6-35B-A3B) run with `--enable-prefix-caching` under concurrent
requests.

Upstream: `vllm-project/vllm#45414` / `#40707`. Two community fix attempts
(`#40709`, `#40757`) are still open and unmerged months later; this branch
fixes the same root cause independently, verified against the current
`v0.32.0` release.

## The fix

See `fix/mamba-scheduler-zero-chunk-collapse` branch,
`vllm/v1/core/sched/scheduler.py`:

1. Drop the `block_size <= max_prefill_tokens` escape — only floor to the
   block boundary when that still leaves real progress past `start`.
2. Defense in depth at both call sites that check for `num_new_tokens == 0`
   right after calling `_mamba_block_aligned_split`: `skip_request(...);
   continue` instead of a bare `break`, mirroring this same scheduling loop's
   own established skip-and-retry idiom used for every other "can't schedule
   this one right now" reason, instead of stranding the whole waiting queue.

Verified unit-level (no GPU needed) against the real installed `v0.32.0`
module: two pathological inputs (`start=100, num_new_tokens=3` and
`start=160, num_new_tokens=1`) collapse to `0` unpatched and return the
expected nonzero value patched; three normal inputs are byte-identical
between unpatched and patched — the fix changes nothing outside the one
collapse condition.

## Simple install — no rebuild needed

This is a single pure-Python file; no CUDA/C++ rebuild is required. Upgrade
to `v0.32.0` normally, then overwrite that one file in your installed
package with the patched version:

```bash
# 1. Install/upgrade to the release this fix is verified against.
pip install --upgrade vllm==0.32.0

# 2. Overwrite the one patched file in your installed vllm package.
SCHED_PATH="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')/v1/core/sched/scheduler.py"
curl -fsSL \
  https://raw.githubusercontent.com/TimS-lab/vllm/fix/mamba-scheduler-zero-chunk-collapse/vllm/v1/core/sched/scheduler.py \
  -o "$SCHED_PATH"
```

To verify it took effect without touching a GPU, run the same check used to
confirm this fix:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/TimS-lab/vllm/fix/mamba-scheduler-zero-chunk-collapse/unit_verify_mamba_fix.py \
  -o /tmp/unit_verify_mamba_fix.py
python /tmp/unit_verify_mamba_fix.py "$SCHED_PATH" "$SCHED_PATH"
# Both PATCHED/PATCHED rows should show no "<-- BUG" marker.
```

(Re-running `pip install --upgrade vllm` later will revert this file —
reapply step 2 after any upgrade until the upstream fix lands.)
