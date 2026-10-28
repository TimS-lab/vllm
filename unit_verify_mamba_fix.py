"""Direct unit-level verification of the _mamba_block_aligned_split fix, same
methodology vLLM's own PR #40757 used: call the real function with a
SimpleNamespace stand-in for `self`/`request`, no GPU or model needed.

Loads BOTH the original (pre-patch, backed up as scheduler.py.orig) and the
patched scheduler.py module under distinct module names so they don't clobber
each other's import cache, then calls the real unbound method on each with
identical crafted inputs that hit the exact collapse path:
  start=100 (not block-aligned), num_new_tokens=3, block_size=16,
  max_prefill_tokens=8192 (block_size <= max_prefill_tokens, the condition
  that used to force the floor even when it collapses past start).
"""

import importlib.machinery
import importlib.util
import sys
from types import SimpleNamespace


def load_scheduler_module(path: str, name: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def make_self_and_request():
    self_ns = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        hash_block_size=16,
        mamba_partial_cache_hit=False,
        block_size=16,
        use_eagle_block_drop=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_shared_prefix_checkpoint=False,
    )
    request_ns = SimpleNamespace(
        num_computed_tokens=100,
        num_prompt_tokens=20000,
        num_tokens=20000,
        shared_prefix_boundary=10**9,
    )
    return self_ns, request_ns


def main() -> None:
    orig_path = sys.argv[1]
    patched_path = sys.argv[2]

    orig_mod = load_scheduler_module(orig_path, "scheduler_orig")
    patched_mod = load_scheduler_module(patched_path, "scheduler_patched")

    cases = [
        (100, 3),   # mid-block start, tiny leftover budget
        (160, 1),   # block-aligned start, 1-token budget
        (97, 15),   # one short of a full block
        (100, 16),  # exactly one block -- should NOT collapse even unpatched
        (100, 20),  # more than one block, not aligned -- normal floor applies on both
    ]
    for start, num_new_tokens in cases:
        for label, mod in [("UNPATCHED", orig_mod), ("PATCHED  ", patched_mod)]:
            self_ns, request_ns = make_self_and_request()
            request_ns.num_computed_tokens = start
            result = mod.Scheduler._mamba_block_aligned_split(
                self=self_ns,
                request=request_ns,
                num_new_tokens=num_new_tokens,
                num_new_local_computed_tokens=0,
                num_external_computed_tokens=0,
            )
            flag = "<-- BUG" if result == 0 else ""
            print(f"[{label}] start={start:4d} num_new_tokens={num_new_tokens:3d} -> {result:4d} {flag}")
        print()


if __name__ == "__main__":
    main()
