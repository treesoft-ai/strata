"""
Pulse dataset generation pipeline.

Runs all four Strata Synthesizer configs in sequence and reports results.

Usage:
    uv run pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    ("pulse-chat-instructions", "SFT — instruction following",       1000),
    ("pulse-chat-consistency",  "SFT — context consistency",         800),
    ("pulse-chat-natural",      "SFT — natural conversation style",  600),
    ("pulse-dpo-alignment",     "DPO — Opus-style alignment",        800),
]


def run_config(name: str, label: str, count: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Config: {name}  |  Target: {count} examples")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "main.py", "synthesize", name],
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.0f}s")
        return False

    print(f"\n  Completed in {elapsed/60:.1f} min")
    return True


def main():
    print("\nPulse Dataset Generation Pipeline")
    print(f"  {len(CONFIGS)} configs  |  {sum(c for _, _, c in CONFIGS):,} total examples\n")

    results = []
    pipeline_start = time.time()

    for name, label, count in CONFIGS:
        success = run_config(name, label, count)
        results.append((name, success))
        if not success:
            print(f"\n  Pipeline stopped at '{name}'. Fix the error and re-run.")
            print("  Completed configs are already saved — they will not be re-generated.")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start

    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")

    datasets = sorted(Path.home().joinpath(".strata/datasets").glob("*.jsonl"))
    print(f"\n  Datasets saved to ~/.strata/datasets/")
    for d in datasets:
        size_kb = d.stat().st_size // 1024
        print(f"    {d.name}  ({size_kb} KB)")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print()


if __name__ == "__main__":
    main()
