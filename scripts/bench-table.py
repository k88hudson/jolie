#!/usr/bin/env python3
"""Run the criterion benchmarks and print a comparison table.

For every benchmarked trait method, shows jolie's time alongside rand_distr's
and statrs's as a percentage of jolie (100% = same speed, <100% = faster than
jolie, >100% = slower). Methods with no equivalent show "-".

Usage:
  uv run scripts/bench-table.py            # run benches, then print the table
  uv run scripts/bench-table.py --no-run   # just parse existing target/criterion
  uv run scripts/bench-table.py --filter uniform/cdf   # restrict to a pattern

Timing is configured in each bench's `criterion_group!` (short windows), so a
full run takes well under a minute.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRIT = ROOT / "target" / "criterion"
IMPLS = ("jolie", "rand_distr", "statrs")

NS_UNITS = ((1e9, "s"), (1e6, "ms"), (1e3, "µs"), (1.0, "ns"), (1e-3, "ps"))


def fmt_ns(ns: float) -> str:
    for threshold, unit in NS_UNITS:
        if ns >= threshold:
            return f"{ns / threshold:.2f} {unit}"
    return f"{ns:.2f} ns"


def run_benches(filt: str | None) -> None:
    cmd = ["cargo", "bench"]
    if filt:
        cmd += ["--", filt]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_results() -> dict[str, dict[str, float]]:
    """{group_id: {impl: ns}} from each new/estimates.json.

    criterion sanitizes "/" out of on-disk paths, so the original group id
    ("uniform/sample") is read from the sibling benchmark.json instead.
    """
    results: dict[str, dict[str, float]] = {}
    if not CRIT.exists():
        return results
    for est in CRIT.rglob("new/estimates.json"):
        meta_path = est.parent / "benchmark.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        group, func = meta["group_id"], meta["function_id"]
        if func not in IMPLS:
            continue
        data = json.loads(est.read_text())
        est_block = data.get("slope") or data.get("mean")
        if not est_block:
            continue
        results.setdefault(group, {})[func] = est_block["point_estimate"]
    return results


def cell(results: dict[str, float], impl: str, baseline: float | None) -> str:
    ns = results.get(impl)
    if ns is None:
        return "-"
    if impl == "jolie" or baseline is None or baseline == 0:
        return fmt_ns(ns)
    return f"{fmt_ns(ns)} ({ns / baseline * 100:.0f}%)"


def print_table(results: dict[str, dict[str, float]]) -> None:
    # group keys look like "uniform/sample" -> dist "uniform", method "sample"
    dists: dict[str, list[str]] = {}
    for group in results:
        dist, _, method = group.partition("/")
        dists.setdefault(dist, []).append(method)

    print("\n% is each impl's time relative to jolie (100% = same, <100% faster).\n")
    for dist in sorted(dists):
        print(f"### {dist}\n")
        print("| method | jolie | rand_distr | statrs |")
        print("| --- | --- | --- | --- |")
        for method in sorted(dists[dist]):
            row = results[f"{dist}/{method}"]
            base = row.get("jolie")
            cells = " | ".join(cell(row, impl, base) for impl in IMPLS)
            print(f"| {method} | {cells} |")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true", help="parse existing results only")
    ap.add_argument("--filter", help="criterion benchmark-name filter")
    args = ap.parse_args()

    if not args.no_run:
        run_benches(args.filter)

    results = load_results()
    if not results:
        print("No benchmark results found under target/criterion.", file=sys.stderr)
        return 1
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
