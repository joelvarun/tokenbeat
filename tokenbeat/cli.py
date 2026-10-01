"""Command line entry point.

    python -m tokenbeat.cli demo
    python -m tokenbeat.cli analyze trace.jsonl

The analyze subcommand reads one JSON object per line. Each line needs a `t`
field holding a monotonic arrival time in seconds, which is what a streaming
client records when it stamps `time.perf_counter()` on every SSE frame it
decodes. A plain text file of one float per line works too.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .baseline import baseline_estimate
from .estimate import analyze
from .simulate import StackConfig


def read_arrivals(path: str) -> np.ndarray:
    values = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                for key in ("t", "timestamp", "arrival", "time"):
                    if key in row:
                        values.append(float(row[key]))
                        break
                else:
                    raise ValueError("no timestamp field in line: %s" % line[:60])
            else:
                values.append(float(line))
    arr = np.array(values, dtype=float)
    if arr.size and np.any(np.diff(arr) < 0):
        raise ValueError("arrival times must be non-decreasing")
    return arr


def report(arrivals: np.ndarray, gamma_max: int, show_baseline: bool) -> None:
    result = analyze(arrivals, gamma_max=gamma_max)
    print(result.summary())
    if show_baseline:
        base = baseline_estimate(arrivals, gamma_max=gamma_max)
        print(
            "\nthreshold baseline       gamma %d, alpha %.3f"
            " (longest burst seen: %d tokens)"
            % (base["gamma"], base["alpha"], base["max_burst"])
        )


def cmd_demo(args) -> int:
    from .harness import run_stream

    cfg = StackConfig(
        alpha=args.alpha, gamma=args.gamma, step_ms=args.step_ms, contam_prob=args.contam
    )
    rng = np.random.default_rng(args.seed)
    print(
        "streaming %d tokens over loopback with alpha=%.2f gamma=%d step=%.1f ms"
        % (args.tokens, cfg.alpha, cfg.gamma, cfg.step_ms),
        flush=True,
    )
    trace = run_stream(cfg, args.tokens, rng)
    print("", flush=True)
    report(trace["arrivals"], args.gamma_max, show_baseline=True)
    print(
        "\nground truth              gamma %d, alpha %.3f, step %.2f ms"
        % (cfg.gamma, cfg.alpha, cfg.step_ms)
    )
    return 0


def cmd_analyze(args) -> int:
    arrivals = read_arrivals(args.path)
    report(arrivals, args.gamma_max, show_baseline=args.baseline)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tokenbeat", description=__doc__.splitlines()[0])
    parser.add_argument("--gamma-max", type=int, default=6, help="largest draft length tried")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="stream a loopback response and read it back")
    demo.add_argument("--alpha", type=float, default=0.8)
    demo.add_argument("--gamma", type=int, default=3)
    demo.add_argument("--step-ms", type=float, default=12.0)
    demo.add_argument("--contam", type=float, default=0.035)
    demo.add_argument("--tokens", type=int, default=300)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("analyze", help="read arrival times from a file")
    run.add_argument("path")
    run.add_argument("--baseline", action="store_true", help="also show the baseline")
    run.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
