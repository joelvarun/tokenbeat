"""End to end demo: three hidden serving stacks, read back from timings alone.

Every stream here goes over a real loopback TCP connection. The analyser is
given nothing but the arrival timestamps, and has to decide whether the server
speculates, how long its draft is, how often drafts are accepted, and how its
step time moves while the stream is running. The server moves its own step time
during a stream, the way batch occupancy does, so the nominal figure printed
below is where the step time started, not where it stayed.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenbeat import analyze, baseline_estimate
from tokenbeat.harness import run_stream
from tokenbeat.simulate import StackConfig

SCENARIOS = [
    ("vendor A, aggressive speculation", StackConfig(alpha=0.88, gamma=5, step_ms=12.0)),
    ("vendor B, short draft", StackConfig(alpha=0.62, gamma=2, step_ms=12.0)),
    ("vendor C, no speculation", StackConfig(alpha=0.0, gamma=0, step_ms=12.0)),
]
HEADER = "%-34s %-22s %-22s %s" % ("hidden stack", "truth", "tokenbeat", "threshold baseline")


def describe(gamma, alpha):
    return "no speculation" if gamma < 1 else "gamma %d, alpha %.2f" % (gamma, alpha)


def main() -> None:
    rng = np.random.default_rng(11)
    print("streaming %d responses over loopback, 300 tokens each\n" % len(SCENARIOS), flush=True)
    print(HEADER)
    print("-" * len(HEADER))

    reports = []
    for name, cfg in SCENARIOS:
        trace = run_stream(cfg, 300, rng)
        rep = analyze(trace["arrivals"])
        base = baseline_estimate(trace["arrivals"])
        reports.append((name, cfg, rep))
        print(
            "%-34s %-22s %-22s %s"
            % (
                name,
                describe(cfg.gamma, cfg.alpha),
                describe(rep.gamma, rep.alpha),
                describe(base["gamma"], base["alpha"]),
            ),
            flush=True,
        )

    print("\nwhat the timings give you beyond the flag\n")
    for name, cfg, rep in reports:
        drift = rep.step_ms_by_window
        print("  %s" % name)
        print(
            "    verification step  %.2f ms measured, %.2f ms nominal at stream start"
            % (rep.step_ms, cfg.step_ms)
        )
        if rep.speculative:
            print(
                "    tokens per target pass  %.2f measured, %.2f actual"
                % (
                    rep.tokens_per_step,
                    (1 - cfg.alpha ** (cfg.gamma + 1)) / (1 - cfg.alpha),
                )
            )
        print(
            "    step time across the stream  %.2f to %.2f ms over %d windows"
            % (drift.min(), drift.max(), drift.size)
        )
        print("    evidence for speculation     log10 Bayes factor %+.1f\n" % rep.log10_bayes_factor)

    print(
        "The baseline never reports the right draft length: one burst merged by a\n"
        "queueing delay is enough to push the longest-burst estimate past the cap."
    )


if __name__ == "__main__":
    main()
