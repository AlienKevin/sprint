#!/usr/bin/env python3
# Copyright (c) 2026 QWOP-bench contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Plot a result document: where command tracking holds, and where it stops.

    python scripts/plot_results.py results/isaaclab-g1-flat.json
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1b1d"
GRID = "#d8d5cf"
GOOD = "#2f6f4f"
BAD = "#a4432b"
REF = "#8a8578"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    for path in args.result:
        doc = json.load(open(path))
        runs = sorted(doc["runs"], key=lambda r: r["commanded_speed"])
        cmd = [r["commanded_speed"] for r in runs]
        got = [r["mean_speed_mps"] for r in runs]
        valid = [r["valid"] for r in runs]

        lim = max(cmd) * 1.05
        ax1.plot([0, lim], [0, lim], "--", color=REF, lw=1, zorder=1,
                 label="perfect tracking")
        ax1.plot(cmd, got, "-", color=INK, lw=1.2, zorder=2, label=doc["label"])
        for c, g, v in zip(cmd, got, valid):
            ax1.plot(c, g, "o", ms=7, color=GOOD if v else BAD, zorder=3)

        # 100 m time where the course was completed; how far it got where not
        finished = [r for r in runs if r["finish_time_s"]]
        ceiling = max((r["finish_time_s"] for r in finished), default=1.0)
        for r in runs:
            if r["finish_time_s"]:
                ax2.bar(r["commanded_speed"], r["finish_time_s"], width=0.16,
                        color=GOOD if r["valid"] else REF)
                ax2.annotate(f"{r['finish_time_s']:.0f}s", (r["commanded_speed"], r["finish_time_s"]),
                             ha="center", va="bottom", fontsize=7, color=INK)
            else:
                ax2.bar(r["commanded_speed"], ceiling, width=0.16, color=BAD, alpha=0.25)
                ax2.annotate(f"DNF\n{r['distance_m']:.0f} m", (r["commanded_speed"], ceiling * 0.04),
                             ha="center", va="bottom", fontsize=7, color=BAD)

    ax1.set_xlabel("commanded forward speed (m/s)")
    ax1.set_ylabel("achieved speed (m/s)")
    ax1.set_title("Velocity tracking", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    ax2.set_xlabel("commanded forward speed (m/s)")
    ax2.set_ylabel("100 m time (s)")
    ax2.set_title("100 m time trial", loc="left", fontsize=11)

    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    out = args.out or os.path.splitext(args.result[0])[0] + ".png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
