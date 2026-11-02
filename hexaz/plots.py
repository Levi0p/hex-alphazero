"""Figures for the README (written to figures/).

    python -m hexaz.plots runs/hex5_seed0 runs/hex7_seed0
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Validated categorical order (blue, orange, aqua, yellow); text stays in ink colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SIMS_COLOUR = {0: SERIES[0], 16: SERIES[1], 64: SERIES[2], 256: SERIES[3]}  # same colour on every chart
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "lines.linewidth": 2, "legend.frameon": False,
})


def forward_flops(n: int, ch: int, blocks: int) -> float:
    """Multiply-adds x 2 for one network evaluation (convs and dense layers)."""
    cells = n * n
    stem = cells * 3 * ch * 9
    res = blocks * 2 * cells * ch * ch * 9
    heads = cells * ch + ch * ch + ch
    return 2.0 * (stem + res + heads)


def compute_per_iter(cfg: dict) -> float:
    """Training compute per iteration: self-play network calls + gradient steps."""
    f = forward_flops(cfg["size"], cfg["channels"], cfg["blocks"])
    selfplay = cfg["games"] * cfg["moves"] * (cfg["sims"] + 1) * f
    train = 3 * f * cfg["batch"] * cfg["steps"]  # backward ~ 2x forward
    return selfplay + train


def strength_figure(runs: list[Path], out: Path):
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        cfg = json.loads((run / "config.json").read_text())
        rows = list(csv.DictReader(open(run / "sweep.csv")))
        per_iter = compute_per_iter(cfg)
        sims_values = sorted({int(r["sims"]) for r in rows})
        for sims in sims_values:
            colour = SIMS_COLOUR[sims]
            pts = sorted((int(r["iter"]), 100 * int(r["won_of_should_win"]) / int(r["should_win"]))
                         for r in rows if int(r["sims"]) == sims)
            x = [it * per_iter for it, _ in pts]
            y = [v for _, v in pts]
            label = "network only" if sims == 0 else f"{sims} sims"
            ax.plot(x, y, color=colour, marker="o", markersize=4, label=label)
        n = cfg["size"]
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-3, 103)
        ax.set_title(f"{n}x{n} Hex vs MoHex (1 s per move)")
        ax.set_xlabel("training compute (FLOPs)")
        ax.set_ylabel("theoretically-won games won (%)")
        ax.legend(loc="lower right", fontsize=9)
        ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def loss_figure(runs: list[Path], out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for colour, run in zip(SERIES, runs):
        cfg = json.loads((run / "config.json").read_text())
        rows = list(csv.DictReader(open(run / "metrics.csv")))
        it = [int(r["iter"]) for r in rows]
        label = f"{cfg['size']}x{cfg['size']}"
        a1.plot(it, [float(r["policy_loss"]) for r in rows], color=colour, label=label, linewidth=1.5)
        a2.plot(it, [float(r["value_loss"]) for r in rows], color=colour, label=label, linewidth=1.5)
    a1.set_title("Policy loss (cross-entropy to MCTS visits)")
    a2.set_title("Value loss (MSE to game result)")
    for a in (a1, a2):
        a.set_xlabel("training iteration")
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    runs = [Path(r) for r in sys.argv[1:]]
    out = Path("figures")
    out.mkdir(exist_ok=True)
    loss_figure(runs, out / "losses.png")
    with_sweep = [r for r in runs if (r / "sweep.csv").exists()]
    if with_sweep:
        strength_figure(with_sweep, out / "strength_vs_compute.png")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
