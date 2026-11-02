"""Strength over training, at several amounts of search.

For each saved checkpoint and each test-time simulation count, run the full
opening sweep against MoHex and record how many theoretically won games the
agent actually won. This is the small-scale version of Jones (2021): how do
training compute and test-time search trade off?

    python -m hexaz.sweep --run runs/hex5_seed0 --sims 0 16 64 256
"""

import argparse
import csv
import time
from pathlib import Path

from hexaz.arena import Agent, opening_labels, opening_sweep
from hexaz.mohex import MoHex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--sims", type=int, nargs="+", default=[0, 16, 64, 256])
    p.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    p.add_argument("--mohex-time", type=float, default=1.0)
    args = p.parse_args()

    run = Path(args.run)
    secs = {int(r["iter"]): float(r["secs"]) for r in csv.DictReader(open(run / "metrics.csv"))}
    ckpts = sorted((run / "ckpt").glob("*.msgpack"))[:: args.every]
    out_path = run / "sweep.csv"
    done = set()
    if out_path.exists():
        done = {(int(r["iter"]), int(r["sims"])) for r in csv.DictReader(open(out_path))}
    new_file = not out_path.exists()
    out = csv.writer(open(out_path, "a", buffering=1))
    if new_file:
        out.writerow(["iter", "train_secs", "sims", "should_win", "won_of_should_win", "upsets"])

    start = time.time()
    with MoHex(max_time=args.mohex_time) as mohex:
        black_wins = None
        for sims in args.sims:
            agent = Agent(args.run, sims, str(ckpts[0]))
            if black_wins is None:
                black_wins = opening_labels(mohex, agent.n)
            for ck in ckpts:
                it = int(ck.stem)
                if (it, sims) in done:
                    continue
                agent.load(str(ck))
                s, _ = opening_sweep(agent, mohex, black_wins=black_wins, verbose=False)
                out.writerow([it, secs.get(it, ""), sims, s["should_win"], s["won_of_should_win"], s["upsets"]])
                print(f"iter {it:5d} sims {sims:4d}: won {s['won_of_should_win']}/{s['should_win']}"
                      f"  upsets {s['upsets']}  ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
