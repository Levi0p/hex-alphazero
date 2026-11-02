"""Our agent vs MoHex, over every opening.

For each of the N*N cells we force Black's first move there, then play two
games: agent as Black and agent as White, MoHex taking the other side. MoHex's
solver tells us who wins each opening with perfect play, so "matching MoHex"
has an exact meaning: the agent wins every game whose opening is a
theoretical win for it.

With --referee, before every agent move we also ask the solver whether the
agent can force a win here, and if so, whether its move kept the win.

    python -m hexaz.arena --run runs/hex5_seed0 --sims 64
"""

import argparse
import json
import time
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from hexaz import hex as H
from hexaz import mcts
from hexaz.mohex import MoHex
from hexaz.net import PolicyValueNet


class Agent:
    def __init__(self, run_dir: str, sims: int, ckpt: str | None = None):
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
        self.n, self.sims = cfg["size"], sims
        self.net = PolicyValueNet(cfg["channels"], cfg["blocks"])
        params = self.net.init(jax.random.PRNGKey(0), jnp.zeros((1, self.n, self.n, 3)))
        path = Path(ckpt) if ckpt else Path(run_dir) / "params.msgpack"
        self.params = flax.serialization.from_bytes(params, path.read_bytes())
        self._template = params
        self._search = jax.jit(
            lambda p, s: mcts.search(
                lambda o: self.net.apply(p, o), s, jax.random.PRNGKey(0), max(sims, 1), noise_fraction=0.0
            )
        )
        self._policy = jax.jit(self.net.apply)

    def load(self, ckpt: str):
        """Swap in another checkpoint of the same network (no recompilation)."""
        self.params = flax.serialization.from_bytes(self._template, Path(ckpt).read_bytes())

    def move(self, state: H.State) -> int:
        """Real board index of the agent's move."""
        batch = jax.tree.map(lambda x: x[None], state)
        if self.sims == 0:
            logits, _ = self._policy(self.params, jax.vmap(H.observation)(batch))
            a = int(jnp.where(state.board == 0, logits[0], -jnp.inf).argmax())
        else:
            a = int(self._search(self.params, batch).visit_counts[0].argmax())
        return int(H.to_real_action(state, a))


def play_game(agent: Agent, mohex: MoHex, opening: int, agent_black: bool, referee: bool,
              opp_random: float = 0.0, rng: np.random.Generator | None = None):
    """One game from a forced first move. With opp_random > 0 the opponent plays a
    random legal move that share of the time instead of MoHex's move, which drags
    the agent into odd positions it never meets against MoHex."""
    n = agent.n
    state = H.init(n)
    mohex.new_game(n)
    moves, checks = [], []

    def apply(real: int, tell_mohex: bool = True):
        nonlocal state
        colour = "b" if bool(state.black_to_move) else "w"
        if tell_mohex:  # MoHex's own genmove moves are already on its board
            mohex.play(colour, H.cell_name(real, n))
        state, _ = H.step(state, H.to_canonical_action(state, real))
        moves.append(H.cell_name(real, n))

    apply(opening)
    while not bool(state.done):
        colour = "b" if bool(state.black_to_move) else "w"
        if bool(state.black_to_move) == agent_black:
            winnable = referee and mohex.solve(colour) == colour
            apply(agent.move(state))
            if winnable:  # could we force a win here? then did our move keep it?
                other = "w" if colour == "b" else "b"
                checks.append(bool(state.done) or mohex.solve(other) == colour)
        elif opp_random > 0 and rng.random() < opp_random:
            empty = np.flatnonzero(np.asarray(H.real_board(state)).reshape(-1) == 0)
            apply(int(rng.choice(empty)))
        else:
            reply = mohex.genmove(colour)
            if reply == "resign":
                return (-1 if colour == "b" else 1), moves + ["resign"], checks
            apply(H.cell_index(reply, n), tell_mohex=False)
    return int(state.winner), moves, checks


def opening_labels(mohex: MoHex, n: int) -> set[str]:
    """The opening cells that win for Black, proven one by one with the solver.
    Slow on 7x7 (a few minutes), so cached in results/."""
    cache = Path(__file__).resolve().parent.parent / "results" / f"openings_{n}x{n}.json"
    if cache.exists():
        return set(json.loads(cache.read_text())["black_wins"])
    wins = []
    for i in range(n * n):
        cell = H.cell_name(i, n)
        mohex.new_game(n)
        mohex.play("b", cell)
        if mohex.solve("w") == "b":
            wins.append(cell)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"board": n, "black_wins": wins}, indent=2))
    return set(wins)


def opening_sweep(agent: Agent, mohex: MoHex, referee: bool = False, black_wins=None, verbose=True):
    """Play every opening with the agent on each side. Returns (summary, games)."""
    n = agent.n
    if black_wins is None:
        black_wins = opening_labels(mohex, n)
    results = []
    start = time.time()
    for opening in range(n * n):
        cell = H.cell_name(opening, n)
        for agent_black in (True, False):
            winner, moves, checks = play_game(agent, mohex, opening, agent_black, referee)
            agent_should_win = (cell in black_wins) == agent_black
            agent_won = (winner == 1) == agent_black
            results.append(
                dict(opening=cell, agent="black" if agent_black else "white",
                     should_win=agent_should_win, won=agent_won, moves=moves,
                     winnable_positions=len(checks), kept_win=int(sum(checks)))
            )
            if verbose:
                tag = "ok" if agent_won == agent_should_win else ("UPSET" if agent_won else "MISSED")
                print(f"{cell:>3} agent={results[-1]['agent']:5}  theory={'win ' if agent_should_win else 'loss'}"
                      f"  result={'win ' if agent_won else 'loss'}  {tag}  ({time.time() - start:.0f}s)", flush=True)

    should = [r for r in results if r["should_win"]]
    summary = dict(
        board=n, sims=agent.sims,
        games=len(results),
        should_win=len(should), won_of_should_win=sum(r["won"] for r in should),
        upsets=sum(r["won"] for r in results if not r["should_win"]),
        winnable_positions=sum(r["winnable_positions"] for r in results),
        kept_win=sum(r["kept_win"] for r in results),
    )
    return summary, results


def exam(agent: Agent, mohex: MoHex, games: int, opp_random: float, seed: int = 0):
    """Referee every agent move in games against an erratic opponent (random
    openings, random colours, random moves mixed into MoHex's play). Returns
    (winnable positions seen, positions where the agent kept the win)."""
    rng = np.random.default_rng(seed)
    seen = kept = 0
    start = time.time()
    for g in range(games):
        opening = int(rng.integers(agent.n**2))
        _, _, checks = play_game(agent, mohex, opening, bool(g % 2 == 0), True, opp_random, rng)
        seen += len(checks)
        kept += sum(checks)
        if (g + 1) % 20 == 0:
            print(f"exam {g + 1}/{games}: kept the win in {kept}/{seen} winnable positions"
                  f"  ({time.time() - start:.0f}s)", flush=True)
    return seen, kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--mohex-time", type=float, default=10.0, help="seconds per MoHex move")
    p.add_argument("--referee", action="store_true")
    p.add_argument("--exam", type=int, default=0, help="run N referee'd games vs an erratic opponent instead")
    p.add_argument("--opp-random", type=float, default=0.3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    agent = Agent(args.run, args.sims, args.ckpt)
    if args.exam:
        with MoHex(max_time=args.mohex_time) as mohex:
            seen, kept = exam(agent, mohex, args.exam, args.opp_random)
        summary = dict(board=agent.n, sims=args.sims, games=args.exam, opp_random=args.opp_random,
                       mohex_time=args.mohex_time, winnable_positions=seen, kept_win=kept)
        print(json.dumps(summary, indent=2))
        out = Path(args.out or Path(args.run) / f"exam_sims{args.sims}.json")
        out.write_text(json.dumps(summary, indent=2))
        return

    with MoHex(max_time=args.mohex_time) as mohex:
        summary, results = opening_sweep(agent, mohex, args.referee)
    summary["mohex_time"] = args.mohex_time
    print(json.dumps(summary, indent=2))
    out = Path(args.out or Path(args.run) / f"arena_sims{args.sims}.json")
    out.write_text(json.dumps({"summary": summary, "games": results}, indent=2))


if __name__ == "__main__":
    main()
