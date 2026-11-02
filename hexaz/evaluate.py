"""Measuring strength.

On 3x3 and 4x4 we can check every position exactly with hexaz.solver: in each
position the player to move can win, does the network's top move keep the win?
Bigger boards are measured against MoHex (see arena.py).
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from hexaz import solver


@lru_cache(maxsize=None)
def _solved(n: int):
    positions = solver.all_positions(n)
    obs, best, wins = [], [], []
    for p in positions:
        b = np.array(p).reshape(n, n)
        obs.append(np.stack([b == 1, b == -1, np.ones_like(b, bool)], -1))
        w = solver.wins(p)
        wins.append(w)
        mask = np.zeros(n * n, bool)
        if w:
            mask[solver.winning_moves(p)] = True
        best.append(mask)
    return np.array(obs, np.float32), np.array(best), np.array(wins)


def solver_accuracy(apply_fn, params, n: int) -> dict:
    """policy_acc: share of winnable positions where argmax(policy) wins.
    value_acc: share of all positions where sign(value) matches the truth."""
    obs, best, wins = _solved(n)
    logits, value = jax.jit(apply_fn)(params, jnp.asarray(obs))
    legal = (obs[..., 0] + obs[..., 1]).reshape(len(obs), -1) == 0
    move = np.asarray(jnp.where(legal, logits, -jnp.inf)).argmax(-1)
    policy_ok = best[np.arange(len(move)), move][wins]
    value_ok = (np.asarray(value) > 0) == wins
    return {"policy_acc": float(policy_ok.mean()), "value_acc": float(value_ok.mean())}


def agent_moves(apply_fn, params, boards: list[tuple], n: int, sims: int) -> np.ndarray:
    """The move the agent actually plays in each canonical board: argmax of the
    network policy if sims == 0, otherwise argmax of MCTS visits (no noise)."""
    from hexaz import hex as H
    from hexaz import mcts

    board = jnp.array(boards, jnp.int8)
    if sims == 0:
        obs = jax.vmap(H.observation)(H.State(board, None, None, None))
        logits, _ = apply_fn(params, obs)
        return np.asarray(jnp.where(board == 0, logits, -jnp.inf)).argmax(-1)
    k = len(boards)
    states = H.State(
        board=board,
        black_to_move=jnp.ones(k, bool),
        done=jnp.zeros(k, bool),
        winner=jnp.zeros(k, jnp.int8),
    )
    out = mcts.search(lambda o: apply_fn(params, o), states, jax.random.PRNGKey(0), sims, noise_fraction=0.0)
    return np.asarray(out.visit_counts).argmax(-1)


def perfect_play_check(apply_fn, params, n: int, sims: int) -> dict:
    """Play the agent against every possible opponent, from both sides.

    Walks the whole game tree where the agent picks one move and the opponent
    tries all moves. A mistake is any position where the agent could win but
    its move throws the win away. Zero mistakes as first player means it wins
    every game; zero as second player means it punishes every opponent error.
    """
    from hexaz.solver import _after

    result = {}
    for role, agent_first in [("first", True), ("second", False)]:
        empty = tuple([0] * (n * n))
        agent_layer = {empty} if agent_first else set()
        if not agent_first:
            for a in range(n * n):
                agent_layer.add(_after(empty, a, n)[0])
        checked, mistakes = 0, 0
        while agent_layer:
            boards = sorted(agent_layer)
            moves = agent_moves(apply_fn, params, boards, n, sims)
            opp_layer = set()
            for b, a in zip(boards, moves):
                checked += 1
                if solver.wins(b) and a not in solver.winning_moves(b):
                    mistakes += 1
                nxt, won = _after(b, int(a), n)
                if not won:
                    opp_layer.add(nxt)
            agent_layer = set()
            for b in opp_layer:
                for a in range(n * n):
                    if b[a] == 0:
                        nxt, won = _after(b, a, n)
                        if not won:
                            agent_layer.add(nxt)
        result[role] = {"positions": checked, "mistakes": mistakes}
    return result
