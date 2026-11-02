"""MCTS with a know-nothing network (uniform prior, value 0) should still find
the right moves on 3x3, because finished games give exact values that get
backed up through the tree. We check it against the exact solver. Near-empty
boards need a few thousand simulations without a trained network (400 misses
16 of 3401 positions, 1500 misses 4, 4000 misses none)."""

import jax
import jax.numpy as jnp
import numpy as np

from hexaz import hex as H
from hexaz import mcts, solver


def uniform_eval(obs):
    b = obs.shape[0]
    return jnp.zeros((b, obs.shape[1] * obs.shape[2])), jnp.zeros(b)


def to_state(board: tuple) -> H.State:
    # black_to_move is irrelevant for search; the canonical board is all it needs.
    return H.State(
        board=jnp.array(board, jnp.int8),
        black_to_move=jnp.array(True),
        done=jnp.array(False),
        winner=jnp.array(0, jnp.int8),
    )


def test_finds_winning_moves_3x3():
    won = [p for p in solver.all_positions(3) if solver.wins(p)]
    for sims, positions in [
        (4000, [p for p in won if sum(map(abs, p)) <= 3]),
        (400, [p for p in won if sum(map(abs, p)) > 3]),
    ]:
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *[to_state(p) for p in positions])
        run = jax.jit(lambda s, k: mcts.search(uniform_eval, s, k, sims, noise_fraction=0.0))
        out = run(states, jax.random.PRNGKey(0))
        best = np.asarray(out.visit_counts).argmax(-1)
        wrong = [p for p, a in zip(positions, best) if a not in solver.winning_moves(p)]
        assert not wrong, f"{len(wrong)}/{len(positions)} positions missed a win at {sims} sims"
        # Root values of won positions should approach +1.
        assert float(np.asarray(out.root_value).mean()) > 0.5


def test_visit_counts_sum():
    s = jax.vmap(lambda _: H.init(5))(jnp.arange(4))
    out = mcts.search(uniform_eval, s, jax.random.PRNGKey(1), 50)
    np.testing.assert_allclose(np.asarray(out.visit_counts).sum(-1), 50)


def test_solver_first_player_wins():
    # Hex strategy-stealing: the first player wins on every board.
    assert solver.wins(tuple([0] * 9))
