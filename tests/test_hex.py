"""Check the JAX rules against a plain-Python Hex written the obvious way."""

import random

import jax
import jax.numpy as jnp
import numpy as np

from hexaz import hex as H

NEIGHBOURS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def ref_winner(board: np.ndarray) -> int:
    """board in real colours. BFS from Black's top row and White's left column."""
    n = board.shape[0]
    for colour, starts, at_goal in [
        (1, [(0, c) for c in range(n)], lambda r, c: r == n - 1),
        (-1, [(r, 0) for r in range(n)], lambda r, c: c == n - 1),
    ]:
        seen = set()
        stack = [s for s in starts if board[s] == colour]
        while stack:
            r, c = stack.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            if at_goal(r, c):
                return colour
            for dr, dc in NEIGHBOURS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == colour:
                    stack.append((rr, cc))
    return 0


def test_random_games_match_reference():
    step = jax.jit(H.step)
    for n in [3, 4, 5, 7]:
        for seed in range(15):
            rng = random.Random(seed)
            s = H.init(n)
            real = np.zeros((n, n), int)
            black = True
            while not bool(s.done):
                legal = np.flatnonzero(np.asarray(H.legal_mask(s)))
                a = int(rng.choice(legal))
                ra = int(H.to_real_action(s, a))
                real[divmod(ra, n)] = 1 if black else -1
                s, _ = step(s, a)
                black = not black
                np.testing.assert_array_equal(np.asarray(H.real_board(s)), real)
                assert int(s.winner) == ref_winner(real)
            # Hex has no draws: someone always wins before the board fills.
            assert int(s.winner) != 0


def test_known_chain():
    # Black column down the middle of 3x3: b1, b2, b3 (with White replies elsewhere).
    s = H.init(3)
    for real in ["b1", "a1", "b2", "a2", "b3"]:
        a = H.to_canonical_action(s, H.cell_index(real, 3))
        s, r = H.step(s, a)
    assert bool(s.done) and int(s.winner) == 1 and float(r) == 1.0


def test_diagonal_is_not_connected():
    # a1-b2 are not neighbours on this layout (b1-a2 are).
    g = jnp.zeros((2, 2), bool).at[0, 0].set(True).at[1, 1].set(True)
    assert not bool(H.connects_top_bottom(g))
    g = jnp.zeros((2, 2), bool).at[0, 1].set(True).at[1, 0].set(True)
    assert bool(H.connects_top_bottom(g))


def test_vmap():
    n, b = 5, 64
    states = jax.vmap(lambda _: H.init(n))(jnp.arange(b))
    key = jax.random.PRNGKey(0)
    step = jax.jit(jax.vmap(H.step))
    for _ in range(n * n):
        key, k = jax.random.split(key)
        logits = jnp.where(jax.vmap(H.legal_mask)(states), 0.0, -jnp.inf)
        a = jax.random.categorical(k, logits)
        a = jnp.where(states.done, 0, a)
        states, _ = step(states, a)
    assert bool(jnp.all(states.done))
