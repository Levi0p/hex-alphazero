"""Hex rules in JAX, written so thousands of games can run in parallel with vmap.

Board layout matches MoHex / HexGUI: an N x N rhombus where cell (r, c)
touches (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c).
Black connects the top edge to the bottom edge, White connects left to right.

The key trick: the board is always stored from the view of the player to move.
+1 is "my" stone, -1 is the opponent's, and "I" always connect top to bottom.
After every move we negate the board and transpose it. Transposing swaps the
two goal directions, and the hex neighbourhood above is symmetric under
transpose, so the next player again connects top to bottom. The network and
the search therefore only ever see one kind of position.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    board: jax.Array  # int8 [N*N], canonical: +1 = player to move, -1 = opponent
    black_to_move: jax.Array  # bool, only needed to map back to real colours
    done: jax.Array  # bool
    winner: jax.Array  # int8, +1 black won, -1 white won, 0 still playing


def init(n: int) -> State:
    return State(
        board=jnp.zeros(n * n, jnp.int8),
        black_to_move=jnp.array(True),
        done=jnp.array(False),
        winner=jnp.array(0, jnp.int8),
    )


def _neighbours(x: jax.Array) -> jax.Array:
    """OR of the six hex-neighbour shifts of a boolean [N, N] grid."""
    p = jnp.pad(x, 1)
    # p[1 + dr, 1 + dc] window gives x shifted so that out[r, c] = x[r + dr, c + dc]
    shifts = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    n = x.shape[0]
    out = jnp.zeros_like(x)
    for dr, dc in shifts:
        out = out | jax.lax.dynamic_slice(p, (1 + dr, 1 + dc), (n, n))
    return out


def connects_top_bottom(stones: jax.Array) -> jax.Array:
    """True if the boolean [N, N] grid has a chain from row 0 to row N-1."""
    reach = stones & (jnp.arange(stones.shape[0])[:, None] == 0)

    def grow(carry):
        reach, _ = carry
        new = stones & (reach | _neighbours(reach))
        return new, jnp.any(new != reach)

    reach, _ = jax.lax.while_loop(lambda c: c[1], grow, (reach, jnp.array(True)))
    return jnp.any(reach[-1])


def legal_mask(state: State) -> jax.Array:
    return (state.board == 0) & ~state.done


def step(state: State, action: jax.Array) -> tuple[State, jax.Array]:
    """Play `action` (canonical index) for the player to move.

    Returns the new state and the reward for the player who just moved:
    1.0 if this move won the game, else 0.0. Stepping a finished game is a no-op.
    """
    n = int(round(state.board.shape[0] ** 0.5))
    board = state.board.at[action].set(1)
    won = connects_top_bottom((board == 1).reshape(n, n)) & ~state.done
    flipped = (-board.reshape(n, n).T).reshape(-1)
    mover = jnp.where(state.black_to_move, 1, -1).astype(jnp.int8)
    new = State(
        board=jnp.where(state.done, state.board, flipped),
        black_to_move=jnp.where(state.done, state.black_to_move, ~state.black_to_move),
        done=state.done | won,
        winner=jnp.where(won, mover, state.winner),
    )
    return new, won.astype(jnp.float32)


def observation(state: State) -> jax.Array:
    """Network input [N, N, 3]: my stones, opponent stones, a plane of ones.

    The ones plane lets the convolutions tell the board edge (zero padding)
    apart from empty cells.
    """
    n = int(round(state.board.shape[0] ** 0.5))
    b = state.board.reshape(n, n)
    return jnp.stack([b == 1, b == -1, jnp.ones_like(b, bool)], -1).astype(jnp.float32)


def real_board(state: State) -> jax.Array:
    """[N, N] board in real colours: +1 black, -1 white."""
    n = int(round(state.board.shape[0] ** 0.5))
    b = state.board.reshape(n, n)
    return jnp.where(state.black_to_move, b, -b.T)


def to_real_action(state: State, action: jax.Array) -> jax.Array:
    """Canonical action index -> real board index (row * N + col)."""
    n = int(round(state.board.shape[0] ** 0.5))
    r, c = action // n, action % n
    return jnp.where(state.black_to_move, action, c * n + r)


def to_canonical_action(state: State, real_action: jax.Array) -> jax.Array:
    # The mapping is a transpose, so it is its own inverse.
    return to_real_action(state, real_action)


def cell_name(real_action: int, n: int) -> str:
    """Real index -> HTP/MoHex name, e.g. 0 -> 'a1' (column letter, row number)."""
    r, c = divmod(int(real_action), n)
    return f"{chr(ord('a') + c)}{r + 1}"


def cell_index(name: str, n: int) -> int:
    c = ord(name[0].lower()) - ord("a")
    r = int(name[1:]) - 1
    return r * n + c


def render(state: State) -> str:
    b = jax.device_get(real_board(state))
    n = b.shape[0]
    sym = {1: "B", -1: "W", 0: "."}
    lines = ["  " + "  ".join(chr(ord("a") + c) for c in range(n))]
    for r in range(n):
        lines.append(" " * r + f"{r + 1:>2}\\" + "  ".join(sym[int(v)] for v in b[r]) + "\\")
    return "\n".join(lines)
