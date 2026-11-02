"""Exact Hex solver for tiny boards (3x3, 4x4), used to test the search and to
measure perfect play. Bigger boards use MoHex's solver instead.

Works on canonical boards (tuples, +1 = player to move, who connects top to
bottom), the same representation as hexaz.hex.
"""

from functools import lru_cache

import numpy as np

NEIGHBOURS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def _connects(stones: np.ndarray) -> bool:
    n = stones.shape[0]
    stack = [(0, c) for c in range(n) if stones[0, c]]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if r == n - 1:
            return True
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and stones[rr, cc]:
                stack.append((rr, cc))
    return False


def _after(board: tuple, a: int, n: int) -> tuple[tuple, bool]:
    """Play a, return (next canonical board, did the mover win)."""
    b = np.array(board, np.int8).reshape(n, n)
    b[divmod(a, n)] = 1
    won = _connects(b == 1)
    return tuple((-b.T).reshape(-1).tolist()), won


@lru_cache(maxsize=None)
def wins(board: tuple) -> bool:
    """True if the player to move wins with perfect play."""
    n = int(round(len(board) ** 0.5))
    for a in range(len(board)):
        if board[a] == 0:
            nxt, won = _after(board, a, n)
            if won or not wins(nxt):
                return True
    return False


def winning_moves(board: tuple) -> list[int]:
    n = int(round(len(board) ** 0.5))
    out = []
    for a in range(len(board)):
        if board[a] == 0:
            nxt, won = _after(board, a, n)
            if won or not wins(nxt):
                out.append(a)
    return out


def all_positions(n: int) -> list[tuple]:
    """Every unfinished position reachable from the empty board."""
    seen, stack = set(), [tuple([0] * (n * n))]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        for a in range(len(b)):
            if b[a] == 0:
                nxt, won = _after(b, a, n)
                if not won:
                    stack.append(nxt)
    return list(seen)
