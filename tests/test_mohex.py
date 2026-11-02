"""The MoHex bridge: colours and cell names must agree with hexaz.hex."""

import pytest

from hexaz import hex as H
from hexaz.mohex import DEFAULT_BINARY, MoHex

pytestmark = pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="MoHex not built")


def test_black_connects_top_to_bottom():
    # Black has a1-a4 down column a; a5 completes top-to-bottom and is the only win.
    with MoHex() as m:
        m.new_game(5)
        for b, w in [("a1", "c3"), ("a2", "d3"), ("a3", "c4"), ("a4", "e5")]:
            m.play("b", b)
            m.play("w", w)
        assert m.winning_moves("b") == ["a5"]
        # Our rules agree that a5 wins.
        s = H.init(5)
        for cell in ["a1", "c3", "a2", "d3", "a3", "c4", "a4", "e5", "a5"]:
            s, _ = H.step(s, H.to_canonical_action(s, H.cell_index(cell, 5)))
        assert bool(s.done) and int(s.winner) == 1


def test_matches_our_solver_on_3x3():
    from hexaz import solver

    with MoHex() as m:
        m.new_game(3)
        ours = sorted(H.cell_name(a, 3) for a in solver.winning_moves(tuple([0] * 9)))
        assert sorted(m.winning_moves("b")) == ours


def test_find_winning_skips_dominated_wins():
    # Why the referee uses dfpn-solve-state: on 7x7, find-winning leaves out
    # the winning opening e6 (it is "dominated"), but solving the position
    # after e6 proves Black wins.
    with MoHex() as m:
        m.new_game(7)
        assert "e6" not in m.winning_moves("b")
        m.play("b", "e6")
        assert m.solve("w") == "b"
