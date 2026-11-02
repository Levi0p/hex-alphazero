"""Talking to MoHex over GTP (the text protocol Go/Hex engines use).

MoHex 2.0 (Benzene project, University of Alberta) won the Computer Olympiad
Hex tournaments and solves small boards exactly. We use it two ways:
  - as an opponent: `genmove`,
  - as a referee: `dfpn-solve-state` proves who wins a position with perfect
    play, so we can tell exactly whether a move threw away a win.

Note: `dfpn-solver-find-winning` looks like the right referee but is not: it
skips moves it can prove are dominated by another move, so on 7x7 it leaves
out openings that do win (e.g. e6 and g1). We only use it in tests.

Colours match hexaz.hex: Black connects top to bottom, cells are named like
'c3' (column letter, row number).
"""

import subprocess
from pathlib import Path

DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "third_party" / "mohex"


class MoHex:
    def __init__(self, binary: Path = DEFAULT_BINARY, max_time: float | None = None, threads: int = 1):
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=binary.parent,
        )
        self.cmd(f"param_mohex num_threads {threads}")
        if max_time is not None:
            self.cmd(f"param_mohex max_time {max_time}")
            # Without this MoHex ignores max_time and uses its own time budget.
            self.cmd("param_mohex use_time_management 0")

    def cmd(self, line: str) -> str:
        """Send one GTP command, return the response text (without '= ')."""
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            out = self.proc.stdout.readline()
            if out == "":
                raise RuntimeError(f"MoHex exited while running {line!r}")
            if out.strip() == "" and lines:
                break
            if out.strip() == "" and not lines:
                continue
            lines.append(out.rstrip("\n"))
        text = "\n".join(lines)
        if text.startswith("?"):
            raise RuntimeError(f"MoHex error for {line!r}: {text}")
        return text[1:].strip()

    def new_game(self, n: int):
        self.cmd(f"boardsize {n} {n}")
        self.cmd("clear_board")

    def play(self, colour: str, cell: str):
        self.cmd(f"play {colour} {cell}")

    def genmove(self, colour: str) -> str:
        return self.cmd(f"genmove {colour}").lower()

    def solve(self, to_move: str) -> str:
        """Exact winner ('b' or 'w') of the current position with `to_move` to play."""
        out = self.cmd(f"dfpn-solve-state {to_move}")
        return {"black": "b", "white": "w"}[out]

    def winning_moves(self, colour: str) -> list[str]:
        """Winning moves for `colour`, minus any MoHex proves dominated (so the
        list can be incomplete; see the module docstring)."""
        out = self.cmd(f"dfpn-solver-find-winning {colour}")
        return [m.lower() for m in out.split()]

    def close(self):
        try:
            self.cmd("quit")
        except Exception:
            pass
        self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
