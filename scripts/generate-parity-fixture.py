#!/usr/bin/env python3
"""Generate deterministic cross-runtime Pathagon parity positions.

The fixture is intentionally rules-only input. TypeScript, Rust, and Python
each calculate the legal actions and every resulting state independently; the
parity harness compares those outputs byte-for-byte after JSON normalization.
"""

from __future__ import annotations

import json
import sys


def config(size: int, reserve: int) -> dict:
    return {
        "rulesVersion": "pathagon-rules-v1",
        "boardSize": size,
        "reservePerPlayer": reserve,
        "maxPlies": size * size * 4,
        "repetitionLimit": 3,
    }


def position(size: int, reserve: int, placements: dict[int, str], *, reserves: tuple[int, int] | None = None, turn: str = "light", forbidden: list[int] | None = None, last_light: int | None = None, last_dark: int | None = None, winner: str | None = None, ply: int = 0) -> dict:
    board = [None] * (size * size)
    for square, player in placements.items():
        if not 0 <= square < len(board) or board[square] is not None:
            raise ValueError(f"invalid generated placement {square}")
        board[square] = player
    light_reserve, dark_reserve = reserves or (reserve, reserve)
    return {
        "config": config(size, reserve),
        "board": board,
        "reserve": {"light": light_reserve, "dark": dark_reserve},
        "turn": turn,
        "forbidden": sorted(forbidden or []),
        "lastRelocatedTo": {"light": last_light, "dark": last_dark},
        "winner": winner,
        "ply": ply,
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for size in range(4, 8):
        cells = size * size
        reserve = size * 2
        cases.append({"name": f"{size}x{size}-initial", "position": position(size, reserve, {})})

        # A placement that can trigger the canonical A-B-A capture pattern,
        # plus a forbidden square that must be excluded from legal actions.
        origin = size + 1
        near = origin + size
        far = origin + 2 * size
        cases.append({
            "name": f"{size}x{size}-capture-placement",
            "position": position(size, reserve, {near: "dark", far: "light", cells - 1: "dark"}, forbidden=[0], ply=3),
        })

        # Movement phase: the just-relocated light piece is excluded as a
        # source, while an occupied and a forbidden square are excluded as
        # destinations.
        light_squares = [0, 1, size + 1]
        dark_squares = [size - 1, 2 * size - 1, cells - 2]
        placements = {square: "light" for square in light_squares}
        placements.update({square: "dark" for square in dark_squares})
        cases.append({
            "name": f"{size}x{size}-movement",
            "position": position(size, reserve, placements, reserves=(0, 0), turn="light", forbidden=[cells - 1], last_light=1, last_dark=None, ply=24),
        })

        dark_placements = {square: "dark" for square in light_squares}
        dark_placements.update({square: "light" for square in dark_squares})
        cases.append({
            "name": f"{size}x{size}-dark-movement",
            "position": position(size, reserve, dark_placements, reserves=(0, 0), turn="dark", forbidden=[cells - 1], last_light=None, last_dark=1, ply=25),
        })

        # Completing a straight light path exercises winner propagation in the
        # full post-action state, not just the boolean outcome.
        path = {row * size: "light" for row in range(1, size)}
        path[1] = "dark"  # keep the destination's row non-trivial
        cases.append({
            "name": f"{size}x{size}-path-completion",
            "position": position(size, 1, path, turn="light", ply=8),
        })

        dark_path = {column: "dark" for column in range(1, size)}
        dark_path[size] = "light"
        cases.append({
            "name": f"{size}x{size}-dark-path-completion",
            "position": position(size, 1, dark_path, turn="dark", ply=9),
        })
    return cases


def main() -> None:
    json.dump({"fixtureVersion": 1, "cases": build_cases()}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
