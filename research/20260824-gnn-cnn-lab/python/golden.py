"""Persistent, historyless Pathagon golden-position tables.

The table format is intentionally smaller than the replay corpus.  A row is
one symmetry-canonical position key followed by one exact W/L/D byte.  An
absent key means unknown; model scores and provenance belong in separate
sidecars.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional

from .game import BoardConfig, GameState, Player
from .symmetry import ALL_SYMMETRIES, transform_state


VALUE_BYTES = 1
BOARD_SIZE = 7
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
RESERVE_PER_PLAYER = 14


def marker_bits_for_board_size(board_size: int) -> int:
    """Return bits for one square marker plus its ``none`` sentinel."""

    if board_size < 3:
        raise ValueError("Pathagon boards need at least 3 rows")
    return (board_size * board_size).bit_length()


def key_bits_for_board_size(board_size: int) -> int:
    cells = board_size * board_size
    return 2 * cells + 1 + 2 * marker_bits_for_board_size(board_size)


def key_bytes_for_board_size(board_size: int) -> int:
    return (key_bits_for_board_size(board_size) + 7) // 8


# Default constants retain the original 7x7/14-reserve API while the codec
# itself supports the smaller boards used by historical curriculum runs.
KEY_BYTES = key_bytes_for_board_size(BOARD_SIZE)
ROW_BYTES = KEY_BYTES + VALUE_BYTES

# Values are from the side-to-move perspective.  Unknown is represented by
# absence, not by a row value.
LOSS = 0
DRAW = 1
WIN = 2
VALID_OUTCOMES = frozenset({LOSS, DRAW, WIN})


class GoldenConflictError(ValueError):
    """Raised when the same canonical key receives contradictory exact values."""


@dataclass(frozen=True)
class UnpackedKey:
    """The rule-relevant fields represented by a packed square-board key."""

    light: int
    dark: int
    forbidden: int
    turn: Player
    last_relocated_to: tuple[Optional[int], Optional[int]]


def _validate_state(
    state: GameState,
    board_size: Optional[int] = None,
    reserve_per_player: Optional[int] = None,
) -> None:
    config = state.config
    expected_size = config.size if board_size is None else board_size
    if config.size != expected_size:
        raise ValueError(f"state board size {config.size} does not match {expected_size}x{expected_size} table")
    if reserve_per_player is not None and config.reserve_per_player != reserve_per_player:
        raise ValueError(
            f"state reserve {config.reserve_per_player} does not match table reserve {reserve_per_player}"
        )
    cells = expected_size * expected_size
    full = (1 << cells) - 1
    if state.light < 0 or state.dark < 0 or state.forbidden < 0:
        raise ValueError("position masks must be non-negative")
    if (state.light | state.dark | state.forbidden) & ~full:
        raise ValueError(f"position mask is outside the {expected_size}x{expected_size} board")
    if state.light & state.dark or state.forbidden & (state.light | state.dark):
        raise ValueError("position masks overlap")
    for marker in state.last_relocated_to:
        if marker is not None and not 0 <= marker < cells:
            raise ValueError(f"relocation marker is outside the {expected_size}x{expected_size} board")


def _marker_code(marker: Optional[int], cells: int) -> int:
    return cells if marker is None else marker


def _pack_untransformed(state: GameState) -> bytes:
    _validate_state(state)
    cells = state.config.cell_count
    marker_bits = marker_bits_for_board_size(state.config.size)
    key_bytes = key_bytes_for_board_size(state.config.size)

    # Two bits per square are denser than three independent 64-bit masks:
    # 00 empty, 01 light, 10 dark, 11 forbidden.
    packed = 0
    for square in range(cells):
        mask = 1 << square
        code = 1 if state.light & mask else 2 if state.dark & mask else 3 if state.forbidden & mask else 0
        packed |= code << (2 * square)

    packed |= int(state.turn) << (2 * cells)
    packed |= _marker_code(state.last_relocated_to[Player.LIGHT], cells) << (2 * cells + 1)
    packed |= _marker_code(state.last_relocated_to[Player.DARK], cells) << (2 * cells + 1 + marker_bits)
    return packed.to_bytes(key_bytes, "little")


def pack_position_key(state: GameState) -> bytes:
    """Pack a position without applying geometric canonicalization."""

    return _pack_untransformed(state)


def canonical_position_key(state: GameState) -> bytes:
    """Return the lexicographically smallest key across all eight D4 transforms."""

    _validate_state(state)
    return min(_pack_untransformed(transform_state(state, symmetry)) for symmetry in ALL_SYMMETRIES)


def unpack_position_key(key: bytes, board_size: int = BOARD_SIZE) -> UnpackedKey:
    """Decode a packed key for inspection and collision verification."""

    cells = board_size * board_size
    marker_bits = marker_bits_for_board_size(board_size)
    key_bytes = key_bytes_for_board_size(board_size)
    total_bits = key_bits_for_board_size(board_size)
    if len(key) != key_bytes:
        raise ValueError(f"position key must be exactly {key_bytes} bytes")
    packed = int.from_bytes(key, "little")
    if packed >> total_bits:
        raise ValueError("reserved key bits must be zero")
    light = dark = forbidden = 0
    for square in range(cells):
        code = (packed >> (2 * square)) & 0b11
        if code == 1:
            light |= 1 << square
        elif code == 2:
            dark |= 1 << square
        elif code == 3:
            forbidden |= 1 << square
    turn = Player((packed >> (2 * cells)) & 1)

    def decode_marker(shift: int) -> Optional[int]:
        marker = (packed >> shift) & ((1 << marker_bits) - 1)
        if marker == cells:
            return None
        if marker >= cells:
            raise ValueError(f"relocation marker is outside the {board_size}x{board_size} board")
        return marker

    marker_shift = 2 * cells + 1
    return UnpackedKey(
        light=light,
        dark=dark,
        forbidden=forbidden,
        turn=turn,
        last_relocated_to=(decode_marker(marker_shift), decode_marker(marker_shift + marker_bits)),
    )


def _validate_key(key: bytes, board_size: int = BOARD_SIZE) -> None:
    key_bytes = key_bytes_for_board_size(board_size)
    if len(key) != key_bytes:
        raise ValueError(f"position key must be exactly {key_bytes} bytes")
    # This also rejects the reserved high bit and malformed marker values.
    unpack_position_key(key, board_size)


class GoldenTable:
    """An exact-key builder that rejects contradictory updates."""

    def __init__(
        self,
        entries: Optional[Mapping[bytes, int]] = None,
        *,
        board_size: int = BOARD_SIZE,
        reserve_per_player: int = RESERVE_PER_PLAYER,
    ) -> None:
        if reserve_per_player < 1:
            raise ValueError("reserve_per_player must be positive")
        self.board_size = board_size
        self.reserve_per_player = reserve_per_player
        self.key_bytes = key_bytes_for_board_size(board_size)
        self.row_bytes = self.key_bytes + VALUE_BYTES
        self._entries: dict[bytes, int] = {}
        if entries:
            for key, outcome in entries.items():
                self.put_key(key, outcome)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[bytes, int]]:
        yield from self.rows()

    def rows(self) -> Iterator[tuple[bytes, int]]:
        for key in sorted(self._entries):
            yield key, self._entries[key]

    def lookup_key(self, key: bytes) -> Optional[int]:
        _validate_key(key, self.board_size)
        return self._entries.get(key)

    def lookup(self, state: GameState) -> Optional[int]:
        _validate_state(state, self.board_size, self.reserve_per_player)
        return self.lookup_key(canonical_position_key(state))

    def put_key(self, key: bytes, outcome: int) -> None:
        _validate_key(key, self.board_size)
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}")
        previous = self._entries.get(key)
        if previous is not None and previous != outcome:
            raise GoldenConflictError(
                f"canonical key already has outcome {previous}, cannot replace with {outcome}"
            )
        self._entries[key] = outcome

    def put(self, state: GameState, outcome: int) -> bytes:
        _validate_state(state, self.board_size, self.reserve_per_player)
        key = canonical_position_key(state)
        self.put_key(key, outcome)
        return key

    def write(self, path: Path) -> int:
        """Write sorted fixed-width rows atomically and return the row count."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                for key, outcome in self.rows():
                    output.write(key)
                    output.write(bytes((outcome,)))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o644)
            return len(self)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


class FlatGoldenTable:
    """Read-only binary lookup over a sorted golden shard."""

    def __init__(
        self,
        path: Path,
        *,
        board_size: int = BOARD_SIZE,
        reserve_per_player: int = RESERVE_PER_PLAYER,
    ) -> None:
        self.path = Path(path)
        self.board_size = board_size
        self.reserve_per_player = reserve_per_player
        self.key_bytes = key_bytes_for_board_size(board_size)
        self.row_bytes = self.key_bytes + VALUE_BYTES
        self._file = self.path.open("rb")
        size = self.path.stat().st_size
        if size % self.row_bytes:
            self.close()
            raise ValueError(f"golden shard size must be a multiple of {self.row_bytes} bytes")
        self.rows = size // self.row_bytes

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "FlatGoldenTable":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def lookup_key(self, key: bytes) -> Optional[int]:
        _validate_key(key, self.board_size)
        low, high = 0, self.rows
        while low < high:
            middle = (low + high) // 2
            self._file.seek(middle * self.row_bytes)
            row = self._file.read(self.row_bytes)
            row_key = row[:self.key_bytes]
            if row_key < key:
                low = middle + 1
            elif row_key > key:
                high = middle
            else:
                return row[self.key_bytes]
        return None

    def lookup(self, state: GameState) -> Optional[int]:
        _validate_state(state, self.board_size, self.reserve_per_player)
        return self.lookup_key(canonical_position_key(state))


def rows_sha256(path: Path) -> str:
    """Hash a shard without loading it into memory."""

    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BOARD_SIZE",
    "CELL_COUNT",
    "DRAW",
    "FlatGoldenTable",
    "GoldenConflictError",
    "GoldenTable",
    "KEY_BYTES",
    "LOSS",
    "RESERVE_PER_PLAYER",
    "ROW_BYTES",
    "UnpackedKey",
    "VALUE_BYTES",
    "VALID_OUTCOMES",
    "WIN",
    "canonical_position_key",
    "key_bits_for_board_size",
    "key_bytes_for_board_size",
    "marker_bits_for_board_size",
    "pack_position_key",
    "rows_sha256",
    "unpack_position_key",
]
