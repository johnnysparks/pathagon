# Pathagon rules

The standard game uses a 7×7 board and 14 pieces per player. Light tries to
connect the near and far edges; dark tries to connect the left and right edges.
Connections are orthogonal—diagonal contact does not join a path.

Players alternate turns. While a player has pieces in reserve, a turn places
one piece on an available square. After that player's reserve is empty, a turn
relocates one of their pieces to an available square. A player may not relocate
the same piece on consecutive turns.

After placing or relocating, every orthogonally adjacent opposing piece trapped
in an A–B–A line between the moved piece and another friendly piece is captured
and returned to its owner's reserve. Squares emptied by that capture are
unavailable to the next player for one turn.

A player wins immediately when their pieces form a continuous orthogonal path
between their goal edges. Supported records also carry a maximum-ply rule and
repetition handling so engines terminate pathological games consistently.

The executable authority is [`../pathagon/engine-rs/src/lib.rs`](../pathagon/engine-rs/src/lib.rs),
with shared rule fixtures in [`../data/fixtures/`](../data/fixtures/).
