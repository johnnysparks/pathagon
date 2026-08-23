# GNN checkpoints

`pathagon-warmstart.pt` is the first replay-warmed GNN checkpoint. It uses
the 100-game Rust archive `rust-selfplay-100-20260823`, 3,719 replay positions,
an 8-layer residual message-passing encoder, and a dynamic placement /
relocation policy head.

It is an initialization point for PUCT self-play, not a promoted game agent.
The checkpoint can be loaded on a 5x5 graph even though it was warmed on 7x7;
that demonstrates architectural transfer, not playing-strength transfer.

The local progression includes:

- `pathagon-generation-1.pt`: trained from 20 neural games / 1,298 positions.
- `pathagon-generation-2.pt`: trained from 10 neural games / 1,352 positions;
  its JSONL is replay-valid and includes six move-cap draws.
- `pathagon-generation-3.pt`: trained from five neural games / 548 positions.
  `selfplay-generation-3.jsonl` contains three wins and two move-cap draws.
- `pathagon-generation-4.pt`: trained from five neural games / 980 positions.
  `selfplay-generation-4.jsonl` is replay-valid; all five games reached the
  196-ply cap, which is useful as a draw diagnostic but weak as a learning
  signal.
- `pathagon-generation-5.pt`: trained from 10 neural games / 1,622 positions.
  `selfplay-generation-5.jsonl` contains three path wins and seven move-cap
  draws.

`learning.gnn.evaluate` provides a seeded, color-balanced smoke arena against
the random baseline. Generation 4 scored 0 wins, 1 loss, and 4 draws in five
games at four simulations per move. This is a diagnostic, not a strength
claim.

Generation 5 scored 0 wins, 0 losses, and 10 draws in a fresh 10-game arena
at eight simulations per move; all ten reached the 196-ply cap.

Games that reach the current move cap remain draws and should not be counted
as wins or losses.
