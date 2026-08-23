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

Games that reach the current move cap remain draws and should not be counted
as wins or losses.
