# GNN checkpoints

`pathagon-warmstart.pt` is the first replay-warmed GNN checkpoint. It uses
the 100-game Rust archive `rust-selfplay-100-20260823`, 3,719 replay positions,
an 8-layer residual message-passing encoder, and a dynamic placement /
relocation policy head.

It is an initialization point for PUCT self-play, not a promoted game agent.
The checkpoint can be loaded on a 5x5 graph even though it was warmed on 7x7;
that demonstrates architectural transfer, not playing-strength transfer.
