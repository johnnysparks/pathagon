# GNN checkpoints

`pathagon-warmstart.pt` is the first replay-warmed GNN checkpoint. It uses
the 100-game Rust archive `rust-selfplay-100-20260823`, 3,719 replay positions,
an 8-layer residual message-passing encoder, and a dynamic placement /
relocation policy head.

It is an initialization point for PUCT self-play, not a promoted game agent.
The checkpoint can be loaded on a 5x5 graph even though it was warmed on 7x7;
that demonstrates architectural transfer, not playing-strength transfer.

`pathagon-cnn-7x7-warmstart.pt` is the first fixed-size CNN comparison
checkpoint. It was trained on the 1,000-game 7x7 neural archive
(`pathagon-rust-7x7-generation-2.jsonl`), covering 88,715 replay positions,
with 200 optimizer updates and symmetry augmentation enabled. Its initial
20-game random smoke evaluation at four PUCT simulations scored 2 wins, 3
losses, and 15 draws; this is a baseline artifact, not a promotion result.

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
- `pathagon-generation-6-5x5.pt`: a 5x5 curriculum fine-tune from generation
  5, trained from 25 games / 687 positions. Its replay file contains 25 path
  wins with no capped draws.
- `pathagon-generation-7-5x5.pt`: a second 5x5 curriculum generation with 25
  games / 926 positions, all path wins. In a fresh 20-game random arena it
  scored 13 wins, 7 losses, and 0 draws at 16 simulations per move.
- `pathagon-generation-8-7x7.pt`: a 7x7 fine-tune from the 5x5 curriculum,
  trained from 10 games / 1,883 positions. The replay contains one path win
  and nine move-cap draws; its fresh 10-game arena scored 4 wins, 3 losses,
  and 3 draws at eight simulations per move.
- `pathagon-generation-9-5x5-r8.pt`: a 5x5 curriculum run with eight pieces
  per side instead of the usual ten. It trained from 50 games / 1,835
  positions, with 48 path wins and two move-cap draws. A fresh 20-game arena
  scored 9 wins, 4 losses, and 7 draws.

`learning.gnn.evaluate` provides a seeded, color-balanced smoke arena against
the random baseline. Generation 4 scored 0 wins, 1 loss, and 4 draws in five
games at four simulations per move. This is a diagnostic, not a strength
claim.

Generation 5 scored 0 wins, 0 losses, and 10 draws in a fresh 10-game arena
at eight simulations per move; all ten reached the 196-ply cap.

Generation 7 transferred zero-shot to 7x7 at 2 wins, 1 loss, and 7 draws in
10 games. Generation 8 is the corresponding 7x7 fine-tune and improved that
smoke arena to 4 wins, 3 losses, and 3 draws. These are small diagnostics, not
promotion gates.

New replay files record `boardSize` and `reservePerPlayer` so variable-size
curriculum games can be loaded without relying on the filename. Generation 6
predates those fields and remains loadable when passed `BoardConfig(5, 10)`.

Games that reach the current move cap remain draws and should not be counted
as wins or losses.
