# Rust held-out evolution run v1

The first native evaluator tournament used four mutated candidates, identical paired-color training openings, and a disjoint promotion set.

- 32 training games across four candidates
- 16 held-out evaluation games for the training winner
- Training winner: `rust-evo-g1-c0-279-109-820-53-117-66`
- Training result: 5 wins, 3 losses
- Held-out result: 6 wins, 6 losses, 4 draws
- Promotion: rejected

This is the system working correctly. The candidate looked better on the games used for selection but failed to beat the incumbent on unseen openings, so the handcrafted generation-zero evaluator remains champion.

`corpus/training` and `corpus/evaluation` are intentionally separate. `report.json` contains every candidate and decision; `champion.json` is the accepted evaluator only.
