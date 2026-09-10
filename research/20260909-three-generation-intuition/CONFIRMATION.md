# Independent confirmation

Completed 2026-09-10. Generation two was the sole finalist selected by the registered two-anchor screening rule.

| Anchor | Games | W–L–D | G2 points | Paired bootstrap 95% |
|---|---:|---:|---:|---:|
| Starting model G0 | 400 | 198–183–19 | 51.875% | 47.875%–55.875% |
| Same search, neural root ordering off | 400 | 203–178–19 | 53.125% | 48.750%–57.625% |

The registered gate required at least 53% points and a paired-bootstrap 95% lower bound above 50% against **each** anchor. Both intervals include 50%; the G0 result also falls below the point-rate threshold. No promotion occurred. The prior default remains supported.

Each arena used 200 new opening seeds, with two games per seed and swapped candidate colors. Seeds are disjoint from training, screening, and the other confirmation arena. Intervals resample opening pairs (10,000 bootstrap replicates). Screening games are not pooled with these results. Repeated trajectories and the limited opponent set still constrain generalization.

Native replay audits passed all 800 confirmation games and 47,016 moves. The full campaign covers 1,740 audited games and 97,567 moves. Representative outcomes are described in `REPLAY-REVIEW.md`. Frozen protocol and lineage evidence is in `verification.json`; detailed confirmation statistics and disposition are in `confirmation-results.json`.

This result does not establish reliable improvement; it also does not prove an exactly zero effect. The three-generation experiment is complete. Additional training and deployment-latency work were not performed after the failed strength gate.
