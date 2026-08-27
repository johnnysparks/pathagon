# Q/Advantage Lambda handoff

Completed 2026-08-26/27 in `us-east-2`.

## Runtime

- Cross-play worker: `pathagon-qadv-crossplay-20260826`, version `1`
- Runtime: Rust `provided.al2023`, `arm64`, 2 GB memory, 900-second timeout
- QAdv model: clean campaign checkpoint exported to ONNX
- Model hash: `sha256:7aff538ca2ddb08eb7b3541994905d0783202cdd630c179b8916575d3a0cf798`
- Recipe: 128 simulations normally, 512 on tactical positions, randomized openings and temperatures, Pathfinder-biased schedule

## Campaign

- 2,500 requested; 2,500 completed; 2,500 unique; 0 failed; 0 exact duplicates
- Seeds: `2026220000`–`2026222499`
- Schedule: Pathfinder 1,000; Surveyor 500; Lunatic 500; Coin Flip 500
- Positions: 66,474 total; 32,861 Q-labeled positions (49.43%, expected because only QAdv turns carry Q labels)

## Replay integrity

- Rule replay: 2,500/2,500
- Malformed games: 0
- Illegal plies: 0
- Capture mismatches: 0
- State repeats / threefold positions: 0
- Missing or unexpected seeds: 0

## QAdv results

| Opponent | QAdv W–L–D | QAdv win rate | Mean plies | Mean captures |
| --- | ---: | ---: | ---: | ---: |
| Pathfinder | 0–1,000–0 | 0.0% | 18.96 | 1.04 |
| Surveyor | 0–500–0 | 0.0% | 18.69 | 2.49 |
| Lunatic | 0–500–0 | 0.0% | 17.22 | 1.79 |
| Coin Flip | 498–2–0 | 99.6% | 59.12 | 10.17 |

The competitive result is a useful warning: the clean checkpoint supplies a valid action-ranking signal, but its current guided-selection recipe is not competitive with the shallow deterministic opponents. The short losses are the main retraining target; the Coin Flip games confirm that the engine can exploit open tactical play.

The batch review reports mark many games `partial-q-coverage`. That is expected cross-play metadata, not a gameplay defect: opponent turns do not emit Q arrays. Structural audits above are the authoritative anomaly check.

## Live archive

- Production site: https://pathagon-game.sparks-house-6466.chatgpt.site/lab
- New archive batches uploaded: 25 × 100 = 2,500 games
- Live imported cross-play total after upload: 5,683 games

## Key artifacts

- `campaign-manifest.json`
- `final-audit/final-corpus-audit.json`
- `all-games.jsonl`
- `batches/` — resumable 100-game archives
- `review-batches/` — bounded manual-review reports
