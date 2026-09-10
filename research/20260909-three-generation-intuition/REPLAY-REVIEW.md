# Representative replay review

## Generation one versus the starting model

All 100 games passed the independent native replay audit (5,425 plies). For
human-readable scrutiny, `src/bin/inspect.rs` replays the same native transitions
and prints the first relocation and final six moves with boards and winning paths.
It adds no game-rule implementation or search changes. Full one-time snapshots
are in ignored `workspace/g1-vs-g0.representative.txt`.

Selection was the median-length candidate win and loss in each color, plus each
of the two non-path terminations. Indices are zero-based within the arena JSONL.

| Case | Index / seed | Observed ending |
|---|---|---|
| Light win | 60 / 49091030 | Light captures on placements 43 and 45, exhausts its reserve, then wins with its first relocation, 4→19 at ply 47. Native winning path connects the top and bottom. |
| Light loss | 24 / 49091012 | Dark relocations at plies 54 and 56 capture Light stones; Dark then completes a left-to-right path at ply 58. |
| Dark win | 29 / 49091014 | Dark relocations at plies 56, 58, and 60 capture four Light stones in total. The final relocation both captures and completes a left-to-right path. |
| Dark loss | 99 / 49091049 | Starting-model Light uses successive relocating captures at plies 35, 37, and 39; the last creates its top-to-bottom path. |
| Repetition draw | 12 / 49091006 | Placements recapture stones around squares 9, 10, 16, and 17, returning captured stones to reserves and repeating the state. The native audit confirms threefold termination. |
| Ply-cap draw | 95 / 49091047 | Legal relocation play continues through ply 196 without either winning path. The native audit confirms the normal cap, not early truncation. |

The sample confirms meaningful play across placement, returned reserves, and
relocation, with both decisive path tactics and legitimate draws. Similar capture
patterns appear in candidate wins and losses. These snapshots do **not** establish
that the network invented a strategy or explain the causal mechanism behind its
61% screening score. They are a qualitative check alongside the full mechanical
audit. Any selected finalist must also be reviewed on independent confirmation
replays before promotion.

## Independent G2 confirmation versus G0

All 400 games passed the native audit: 23,622 plies, including 6,631 relocation
moves and 6,213 captures. Selection again used the median-length candidate win
and loss in each color, plus the median-length example of each draw reason.
Snapshots are in ignored `workspace/confirmation-g2-vs-g0.representative.txt`.

| Case | Index / seed | Observed ending |
|---|---|---|
| Light win | 154 / 59091077 | Light relocates 3→44 to capture square 37 at ply 47, then relocates 13→37 at ply 49 to complete its top-to-bottom path. |
| Light loss | 232 / 59091116 | G0 Dark relocates 0→24, capturing Light at 25, then relocates 3→25 to connect left to right at ply 46. |
| Dark win | 67 / 59091033 | G2 Dark relocates 26→47 to capture square 40, then fills 40 from 29 at ply 48, completing a long left-to-right path. |
| Dark loss | 277 / 59091138 | G0 Light relocates 3→24 and captures square 31, then relocates 5→31 to complete its path at ply 45. |
| Repetition draw | 397 / 59091198 | The 110-ply game includes relocation and recapture with returned reserves; the full native audit verifies the repeated state and correct threefold termination. |
| Ply-cap draw | 166 / 59091083 | Play remains legal through ply 196. A late capture, replacement, and recapture return a stone to Light's reserve without producing a winning path. |

The decisive samples again show capture followed by occupation of the cleared
square as a path-completion mechanism for both the candidate and G0. They
support the legality and phase-coverage checks but do not reveal a qualitative
new strategy or explain a reliable advantage. The independent aggregate is
51.875% points with an interval spanning 50%; it fails the registered strength
gate. No promotion or conditional deployment-latency work follows that failure.

## Independent G2 confirmation versus NN-off search

All 400 games passed the native audit: 23,394 plies, including 6,388 relocation
moves and 6,224 captures. Supplementary review selected the median-length G2
Dark win, G2 Light loss, repetition draw, and cap draw. Snapshots are in ignored
`workspace/confirmation-g2-vs-unguided.representative.txt`.

| Case | Index / seed | Observed ending |
|---|---|---|
| Dark win | 199 / 59092099 | G2 relocates 0→25 to capture square 32, then relocates 22→32 to connect its left-to-right path at ply 46. |
| Light loss | 186 / 59092093 | NN-off Dark relocates 9→24 to capture square 31, then relocates 13→31 to complete its path at ply 50. |
| Repetition draw | 173 / 59092086 | A placement/recapture cycle around squares 9, 10, 16, and 17 ends at ply 85; the full native audit confirms threefold repetition. |
| Ply-cap draw | 252 / 59092126 | Late relocating captures return stones to Light's reserve. Neither side has a winning path at the normal 196-ply cap. |

The sample again contains valid play in both phases and the same capture-then-
connect motif for both sides. No replay inconsistency was found. The aggregate
53.125% score has a paired interval spanning 50%, so these examples do not turn
the failed confirmation into evidence of a reliable strength advantage.
