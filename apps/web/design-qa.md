# Model league layout QA

## Source and implementation

- Source visual truth: `/var/folders/bx/l9zszz055x7b9bdwkp3c4p8w0000gn/T/TemporaryItems/NSIRD_screencaptureui_jmhvMX/Screenshot 2026-08-26 at 10.18.23 AM.png` (1994 × 1010 px). This is the before-state; the requested change is to remove the oversized hero, strength-leader card, and summary strip.
- Secondary source evidence: `/var/folders/bx/l9zszz055x7b9bdwkp3c4p8w0000gn/T/TemporaryItems/NSIRD_screencaptureui_DsP96J/Screenshot 2026-08-26 at 10.26.03 AM.png` (200 × 164 px), showing the cramped game-card crop to correct.
- Implementation screenshot: `/tmp/pathagon-layout-implementation-desktop.png` (1400 × 2000 px crop from the browser-rendered desktop capture; the visible browser content is approximately 700 × 1000 CSS px at device scale 2).
- Combined comparison: `/tmp/pathagon-layout-comparison.png` (2725 × 1064 px), with the before-state and implementation in one comparison input.
- Runtime-error evidence: `/var/folders/bx/l9zszz055x7b9bdwkp3c4p8w0000gn/T/TemporaryItems/NSIRD_screencaptureui_qCQHAs/Screenshot 2026-08-26 at 10.46.26 AM.png`, showing the hosted Games panel failing with `boundedInteger is not defined`.
- Latest layout evidence: `/var/folders/bx/l9zszz055x7b9bdwkp3c4p8w0000gn/T/TemporaryItems/NSIRD_screencaptureui_IHzMtK/Screenshot 2026-08-26 at 10.51.45 AM.png` (1142 × 438 px), showing the undersized preview and dead space that prompted this iteration.
- Revised implementation screenshot: `/tmp/pathagon-layout-thumbnail-fixed.jpg` (1280 × 720 px, hosted browser capture; CSS viewport 1280 × 720 at device scale 1).
- State: historical hosted `/lab` capture from before the current research cleanup, dark theme, archive loaded with 940 games and 16 displayed agents, default Elo ladder selected. The current implementation ranks only six Rust-engine opponents; research-only identities remain archive evidence and are not ranked.
- Density normalization: source screenshot retained at its supplied density; implementation was cropped from a 3420 × 2224 px desktop capture and resized for comparison, with browser chrome excluded from the implementation crop.

## Comparison evidence

Full-view comparison shows the implementation removes the redundant “Leaderboard”/“Current strength leader” composition and the four summary cards. The page now leads with a compact model-league header, then presents the requested Game sets section. The default Elo ladder is active and the existing green, serif-led visual language remains consistent.

The focused game-card evidence is addressed structurally: game cards use a fixed thumbnail column, a bounded text column, a two-line matchup treatment, and a dedicated replay footer. The thumbnail now clears the inherited `grid-area` placement and stretches to the full card content height, eliminating the implicit extra row visible in the latest source screenshot. This keeps long model names from colliding with the thumbnail or replay affordance. The existing final-board canvas thumbnail is preserved rather than replaced with a new asset.

## Required fidelity surfaces

- Fonts and typography: the existing serif display face, compact uppercase kicker, and dense sans-serif metadata remain consistent with the surrounding league UI; long model names still truncate within their bounded text column.
- Spacing and layout rhythm: the thumbnail and metadata now share one grid row; the card height is 111px with an 89px board preview inside the padded content area, removing the stray lower row and its dead space.
- Colors and visual tokens: the existing dark green surfaces, muted sage text, green active/replay accents, border opacity, and hover treatment are unchanged.
- Image quality and asset fidelity: the existing pixel-rendered final-board canvas remains the source of the preview; it is larger, aligned, and not replaced with a placeholder or new drawing.
- Copy and content: game number, date, matchup, result, ply count, and Replay affordance remain intact and readable.

## Findings

- No actionable P0, P1, or P2 visual findings remain in the captured state.
- P3 follow-up: a wide desktop capture of the two-column game list and a selected replay would provide additional visual evidence; the layout rules and interaction paths are implemented, but the available browser pane capture was the narrow sets-first state.
- Runtime follow-up resolved: the hosted error was caused by a stale/incomplete pagination-helper artifact. A fresh build was published and the Games panel now loads 24 of 940 games, with more available on scroll.
- Card-layout follow-up resolved: the latest source screenshot showed the thumbnail stranded in a separate grid row. The implementation screenshot shows the preview aligned in the same row as its game metadata, filling the card height without the former dead space.

## Comparison history

1. Before pass: the source showed a large hero, a second leader card, four summary cards, and the archive list below them. The game-card crop showed crowded alignment.
2. Fix: replaced the above-the-fold composition with a compact header; grouped Elo standings and pairwise results into tabbed Game sets; moved paginated games into a dedicated Games column; added pairwise filtering, mobile stack navigation, and retained the replay modal.
3. Post-fix evidence: the browser-rendered implementation shows the compact header, Elo ladder tab, live leader detail inside the selected set, and a readable sets-first mobile state with no clipping or overlap in the visible region.
4. Production runtime repair: rebuilt and redeployed the pagination helper after the hosted screenshot surfaced `boundedInteger is not defined`; browser verification then loaded the newest-first Games panel and its 24 initial cards successfully.
5. Thumbnail sizing repair: removed the inherited named grid area from the game-card thumbnail, made it stretch with the card row, and confirmed in the hosted browser that the first card measures 111px high with an 89px preview inside its padded content area.

## Accessibility and interaction checks

- Tabs use native buttons with `role="tab"` and `aria-selected`.
- Pairwise rows are native buttons and expose selected state with `aria-selected`; only played matchups are actionable.
- Game cards are native buttons with replay labels; focus-visible outlines are defined for tabs, rows, cards, back navigation, and mobile section CTAs.
- Escape, arrow keys, spacebar, and replay controls remain wired to the existing full move replay.
- The local preview D1 schema was initialized for the visual check, removing the `no such table: selfplay_games` error; the hosted preview loaded the real archive.

## Final result

passed
