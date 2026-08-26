# Design QA: Recent-game board thumbnails

## Source visual truth

- `/tmp/codex-remote-attachments/01a03c96-1e77-7503-a5fc-c48b66cfc455/5156DEBD-4D88-420D-A0BF-839C82378B4B/1-Photo-1.jpg` — 961 × 1281 px. Defines the existing final-board pixel treatment in game playback.
- `/tmp/codex-remote-attachments/01a03c96-1e77-7503-a5fc-c48b66cfc455/5156DEBD-4D88-420D-A0BF-839C82378B4B/2-Photo-2.jpg` — 2880 × 3840 px. Defines the recent-game table context and the requested leading-image placement.

## Implementation evidence

- `/tmp/pathagon-lab-live-games-desktop.png` — full Lab page, 1200 × 8728 px, CSS viewport 1200 × 773 px, devicePixelRatio 1.
- `/tmp/pathagon-lab-live-games-desktop-focus.png` — recent-game table, 1102 × 229 px, same CSS viewport and density.
- `/tmp/pathagon-lab-mobile-full.jpg` — full Lab page, CSS viewport 390 × 844 px, devicePixelRatio 1.
- `/tmp/pathagon-lab-live-games-mobile.png` — recent-game table, 326 × 258 px, CSS viewport 390 × 844 px, devicePixelRatio 1.

## State and comparison

- Theme: dark.
- State: four local cross-play records supplied final board states so the new thumbnails could be inspected; those temporary records were deleted after capture. The remote archive was not modified.
- Full-view evidence confirms the new rows sit within the existing archive panel without changing the surrounding Lab hierarchy. The source is a camera photo rather than a same-viewport browser capture, so the focused table comparison is the fidelity gate.
- Focused comparison: each row now leads with a compact 7×7 final-board image using the same pixel colors and winning-path highlight as the playback badge. Desktop thumbnails measure 44 × 44 px; mobile thumbnails measure 42 × 42 px. The result text remains in the row, so the thumbnail intentionally omits duplicate `Final` and winner labels.
- Resolution pass: the row thumbnail is backed by a 256 × 256 canvas and downscaled to the measured display sizes above, preserving the compact layout while improving raster sharpness.

## Required fidelity surfaces

- Fonts and typography: existing Lab typography and row text are unchanged; model names and result metadata retain their existing truncation and weight.
- Spacing and layout rhythm: the row grows to accommodate the 44 px desktop thumbnail and uses an explicit two-line mobile grid; no horizontal overflow was detected at 1200 px or 390 px.
- Colors and visual tokens: the thumbnail reuses the existing dark archive surface, green border, board cells, light/dark pieces, and winning-path accent.
- Image quality and asset fidelity: the thumbnail uses a 256 × 256 canvas with real final board data from the archive summary; the compact display remains pixel-crisp and no placeholder asset was introduced.
- Copy and content: row labels and replay affordance are unchanged; the board is decorative and the button retains its accessible replay label.

## Findings

No actionable P0, P1, or P2 findings. The requested visual change is present in both responsive states, and the replay interaction remains intact.

## Primary interactions tested

- Loaded `/lab` in Chrome at the default 1200 px viewport.
- Confirmed four `.live-game-thumbnail` elements render in the recent-game list.
- Clicked a recent-game row and confirmed the playback dialog opens.
- Jumped to the final position and confirmed the original 7×7 final-state badge still renders with 49 pixels.
- Rechecked the layout at 390 × 844 px.
- Checked browser console errors: none.

## Comparison history

Initial implementation pass had no actionable P0/P1/P2 findings. The only responsive adjustment was the explicit mobile grid placement for the thumbnail, game number, players, and result line; the post-adjustment desktop and mobile captures above show the corrected layout.

## Implementation checklist

- [x] Include final board and winning path in the cross-play summary response.
- [x] Render the board thumbnail at the leading edge of each recent-game row.
- [x] Preserve replay row behavior and final playback badge.
- [x] Verify desktop and mobile layout, overflow, and console health.
- [x] Back the row thumbnail with a 256 × 256 raster surface without changing its responsive display size.

final result: passed
