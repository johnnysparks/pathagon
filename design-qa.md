# Beam-width control design QA

## Comparison target

- Source visual truth: `/Users/johnny/.codex/attachments/28e9faff-ea51-4619-8e7d-81c2dea3a49f/image-1.png`
- Source pixels: 1858 × 1442 (displayed by the image viewer at a resized 1792 × 1390 preview)
- Implementation screenshot: `/Users/johnny/.codex/attachments/f6c64608-f983-4dfd-9686-b61fd21386e6/Screenshot 2026-08-31 at 5.18.17 PM.png` (user-provided capture of the currently visible build)
- Implementation URL: local production build served at `http://127.0.0.1:4173/` with a temporary local compatibility-date override
- Viewport: unavailable; browser surface was not connected
- CSS size and density normalization: unavailable; no implementation capture was produced
- Intended state: Pathagon game with a Pathfinder opponent selected and the Pathfinder control card visible

## Evidence

The local implementation build passed and its server-rendered HTML contains the new `Pathfinder control`, `Search coverage`, `wide beam`, and accessible `Pathfinder beam width` control markers. The user-provided capture shows the currently visible remote build stopping after the look-ahead control, before the beam-width subsection; `origin/master` at `02fc2c8` likewise has no beam-width UI. The connected in-app browser reported no available browser surfaces, so the local implementation could not be opened at a matched viewport.

The focused-region comparison confirms that the visible build is stale; no comparison of the local implementation at a matched viewport was possible.

## Findings

- [P2] Visible build predates the beam-width change.
  Location: Pathfinder control card.
  Evidence: the user-provided implementation capture has the look-ahead slider followed immediately by max-position controls; the beam subsection exists only in the uncommitted local source.
  Impact: the user cannot configure beam width in the build shown by the capture.
  Fix: commit and publish the local beam-width change, then recapture the Pathfinder card.
- [P2] Screenshot-level comparison of the local implementation remains blocked.
  Location: Pathfinder control card.
  Evidence: no browser-rendered capture of the local source could be produced because no browser surface was connected.
  Impact: local typography, spacing, responsive wrapping, active-slider appearance, and the interaction state cannot be verified visually.
  Fix: reconnect an in-app browser and capture the Pathfinder card at the source viewport, then compare the initial and changed beam-width states.

## Comparison history

One P2 evidence pass confirmed that the visible build is stale; no local implementation comparison was completed because the browser surface is unavailable.

## Fidelity surface checklist

- Fonts and typography: not visually verified; source uses a serif display treatment with compact uppercase labels.
- Spacing and layout rhythm: not visually verified; the new control reuses the existing look-ahead slider/card styles.
- Colors and visual tokens: not visually verified; the new control reuses existing green Pathfinder control tokens.
- Image quality and asset fidelity: no new image assets were introduced.
- Copy and content: automated SSR smoke check found the new beam-width labels and explanatory copy.
- Icons: no new icons were introduced.
- Accessibility: the control has a semantic range input, accessible label, value text, and described scale/help text; keyboard interaction was not browser-tested.

## Automated checks

- Web CI: passed; build and 43 tests passed.
- HTTP smoke: passed; local production build returned HTTP 200 and rendered the new control markers.

final result: blocked
