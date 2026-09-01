# Beam-width control design QA

## Comparison target

- Source visual truth: `/Users/johnny/.codex/attachments/28e9faff-ea51-4619-8e7d-81c2dea3a49f/image-1.png`
- Source pixels: 1858 × 1442 (displayed by the image viewer at a resized 1792 × 1390 preview)
- Implementation screenshot: `/Users/johnny/.codex/attachments/f6c64608-f983-4dfd-9686-b61fd21386e6/Screenshot 2026-08-31 at 5.18.17 PM.png` (user-provided capture of the currently visible build)
- Implementation URL: post-ship production Worker at `https://pathagon-game.johnnymsparks.workers.dev/`
- Viewport: unavailable; browser surface was not connected
- CSS size and density normalization: unavailable; no implementation capture was produced
- Intended state: Pathagon game with a Pathfinder opponent selected and the Pathfinder control card visible

## Evidence

The user-provided capture shows the pre-ship remote build stopping after the look-ahead control, before the beam-width subsection; `origin/master` at `02fc2c8` likewise had no beam-width UI. Commit `ff71f57` shipped successfully through GitHub Actions run `33454470617`, and the live production response now returns HTTP 200 with `Pathfinder control`, `Beam width`, `wide beam`, and accessible `Pathfinder beam width` markers. The connected in-app browser reported no available browser surfaces, so the live implementation could not be opened at a matched viewport.

The focused-region comparison confirms the supplied capture was stale. The live HTML confirms the beam subsection is now present, but no matched-viewport screenshot comparison was possible.

## Findings

- [P2] Supplied capture showed a pre-ship build. Resolved by shipping `ff71f57`.
  Location: Pathfinder control card.
  Evidence: the user-provided implementation capture has the look-ahead slider followed immediately by max-position controls; the subsequent production response contains the beam-width markers.
  Impact: the supplied image was not evidence of the intended release state.
  Fix: none for the shipped state; refresh the production app if an older cached document is still visible.
- [P2] The beam control needed a more direct visible label.
  Location: Pathfinder control card, beam subsection.
  Evidence: the first local implementation called the subsection `Search coverage`, while the user asked for a beam-width knob.
  Impact: a user scanning for “beam width” could miss the setting.
  Fix: the shipped polish change labels it `Beam width` while retaining the explanatory coverage copy.
- [P2] Screenshot-level comparison of the local implementation remains blocked.
  Location: Pathfinder control card.
  Evidence: no browser-rendered capture of the post-ship production source could be produced because no browser surface was connected.
  Impact: local typography, spacing, responsive wrapping, active-slider appearance, and the interaction state cannot be verified visually.
  Fix: reconnect an in-app browser and capture the Pathfinder card at the source viewport, then compare the initial and changed beam-width states.

## Comparison history

1. The supplied screenshot showed the pre-ship build and exposed the release gap.
2. `ff71f57` was committed, pushed, verified, and deployed successfully.
3. The visible label was tightened from `Search coverage` to `Beam width` for discoverability.
4. Live HTML confirms the shipped control exists; a visual screenshot comparison remains unavailable because the browser surface is not connected.

## Fidelity surface checklist

- Fonts and typography: not visually verified; source uses a serif display treatment with compact uppercase labels.
- Spacing and layout rhythm: not visually verified; the new control reuses the existing look-ahead slider/card styles.
- Colors and visual tokens: not visually verified; the new control reuses existing green Pathfinder control tokens.
- Image quality and asset fidelity: no new image assets were introduced.
- Copy and content: live production HTML contains the explicit `Beam width` label, range semantics, and explanatory copy.
- Icons: no new icons were introduced.
- Accessibility: the control has a semantic range input, accessible label, value text, and described scale/help text; keyboard interaction was not browser-tested.

## Automated checks

- Web CI: passed; build and 43 tests passed.
- HTTP smoke: passed; production returned HTTP 200 and rendered the new control markers.
- Deployment: passed; GitHub Actions run `33454470617` completed verification, migrations, and production Worker deployment.

final result: blocked
