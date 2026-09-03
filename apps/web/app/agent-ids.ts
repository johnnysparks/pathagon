/** Stable public IDs for promoted Pathfinder opponents. */
export const PATHFINDER_TACTICAL_FILTER_ID = "pathfinder-v0.4.0-tactical-filter";
export const TRAINED_PATHFINDER_ID = "pathfinder-v0.5.0-trained-evaluator";
export const TRANSITION_PATHFINDER_ID = "pathfinder-action-transition-v4-xent";

/** Canonical IDs for the six player-facing opponent cards. */
export const PATHMAN_ID = "pathman-v1.0.0";
export const TILE_DRIVER_ID = "tile-driver-gnn-v1.0.0";
export const SEER_ID = "seer-cnn-puct-v1.0.0";
export const DOUBLE_DRAGON_ID = "double-dragon-qadv-v1.0.0";
export const YANN_TILESON_ID = "yann-tileson-jepa-v1.0.0";
export const RANDO_RACCON_ID = "rando-raccon-seeded-v1.0.0";

export const LEGACY_OPPONENT_ALIASES: Readonly<Record<string, string>> = {
  [PATHFINDER_TACTICAL_FILTER_ID]: PATHMAN_ID,
  [TRAINED_PATHFINDER_ID]: PATHMAN_ID,
  [TRANSITION_PATHFINDER_ID]: PATHMAN_ID,
  "surveyor-v0.2.0": PATHMAN_ID,
  "lunatic-v0.1.0": PATHMAN_ID,
  "coin-flip-v0.0.1": RANDO_RACCON_ID,
  "coin-flip-v0": RANDO_RACCON_ID,
  "cnn-puct-v0": SEER_ID,
};
