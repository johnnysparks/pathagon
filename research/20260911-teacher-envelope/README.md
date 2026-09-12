# Teacher quality at the quiet-regret deployment envelope

Status: completed — diagnostic only; no promotion

## Idea

Before another learner, measure whether the depth-4/8k quiet-regret teacher is
aligned with the fixed search budgets used in the arena.

## Outcome

On all 192 audited roots, the teacher's selected action matched tactical-filter
search 46.4% at 2k, 38.5% at 8k, and 50.0% at 32k. Movement roots were the
largest mismatch: 11.1%, 4.4%, and 8.9%. The diagnostic took 576 native
searches and wrote a per-root report under ignored `workspace/`.

## Project impact

The training target was not a reliable proxy for the deployment decision,
especially at 8k. Future training must label at the intended search envelope or
use a budget-conditioned target; offline agreement with a mismatched teacher is
not evidence of intuition.
