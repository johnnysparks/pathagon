# Promote-only learned sorter

Status: promising but not promotable — completed on 2026-09-11

## Idea

Full QAdv reordering improved some action metrics but consumed more nodes and
reduced completed search depth. The Rust integration now supports
`--sorter-promote-only`: the learned model may move one action to the front of
the heuristic root order, while every other action keeps native order.

## Outcome

Using the envelope-matched model, the fresh 48-game 32k arena scored 28–18–2
(60.4% points) against control's 26–22–0 (54.2%), a +6.25 point difference.
All six candidate/control logs passed native replay audits. The candidate still
had lower mean completed depth (3.017 vs 3.084) and higher mean nodes per move
(20,266 vs 19,711), so the registered cost gate failed even though the strength
signal was positive across the three seeds.

## Project impact and decision

This is the first meaningful whole-game strength signal from the learned
intuition line. It supports the idea that intuition should be an advisory
tie-breaker layered on top of a rules-authoritative search. The candidate is
not promoted because its efficiency regression violates the registered gate;
the v4 transition-policy default and existing controls remain unchanged. A
follow-up should reduce promotion frequency or preserve depth before any
production consideration.
