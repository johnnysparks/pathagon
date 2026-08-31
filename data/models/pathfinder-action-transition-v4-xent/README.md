# Pathfinder action-transition v4 xent

This is the scaled next-generation explicit transition-policy scorer from
`research/20260830-nextgen-scaled`. It uses the same 32-feature
rules-authoritative wrapper as v3, trained on 14,000 source-disjoint roots with
39 selective depth-8 teacher replacements. The xent variant won the held-out
ranking comparison and was evaluated against the supported v0.5 opponent in a
1,000-game cloud arena.

- Artifact: `transition-policy.json`
- SHA-256: `f11d7ddee101ccab35ee162e53c95ced076b1fb10242443ad562dbd51c1085d4`
- Rules namespace: `pathagon-rules-v1`
- Arena: 565 wins, 401 losses, 34 draws; 58.2% point rate
- Color split: 57.5% Light and 58.9% Dark point rate
- Status: user-facing default Pathfinder model; v3 remains available as a prior version

The model is loaded by the default browser Pathfinder opponent through the v4
browser asset. The cloud arena used the private `pathagon-transition-v4-sanity-20260830`
Lambda function in `us-east-2`; no public endpoint is part of this artifact.
