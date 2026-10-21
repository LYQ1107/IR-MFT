# Causal auxiliary modality audit

The strict fallback is a causal RGB-derived auxiliary representation, not a physical thermal sensor. It computes the current luminance difference from a prefix EMA background and updates the background only after emitting the current output.

Overall status: **PASS**.

| Split | Sequence | Frames sampled | Prefix invariant | Reset invariant | Status |
|---|---|---:|---|---|---|
| train | BT-001 | 8 | True | True | PASS |
| train | BT-003 | 8 | True | True | PASS |
| train | MSK-002 | 8 | True | True | PASS |
| train | PF-001 | 8 | True | True | PASS |
| train | SN-001 | 8 | True | True | PASS |
| train | SN-015 | 8 | True | True | PASS |
| train | BT-005 | 8 | True | True | PASS |
| train | SN-013 | 8 | True | True | PASS |
| test | BT-002 | 8 | True | True | PASS |
| test | BT-004 | 8 | True | True | PASS |
| test | MSK-003 | 8 | True | True | PASS |
| test | PF-002 | 8 | True | True | PASS |
| test | SN-009 | 8 | True | True | PASS |
| test | SN-011 | 8 | True | True | PASS |
| test | SN-014 | 8 | True | True | PASS |

The archived historical `infrared` folders remain a `legacy` intervention. They are not silently relabelled as independent TIR measurements; the strict online candidate uses this fallback or a separately verified causal cache.
