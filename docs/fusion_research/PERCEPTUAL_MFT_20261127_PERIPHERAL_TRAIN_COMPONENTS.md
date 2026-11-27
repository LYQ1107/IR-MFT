# Selected protected method: Train-only component removals

The method was selected from the registered Train/Dev rule before Test. These component cases were fixed before reading any new Test metrics and cannot alter that selection. Existing GT-free source snapshots and the same selected need mode/threshold are reused, with no new neural training, descriptor bank, image or seed.

All27,251 Train frames warm the publisher; only the final6,814 frames are scored. This is development-window evidence; older studies inspected the window and old ReID encoders saw Train identities. It is not an untouched or unseen-domain test.

| Case | HOTA |
|---|---:|
| Original same-input reference | 44.920663037 |
| Full selected method | 46.227954728 |
| Without spatial explanation | 45.993916261 |
| Without identity handoff | 45.698007883 |
| Without both | 45.421800946 |
| Without need gate (unconditional, same spatial/handoff) | 46.105608297 |

The spatial contribution with handoff retained is **+0.234038467 HOTA**; handoff with spatial explanation retained is **+0.529946844**. Their factorial interaction is **-0.042168471**. The selected need gate versus unconditional is **+0.122346431**. These are conditional ablation differences, not additive independent gains.

Every trusted-row geometry/score and injective constant alias was checked by the component worker against the immutable source. Three original detailed276-column CSVs, all47 aggregate/per-sequence metrics, summaries, source hashes and actual worker exit0 are preserved. A separate parser recomputed987 official fields and all factorial arithmetic; its scope does not include a second visual producer or independent full Kalman/LAP replay.

Numeric and mechanism differences do not establish an IR-specific effect. All variants retain the fixed IR visual detector and older parent modules. The primary original-RGB Test7 baseline stays29.5289706874; the minimum Test target stays34.5289706874. This report produces no new Test result and counts no new independently validated multimodal innovation. Goal remains active.
