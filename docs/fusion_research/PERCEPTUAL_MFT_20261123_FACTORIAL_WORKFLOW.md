# Complete component study: execution and evidence boundaries

The fixed primary remains IR111 epoch6/raw. Current complete Dev HOTA is51.329
versus original RGB47.579 and the strongest same-structure control49.132.
The >=34.529 full-Test target and2–3original-contribution goal remain unproven.

| S C R | Configuration | Evidence source |
| --- | --- | --- |
| 000 | Trained static local residual | New fit, queued |
| 001 | Confidence placement only | New fit, queued |
| 010 | Semantic centering only | New fit, queued |
| 011 | Centering + confidence placement | Completed no-structure curve |
| 100 | Structural matching only | New fit, queued |
| 101 | Structure + confidence placement | Completed no-centering curve |
| 110 | Structure + centering | Completed no-risk curve |
| 111 | All three | Completed fixed no-cycle IR curve |

S jointly includes structural weights, validity and matching-cost confidence.
C changes the semantic reference from RGB to the quarter-FPN-shift null.
R removes only the teacher factor `(1-p)*(p<0.6)` when disabled. All conditions
retain the feature/logit bounds, RGB dense geometry, supervision and both
regularizers. 000 is trained with the same145600parameters and losses;
it is not an untrained identity adapter or a new RGB baseline.

The four missing fits use the same1152cached Trainframes,24cachepasses /
6912updates,seed0 and6/12/18/24snapshots. Existing full/pair fits are not
repeated. They do not use the separately prepared14897-frame coverage schedule.

The registered execution order is:

1. Finish every source producer of the current14-case no-cycle Test on GPU4/5.
2. Run the four missing component fits on GPU4/5, with two-step engineering
   checks, identical complete sample digests and the existing64GiB work cap.
3. Run each Dev video through17zero-adapter streams, then37trained prefix
   streams, then all37full streams. Verify exact parentIR raw/priority anchors,
   missing-input fallback, original dense-box provenance and every prefix.
4. Compute all32new official Dev results, then read the family together.
   Reuse32IR component rows and17RGB/gray/baseline rows:81reported rows,
   not81new runs. Retain paired6/raw and equal4checkpoint/bothpolicy curves.
5. Use GPU4 after its shorter Dev producer exits to profile six methods under
   live-file and RAM-input conditions while GPU5 finishes the longer video.
   Every timed method/frame is checked against its immutable32-frame prefix.
6. Freeze paired6/raw and Dev-best roles for every new formal component,
   including negative ones. Deduplicate identical roles and evaluate all7Test
   sequences /15985frames. No Test criterion can skip a formal component.
7. Reuse parent full/pair Test metrics only after exact whole-video primary
   anchors. Keep new freshRGB and missingIR cases separate, requiring exact
   tracks, all39headline metrics and officialCSV fallback equality.
8. Export all original metrics, per-sequence results and the complete component
   table. A paired HOTA interaction decomposition explicitly avoids adding the
   three conditional removal deltas.

The source-only numerical protocol is inherited unchanged: matched2px/.005
bounds, matched0.6/0.7memberships fixed, at most one weak unmatched score<0.6
in the final4frames. It does not imply canonical/fresh tracking equivalence.
Both are kept explicit. The primary is never replaced by a Test winner.

All new pipeline processes are waiting; the new four fits,37-stream Dev checks,
runtime checks and component Test have not executed yet. Python compilation,
three CLI checks and180maintained CPU tests passed. Tests cover old-component
operator equivalence, missing-view fallback, complete interaction accounting,
and retention of negative formal candidates. These are engineering evidence,
not future HOTA results or proof that the whole queued pipeline has run.

Original taskbook requirements for distinct fusion architectures, prior-art
comparison and full Test of every formal candidate remain separate. These
eight component configurations are one architecture family, not eight novel
architectures. Primitive self-similarity, confidence placement and subtraction
alone do not establish originality. Historical Test exposure and original RGB
pretraining on present Dev remain limitations. The goal stays active.
