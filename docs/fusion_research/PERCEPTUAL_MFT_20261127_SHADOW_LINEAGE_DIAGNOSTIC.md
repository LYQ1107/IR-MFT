# Continuous shadow lineage: Train/Dev-only diagnosis

The GT-free producer processes all6Train27,251 and2Dev4,834 frames before a separate worker reads endpoint labels. A candidate new trusted identity must share the same current proposal with a shadow that previously agreed with a different trusted identity for at least3 consecutive frames; that old identity must be absent now, and the current shadow's measured streak must span the entire gap plus3. No identity is merged, no output tracker is created, and no HOTA/model/threshold selection is performed.

Only **2 Train events** and **1 Dev event** meet this fixed rule. The early-Train short-gap event (6–25 frames) has matching known GT endpoints; its long-gap event (26–250) has **different** known GT endpoints. Dev has one short-gap event with matching known endpoints. The final25percent Train validation window has **no event**. All labels retain the original Hungarian/IoU/ignored-region rules, and unknown is not background.

This small inventory does not support building a long-gap automatic identity merger or predict a meaningful HOTA gain. In particular, the only long-gap Train candidate is wrong under endpoint identity labels. The conclusion is local to this registered continuous-shadow rule, not a proof that all reidentification strategies are unhelpful.

Both actual workers exit0. Aggregate/event-hash/source checks are recorded; this publication check is not a second temporal-event or GT-label producer. Private event rows and GT identities are excluded from Git; aggregate counts, execution/source hashes and logs are retained. This diagnostic creates no new Test result and reads no Test labels/metrics. Historical Test exposure remains disclosed.

The previous formal selected PP method remains32.4955880280 HOTA, +2.9666173406 over the original RGB-only YOLOX-X+ByteTrack29.5289706874, below the+5 minimum34.5289706874. Fixed November DiffV2IR/ProtocolV2, seed0, no new neural training/images/descriptors/weights/GPU work. Goal remains active and no new independently attributable multimodal innovation is claimed.
