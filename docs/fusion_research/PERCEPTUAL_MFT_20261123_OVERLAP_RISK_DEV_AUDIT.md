# Existing-descriptor overlap-risk diagnosis

Ordinary binary box-overlap risk probes only. Not actual occlusion visibility, tracking HOTA, independent identity evidence or deployed tracker.

All7 ordinary logistic models were fitted on110570 Train foreground crops before any Dev labels. The label is GT-box intersection/min-area>=0.35 with another fish; it is an overlap proxy, not observed visibility.

| Dev sequence | Global source | Reference AUC | Best RGB AUC | Best IR AUC |
| --- | --- | ---: | ---: | ---: |
| BT-005 | rgb | 0.7953 | 0.8117 | 0.7948 |
| BT-005 | new | 0.7990 | 0.8142 | 0.8002 |
| SN-013 | rgb | 0.7747 | 0.7901 | 0.7747 |
| SN-013 | new | 0.7756 | 0.7915 | 0.7739 |

IR does not exceed the strongest RGB control on either video or either source. Do not deploy an IR risk gate from this probe. Existing native encoders were optimized for identity; the result concerns these frozen representations and does not rule out every possible occlusion model. No new tracking/Test claim is made.
