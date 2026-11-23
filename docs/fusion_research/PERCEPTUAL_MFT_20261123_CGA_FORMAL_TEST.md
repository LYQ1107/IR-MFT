# Frozen seed0 CGA: all7 Test sequences

Full7 historical Test sequences,15985 frames, official combined TrackEval. All14 cases were frozen from complete Dev. No Test-based checkpoint/readout selection. Historic Test was previously exposed; this remains exploratory research.

| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CGA_rgb_Dev_selected | 29.812 | 45.678 | 19.535 | 33.470 | 50.462 | 484 |
| CGA_gray_Dev_selected | 28.997 | 46.058 | 18.338 | 32.476 | 50.314 | 480 |
| CGA_ir_Dev_selected | 29.431 | 46.052 | 18.898 | 32.394 | 51.136 | 476 |
| CGA_IR_full_fused_readout | 28.929 | 45.866 | 18.315 | 32.747 | 50.384 | 496 |
| CGA_IR_aligned_paired24 | 30.314 | 46.407 | 19.885 | 33.276 | 51.102 | 463 |
| CGA_IR_aligned_Dev_best4 | 29.906 | 45.802 | 19.609 | 32.937 | 50.953 | 437 |
| CGA_IR_no_gate_paired24 | 30.318 | 47.479 | 19.448 | 33.039 | 51.375 | 508 |
| CGA_IR_no_gate_Dev_best4 | 29.157 | 46.351 | 18.431 | 32.182 | 51.182 | 479 |
| CGA_IR_no_counterfactual_paired24 | 28.677 | 45.433 | 18.191 | 30.666 | 50.061 | 500 |
| CGA_IR_no_counterfactual_Dev_best4 | 29.388 | 45.728 | 18.975 | 32.655 | 51.053 | 446 |
| CGA_IR_unavailable | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| CGA_IR_spatial_roll | 29.867 | 46.176 | 19.406 | 32.709 | 50.942 | 459 |
| CGA_source_RGB_fresh | 29.529 | 46.187 | 18.968 | 33.072 | 50.897 | 450 |
| CGA_source_RGB_canonical | 29.529 | 46.187 | 18.968 | 33.072 | 50.898 | 450 |

Frozen primary versus canonical RGB: -0.098 HOTA; versus fresh RGB: -0.098; versus strongest matched RGB/gray: -0.381.

| Removal | Full − paired24 HOTA | Full − Dev-selected best4 HOTA |
| --- | ---: | ---: |
| aligned | -0.883 | -0.475 |
| no_gate | -0.887 | +0.274 |
| no_counterfactual | +0.754 | +0.043 |

Positive seed0 Dev removals do not establish original innovations or Test gains. no_counterfactual removes both reliableRGB and wrong-aux losses; no_gate removes feature risk/bound and priority, although raw readout excludes priority.

Cached ByteTrack replay FPS excludes detector/encoder/adapter preparation. Shared pixel production time is not per-method end-to-end FPS.

The original exact-count source audit failed on one weak NMS candidate at the MSK-003 tail. Failure retained and tolerance amendment recorded before any14Test accuracy. Matched numeric bounds and high/birth memberships were preserved. Fresh and canonical RGB are separate controls; low-confidence tracking equivalence is not assumed.

Reaching a numeric threshold alone does not establish2–3 original multimodal contributions. Module support, strongest controls and prior-art boundaries require explicit interpretation.
