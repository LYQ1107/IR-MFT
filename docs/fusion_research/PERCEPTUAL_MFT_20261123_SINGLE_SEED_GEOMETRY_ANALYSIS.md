# Single-seed NMS / emitted geometry diagnosis

Full4834frame Dev, fixed fitted seed0 by explicit user directive. RGB18/gray18/IR24 weights unchanged. Raw2x2 crosses NMS geometry and emitted geometry while holding fused dense scores fixed. Priority integration adds another selection step; do not interpret its proposal sets as fixed.

| Cue | Role | Policy | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rgb | full | priority | 45.770 | 68.860 | 30.461 | 51.951 | 53.556 | 39.902 |
| rgb | full | raw | 49.957 | 68.620 | 36.397 | 58.252 | 59.984 | 42.175 |
| rgb | rgb_boxes | priority | 46.865 | 68.855 | 31.933 | 53.995 | 55.484 | 40.286 |
| rgb | rgb_boxes | raw | 48.945 | 68.884 | 34.812 | 57.526 | 57.771 | 42.240 |
| rgb | rgb_scores | priority | 46.965 | 68.792 | 32.101 | 55.067 | 54.045 | 41.718 |
| rgb | rgb_scores | raw | 45.784 | 68.567 | 30.611 | 53.308 | 51.106 | 41.950 |
| rgb | fused_nms_rgb_boxes | priority | 44.846 | 68.832 | 29.257 | 51.241 | 51.526 | 39.902 |
| rgb | fused_nms_rgb_boxes | raw | 48.927 | 68.850 | 34.803 | 57.344 | 57.726 | 42.249 |
| rgb | rgb_nms_fused_boxes | priority | 48.693 | 68.882 | 34.451 | 55.517 | 59.354 | 40.286 |
| rgb | rgb_nms_fused_boxes | raw | 48.640 | 68.672 | 34.480 | 56.044 | 59.751 | 39.782 |
| gray | full | priority | 47.397 | 68.633 | 32.771 | 53.653 | 55.689 | 41.110 |
| gray | full | raw | 49.965 | 68.097 | 36.707 | 58.417 | 56.749 | 45.026 |
| gray | rgb_boxes | priority | 48.479 | 68.660 | 34.264 | 56.714 | 55.524 | 43.297 |
| gray | rgb_boxes | raw | 48.111 | 68.445 | 33.856 | 55.609 | 57.113 | 41.229 |
| gray | rgb_scores | priority | 47.450 | 68.813 | 32.753 | 55.524 | 55.109 | 41.715 |
| gray | rgb_scores | raw | 50.044 | 68.363 | 36.668 | 59.629 | 57.001 | 44.946 |
| gray | fused_nms_rgb_boxes | priority | 47.307 | 68.702 | 32.614 | 53.641 | 55.527 | 41.096 |
| gray | fused_nms_rgb_boxes | raw | 49.291 | 68.601 | 35.457 | 57.278 | 57.094 | 43.495 |
| gray | rgb_nms_fused_boxes | priority | 48.565 | 68.589 | 34.421 | 56.725 | 55.686 | 43.309 |
| gray | rgb_nms_fused_boxes | raw | 48.554 | 68.215 | 34.597 | 56.596 | 56.761 | 42.372 |
| ir | full | priority | 47.388 | 68.873 | 32.639 | 54.337 | 54.261 | 42.336 |
| ir | full | raw | 49.771 | 68.456 | 36.217 | 59.972 | 56.459 | 44.898 |
| ir | rgb_boxes | priority | 48.916 | 68.735 | 34.842 | 56.580 | 57.374 | 42.519 |
| ir | rgb_boxes | raw | 50.515 | 68.697 | 37.170 | 59.907 | 60.948 | 42.354 |
| ir | rgb_scores | priority | 47.743 | 68.861 | 33.134 | 55.954 | 55.766 | 41.708 |
| ir | rgb_scores | raw | 48.107 | 68.247 | 33.943 | 56.798 | 56.849 | 41.471 |
| ir | fused_nms_rgb_boxes | priority | 47.396 | 68.881 | 32.645 | 54.339 | 54.282 | 42.334 |
| ir | fused_nms_rgb_boxes | raw | 48.979 | 68.242 | 35.188 | 58.543 | 56.079 | 43.796 |
| ir | rgb_nms_fused_boxes | priority | 48.888 | 68.726 | 34.807 | 56.544 | 57.312 | 42.519 |
| ir | rgb_nms_fused_boxes | raw | 49.861 | 68.369 | 36.390 | 58.938 | 59.899 | 42.088 |

Coordinate copying/NMS/classical scoring alone is not an original multimodal mechanism. no_counterfactual removes BOTH reliableRGBdistillation andwrong-auxiliary regularization. no_gate removes feature risk/bound andfinalpriority integration;339gateweights remainallocatedbutunused. Each removal has all4epochs; paired comparisons above hold the readout/epoch24/policy fixed.

No new Test or >=+5 Test HOTA /2–3contribution claim. All intervention metrics, per-sequence values and exact2x2 differences are in the adjacent JSON.
