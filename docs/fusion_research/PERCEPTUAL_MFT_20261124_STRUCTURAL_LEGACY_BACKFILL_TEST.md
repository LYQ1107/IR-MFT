# Initial structural formal Test backfill

Initial full-cycle RGB/gray and full-cycle no-centering formal coverage only; not another architecture or evidence of a new multimodal contribution. HistoricalTest is exposed.

| Role | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_matched_RGB | 28.821 | 31.764 | 45.534 | 18.329 | 50.769 | 448 |
| legacy_matched_gray | 29.638 | 33.787 | 46.012 | 19.180 | 50.583 | 474 |
| legacy_no_centering_paired6 | 30.040 | 33.725 | 45.878 | 19.764 | 50.453 | 452 |
| legacy_no_centering_Dev_best | 29.732 | 33.226 | 45.960 | 19.334 | 50.462 | 469 |
| restored_cycle_IR_reference | 30.275 | 33.970 | 46.074 | 19.989 | 50.622 | 436 |
| IR_image_roll | 29.814 | 32.796 | 46.246 | 19.320 | 50.768 | 476 |
| IR_unavailable | 29.529 | 33.072 | 46.187 | 18.968 | 50.897 | 450 |
| source_RGB_fresh | 29.529 | 33.072 | 46.187 | 18.968 | 50.897 | 450 |
| source_RGB_canonical | 29.529 | 33.072 | 46.187 | 18.968 | 50.898 | 450 |

All9 cases cover7 sequences/15985frames; all39 combined metrics, per-sequence core values and officialCSV hashes are retained.
Current primary remains unchanged. No goal completion, new architecture or contribution is inferred from this coverage backfill.
Shared producer runtime and cached replay FPS are not individual-method end-to-end FPS.
