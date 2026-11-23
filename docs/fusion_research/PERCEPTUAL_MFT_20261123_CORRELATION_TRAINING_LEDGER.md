# Matched correlation adapter training ledger

24 passes over a fixed1152-frame Train cache,6912 updates per fit; not24 passes over all27251 original Train frames. Train480x864; Dev800x1440.

RGB-repeat reuses RGB encoding; gray and actualIR each require a second shared-weight encoder pass. Equal adapter capacity is not equal backbone compute.

| Fit | Epochs | Updates | Parameters | GPU | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| rgb_full_s0 | 24 | 6912 | 217619 | 2 | 1524.8 |
| gray_full_s0 | 24 | 6912 | 217619 | 2 | 1458.1 |
| ir_full_s0 | 24 | 6912 | 217619 | 3 | 1522.3 |
| rgb_full_s1 | 24 | 6912 | 217619 | 3 | 1229.2 |
| gray_full_s1 | 24 | 6912 | 217619 | 2 | 1303.6 |
| ir_full_s1 | 24 | 6912 | 217619 | 3 | 1181.7 |
| rgb_full_s2 | 24 | 6912 | 217619 | 2 | 1097.5 |
| gray_full_s2 | 24 | 6912 | 217619 | 3 | 1243.1 |
| ir_full_s2 | 24 | 6912 | 217619 | 2 | 1215.7 |

All24 epoch sample digests match across the three cues for each seed. The adjacent JSON retains every epoch history, six cache-manifest hashes, source hashes and four checkpoint hashes per fit.

Training provenance only. No accuracy, robustness or novelty claim. Checkpoints stay outside Git; hashes and small metadata are published.
