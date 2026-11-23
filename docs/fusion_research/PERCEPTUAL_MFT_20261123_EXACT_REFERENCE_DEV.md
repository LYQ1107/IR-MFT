# Exact same-anchor RGB association confidence: all6 Dev cases

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

Fixedseed0 only. All3offcontrols must exactly reproduce39fields, both tracks anddetailedCSV before3on cases. Only highRGBreference>=.6 at SAMEoriginaldenseanchor changes primary/pending confidence costs; weakRGB retains fusedconfidence, allsource detections/births/outputconfidence/secondaryIoU unchanged. No newpixels/NN/fits, no standalone novelty credit.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EX_gray_off | 48.111 | 68.445 | 33.856 | 55.609 | 89.966 | 103 | 0 | 0 |
| EX_gray_on | 48.190 | 68.452 | 33.963 | 55.613 | 89.987 | 102 | 0 | 0 |
| EX_ir_off | 50.515 | 68.697 | 37.170 | 59.907 | 90.235 | 86 | 0 | 0 |
| EX_ir_on | 50.515 | 68.695 | 37.170 | 59.906 | 90.233 | 86 | 0 | 0 |
| EX_rgb_off | 48.945 | 68.884 | 34.812 | 57.526 | 90.237 | 93 | 0 | 0 |
| EX_rgb_on | 48.927 | 68.924 | 34.766 | 57.307 | 90.241 | 93 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
