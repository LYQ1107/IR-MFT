# 完整开发集：普通RGB分数校准强对照

Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.

All3 fullDev cases complete. Frozen originalRGB headFP32 on1152 cachedTrain images yielded17163 labeled proposals; temperature/Platt maps fit onTrain only with equalvideo naturalclass proportions, noDev parameter search. Identity reproduces all39 originalRGB metrics, bothtracks anddetailedCSV exactly. Temperature44.186 andPlatt33.043 do not reproduce tentativeIR50.515. These two fixed procedures do not rule out all calibration methods. CanonicalNMS>=0.1 candidates only; no new low-scoreanchors. NoIR,Test ornoveltyclaim.

| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SC_identity | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 | 0 | 0 |
| SC_platt | 33.043 | 61.986 | 17.709 | 31.735 | 78.220 | 469 | 0 | 0 |
| SC_temperature | 44.186 | 68.881 | 28.390 | 49.968 | 90.157 | 130 | 0 | 0 |

No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.

Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.
