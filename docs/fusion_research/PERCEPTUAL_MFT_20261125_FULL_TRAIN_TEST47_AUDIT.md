# Full-Train frozen family: independent complete Test audit

Independent extraction of all13frozen/all7Test15985frame original276column CSVs. All47pervideo/combined,actualMOT/proposal/checkpoint/sourcehashes and unchanged Devfreeze audited. Original10case independent B4prefix costs retained;three unmeasured cases explicitly unknown. Cached replay is not NN cost. No Test reselection,new fit,IRregeneration,extra seed,novelty or completed-goal claim.

Frozen primary: FTT_primary_IR; HOTA 29.834177.
Versus fixed original RGB29.529: +0.305177.
Versus frozen own-Dev-best RGB: +0.366995.

| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FTT_primary_IR | 29.834177 | 46.162697 | 19.367217 | 33.058532 | 51.041322 | 459 |
| FTT_matched_rgb | 29.467182 | 46.240558 | 18.864496 | 32.337002 | 50.877774 | 456 |
| FTT_matched_gray | 29.421963 | 45.966567 | 18.917890 | 32.165328 | 51.120680 | 454 |
| FTT_no_S_paired | 29.581149 | 46.386249 | 18.969881 | 32.738399 | 50.955062 | 451 |
| FTT_no_S_Dev_best | 29.736962 | 46.587744 | 19.066636 | 32.773894 | 50.788755 | 460 |
| FTT_no_C_paired | 29.357239 | 46.105122 | 18.773373 | 32.303165 | 50.761152 | 454 |
| FTT_no_C_Dev_best | 29.419431 | 46.128683 | 18.843435 | 31.896815 | 50.830159 | 476 |
| FTT_no_R_paired | 30.256090 | 46.144467 | 19.933313 | 33.917475 | 50.772883 | 455 |
| FTT_no_R_Dev_best | 31.060100 | 46.615572 | 20.790320 | 34.614301 | 51.056503 | 448 |
| FTT_IR_image_roll | 29.210388 | 46.319294 | 18.499609 | 32.140861 | 50.885365 | 447 |
| FTT_IR_unavailable | 29.528964 | 46.186546 | 18.967559 | 33.072407 | 50.897096 | 450 |
| FTT_source_RGB_fresh | 29.528964 | 46.186546 | 18.967559 | 33.072407 | 50.897096 | 450 |
| FTT_source_RGB_canonical | 29.528971 | 46.186593 | 18.967548 | 33.072407 | 50.898476 | 450 |

| Removed mechanism | Primary minus paired Test | Primary minus own-Dev-best removal Test |
| --- | ---: | ---: |
| S | +0.253029 | +0.097215 |
| C | +0.476939 | +0.414747 |
| R | -0.421913 | -1.225923 |

All original negative cases retained. Three complete Train traversals per fit,not72full epochs. HistoricalTestandpretrainingDev exposure remain. Independent contribution count0;globalgoal remains active.
