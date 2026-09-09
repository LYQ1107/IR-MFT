# Historical reference baseline

This result was obtained previously from the visible-only ByteTrack pipeline.
It is used as a reference in the progressive RGB-IR development and is not
re-run in this development round.

```text
Method: ByteTrack visible-only
Dataset: MFT26-IR test split

IDF1       34.1
MOTA       51.8
Recall     74.8
Precision  76.8
FP         32691
FN         36553
IDSW       560
```

The first numeric column in the supplied comparison table is treated as IDF1:
these values match the existing motmetrics log. HOTA is not separately
reported in that historical table.

Source log:

```text
/data1/LWR/vranlee/CELL/ByteTrack/bytetrack_mft26ir_visible_motmetrics.txt
```

This document records the value only; it does not assert that the old training
and evaluation environment can be reproduced from the clean development copy.
