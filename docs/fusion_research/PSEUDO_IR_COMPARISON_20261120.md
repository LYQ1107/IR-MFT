# Downstream comparison of the November DiffV2IR images

Status: **COMPLETE — eight main and two historical-reference evaluations audited.**

The frozen plan is `configs/pseudo_ir_comparison_20261120.json`. This is a
new experiment using the actual November fixed-caption DiffV2IR PNGs. Earlier
HOTA 49.297/53.591 identity experiments and F1–F3 background-difference
experiments are historical evidence and are not substituted for these runs.

## Completed short-budget results

The preserved single-branch visible detector has now been evaluated on both
complete splits with the common ByteTrack settings below. These are reference
results. Both auxiliary training runs completed three calibration plus three
joint epochs. The RGB control also completed its three joint epochs. All four
arms now have complete Dev and Test evaluations with verified frame coverage.

| Method | Split | Frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RGB ByteTrack | Dev | 4,834 | 43.936 | 70.257 | 27.514 | 50.402 | 91.472 | 83 |
| Original RGB ByteTrack | Test | 15,985 | 29.191 | 43.866 | 19.570 | 30.932 | 47.207 | 390 |
| V3 + old legacy IR | Dev | 4,834 | 45.764 | 67.849 | 30.899 | 52.179 | 90.345 | 89 |
| V3 + old legacy IR | Test | 15,985 | 29.738 | 46.595 | 19.068 | 33.736 | 51.667 | 455 |
| V3 + new DiffV2IR | Dev | 4,834 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 |
| V3 + new DiffV2IR | Test | 15,985 | 29.195 | 43.916 | 19.487 | 32.678 | 49.603 | 383 |
| RGB matched fine-tune + ByteTrack | Dev | 4,834 | 47.579 | 68.864 | 32.908 | 55.865 | 90.200 | 94 |
| RGB matched fine-tune + ByteTrack | Test | 15,985 | 29.529 | 46.187 | 18.968 | 33.072 | 50.898 | 450 |

At this short budget, new-minus-old Dev HOTA is+0.472 percentage points,
IDF1+0.940, MOTA-0.121 and IDSW+19. The changes are mixed. Both paired arms
are below the equally fine-tuned RGB control on Dev: new-minus-matched-RGB
HOTA-1.343pp, AssA-1.771pp, IDF1-2.746pp, MOTA+0.024pp and IDSW+14. Thus the
improvement over the original unadapted RGB checkpoint cannot be attributed
to the auxiliary images alone. These negative controls are retained.

The complete Test reverses the new-versus-old Dev direction: new-minus-old
HOTA-0.543pp, DetA-2.679pp, AssA+0.419pp, IDF1-1.058pp, MOTA-2.064pp and
IDSW-72. Against the original unadapted RGB checkpoint, new-input HOTA changes
by only+0.004pp, while IDF1+1.746pp, MOTA+2.396pp and IDSW-7. This is not a
clear HOTA improvement. The equally fine-tuned RGB Test control reaches
HOTA29.529. New-minus-matched-RGB Test HOTA is-0.334pp, DetA-2.271pp,
AssA+0.519pp, IDF1-0.394pp, MOTA-1.295pp and IDSW-67. The short budget
therefore does not establish a benefit from the new auxiliary input over
either the matched RGB control or the old auxiliary input.

The largest new-versus-old loss is the dense49-fish MSK-003 sequence:
HOTA-8.503pp and DetA-13.883pp. BT-002/BT-004/PF-002 improve in HOTA, while
MSK-003/SN-009/SN-011/SN-014 decline. Official sequence combination, rather
than an unweighted average of those seven changes, gives the aggregate above.
Across Test, old/new CLEAR false negatives are43,371/49,455 and false
positives26,214/23,193; emitted track detections are127,755/118,650. The72
fewer switches occur alongside6,084 more misses and9,105 fewer emitted
detections, so switch counts alone do not establish better association.
The observed downstream failure is concentrated in tracker-output recall on the dense scene;
the generator's causal mechanism is not proven by these downstream scores.

Both paired joint training losses were still decreasing at the final epoch;
three calibration plus three
joint epochs are an initial adaptation comparison, not evidence of convergence.
The user requested a training-adequacy check, so a separate Dev-driven plan
extends all three adapted arms to at least12 and at most24 joint epochs:
[`PSEUDO_IR_BUDGET_EXTENSION_20261121.md`](PSEUDO_IR_BUDGET_EXTENSION_20261121.md).
These short-budget results and checkpoints remain intact.

![Audited short-budget overview](PSEUDO_IR_COMPARISON_20261120_OVERVIEW.png)

The overview shows both full-split HOTA tables, every Test sequence's
new-minus-old HOTA and the still-decreasing joint losses. It is generated
from the audited JSON by `tools/plot_pseudo_ir_comparison.py`; the185,768-byte
PNG is a lightweight analysis artifact. Original RGB has no adaptation;
the other three arms have three joint epochs and the paired arms also have
their three calibration epochs.

The machine-readable
[`PSEUDO_IR_COMPARISON_20261120_RESULTS.json`](PSEUDO_IR_COMPARISON_20261120_RESULTS.json)
records completion status, source hashes and all official TrackEval metrics.
[`PSEUDO_IR_COMPARISON_20261120_TABLES.md`](PSEUDO_IR_COMPARISON_20261120_TABLES.md)
includes individual sequence results and supplementary detection components.
The final export passed `--require-complete --require-historical-reference`:
eight main and two secondary-reference evaluations, with zero missing items.
The secondary historical V3 replay uses its preserved checkpoint and original
normalization, with a different training history from the matched arms:

| Historical reference | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 46.091 | 70.212 | 30.276 | 52.621 | 92.011 | 106 |
| Test | 30.551 | 48.513 | 19.336 | 33.676 | 51.957 | 504 |

This frozen reference is stronger on Test HOTA than all three short-budget
adaptations. It is included as context, without treating its different
training history as a matched-input ablation. Earlier historical Test31.091
is not substituted for this fresh30.551 replay under the current protocol.

## Inputs and method

The main method is the previous V3 gated-residual detector followed by the
unchanged ByteTrack association. It isolates the auxiliary image treatment.
The A/B/C identity modules are not part of this campaign and are not credited
with any gain in this comparison.

| Arm | Detector initialization | Auxiliary data | Adaptation |
| --- | --- | --- | --- |
| Original RGB ByteTrack | preserved visible checkpoint | none | none |
| V3 + old IR | same visible checkpoint copied into both branches | original dataset `infrared/*.jpg` | 3 IR-only calibration epochs + 3 V3 joint epochs |
| V3 + new IR | same visible checkpoint copied into both branches | November DiffV2IR grayscale PNGs | identical old-IR budget |
| RGB training control | same visible checkpoint | none | same 3 RGB/head fine-tuning epochs as the paired arms |

At the new execution audit, the old causal-cache frame directories were
empty although their metadata remained. The paired old-IR arm therefore uses
the physically present archived `infrared` JPEGs, explicitly recorded in the
frozen-plan amendment before paired training. It is not labelled as a causal
background-difference run.

The original DiffV2IR files remain in place at
`/data3/liuyeqiang/nas/A100/IR-MFT/diffv2ir_final_20261104/full_generation`.
Dev has its own auxiliary directory even though the Dev RGB frames remain in
the original dataset's `train` folder. The reader repeats scalar intensity
into three channels without an INFERNO colour map. Missing generated files
raise an error; no other image source or split is substituted.

A read-only audit of the preserved generation implementation confirmed that
SAM calls `SamAutomaticMaskGenerator.generate(rgb)` and unions its automatic
masks. GT boxes/masks are not supplied as SAM prompts in that implementation;
the annotation table is used to enumerate frame IDs, paths and splits.
DiffV2IR conditions on the current RGB, automatic SAM union and fixed caption.
Six source hashes and two representative Train/Test metadata records are in
[`PSEUDO_IR_GENERATION_PROVENANCE_20261120.json`](PSEUDO_IR_GENERATION_PROVENANCE_20261120.json).
This source audit does not remove the historical Test-based configuration
selection limitation or independently replay all generation jobs.

## Fixed training and evaluation

Train consists of six complete videos / 27,251 frames; Dev consists of
BT-005 and SN-013 / 4,834 frames; historical Test consists of seven complete
videos / 15,985 frames. New adaptation excludes Dev. Both paired arms use
seed0, 480x864 training, effective batch8, SGD with momentum0.9, Nesterov,
weight decay0.0005 and the same phase-wise cosine schedule. Original V2/V3
rates are scaled from the reference batch64: effective batch8 starts at
LR0.000125 for IR calibration and LR0.000025 for V3 joint fine-tuning.
The per-image LR semantics were corrected before any paired-arm evaluation;
the incomplete earlier prototype is preserved separately and excluded.
The synchronized horizontal flip is the only random image augmentation.
BatchNorm running statistics remain fixed for the small training batch in
all arms. EMA weights from the fixed final epoch are evaluated.

The new input uses exact full-Train scalar normalization mean0.3187656968 /
std0.1145435886, replicated across channels. The old input uses an exact
RGB-order histogram over all original Train IR pixels; no Dev/Test pixels
are used to calculate normalization.

All tracking evaluations use 800x1440, confidence0.1, NMS0.7,
track-threshold0.6, buffer30, match-threshold0.9, frame-rate25 and
minimum box area100 with the same protocol-derived GT and TrackEval.
The final table must include HOTA, DetA, AssA, IDF1, MOTA and IDSW, with
new-minus-old and new-minus-RGB differences. It must not subtract the earlier
six-video identity experiment from the two-video Dev experiment.

Only physical GPU2/GPU3 are used. Jobs on other GPUs and unrelated jobs on
these cards are not modified. Concurrent load makes recorded throughput a
shared-host observation, not a controlled speed benchmark. Inference batches
independent frames and then runs ByteTrack sequentially; reported replay FPS
is not raw single-frame end-to-end online throughput.

## Storage and recovery

Experiment outputs are external under
`/data3/liuyeqiang/IR-MFT_assets/pseudo_ir_comparison_20261120`.
Each training arm retains one atomic checkpoint with training model, EMA,
optimizer, scaler and epoch-boundary RNG states; no epoch-snapshot series is
created. No durable image copy is made. A Train-only, at most10 GiB volatile
`/dev/shm` byte cache is allowed to avoid repeated 0.1--0.4 second NAS reads.
Every cached PNG is checked against its source SHA256, no image transform is
performed, and host available memory keeps a40 GiB reserve. Switching roots
uses an exact saved epoch boundary and RNG state. The temporary cache is
removed after training. Evaluation continues to use the original NAS root.
The task-owned tmpfs directory was removed on2026-11-21 after the final
new-input checkpoint completed; all original NAS images remain untouched.
The prepared cache contains all27,251 Train PNGs /7,748,699,851 bytes
(7.22 GiB), each byte-verified. Training resumed from the complete first-epoch
checkpoint with its saved model, optimizer, EMA, scaler and RNG states. The
first100 updates of epoch2 reproduce the NAS run's mean loss exactly
(3.2701871585845947); this is a scalar-prefix recovery check, not a claim that
all parameters were independently compared bit by bit. The compact record is
[`PSEUDO_IR_COMPARISON_20261120_IO_AUDIT.json`](PSEUDO_IR_COMPARISON_20261120_IO_AUDIT.json).
Only compressed post-NMS boxes,
text tracks and small reports are exported. The local reserve is 40 GiB.
A disposable 2.1 GiB smoke checkpoint was removed after the training check.

After all ten short-run evaluations completed, the three immutable full
checkpoints were archived under
`/data3/liuyeqiang/nas/A100/IR-MFT/pseudo_ir_checkpoint_archive_20261121`.
`tools/archive_pseudo_ir_prefix_checkpoints.py` first verified each local
source, copied it, flushed it and independently read the NAS file back for a
complete SHA256 check, then atomically replaced the local file with a symlink.
The original paths, full states and hashes remain available. This released
5,664,713,777 bytes /5.276 GiB locally; the short output folder now occupies
about115 MiB locally. All eight main plus two reference audits still pass,
and their entire exported JSON is identical to the pre-archive export.
[`PSEUDO_IR_CHECKPOINT_ARCHIVE_20261121.json`](PSEUDO_IR_CHECKPOINT_ARCHIVE_20261121.json)
records the targets and checksums. Active continuation checkpoints stay local;
only lightweight code, reports and audit metadata are committed to GitHub.

`tools/launch_pseudo_ir_campaign.py` runs dependent jobs per GPU and retains
commands, PIDs, logs and failure states. Resume uses the saved full epoch
boundary; no incomplete run is silently counted as a completed experiment.
During terminal cleanup, the first RGB-control Test worker ended after14,592
of15,985 frames without saving complete metrics; a reference-model Test
attempt also ended after1,024 frames. Those attempts are excluded and their
logs retained. Only these two incomplete Test replays were restarted from
frame1 with the same weights and settings. The queues now run in independent
OS sessions with file logging and no terminal stdin. The still-idle extension
controller was moved the same way before starting any additional training.
[`PSEUDO_IR_COMPARISON_20261120_EXECUTION_AUDIT.json`](PSEUDO_IR_COMPARISON_20261120_EXECUTION_AUDIT.json)
records the observed process absence, progress and recovery commands.
`tools/summarize_pseudo_ir_comparison.py --require-complete` verifies frame
and sequence coverage and common tracker thresholds before final reporting.

The reader tests verify Dev namespace isolation, scalar replication and
failure on missing generated frames. The full local suite passed 9 tests
with the actual dataset root set; GPU smoke training had finite loss and no
skipped updates after standard AMP scale handling was corrected.

## Scientific limits

The visible initialization historically saw all eight original Train videos,
including the current Dev videos. The historical Test videos participated in
past tracking tuning and DiffV2IR configuration probes. This comparison is
therefore a frozen downstream diagnostic, not a new blind benchmark. The
initial comparison uses one training seed. Any positive result must be
reported with these limits and cannot establish independent-sensor thermal
information or seed-independent improvement.
