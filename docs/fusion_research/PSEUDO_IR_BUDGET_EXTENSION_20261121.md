# Dev-driven extension of the pseudo-IR training budget

Status: **RUNNING — first extended Dev evaluation complete; common final budget pending.**

The first extended Dev checkpoint is a negative result. New DiffV2IR HOTA
decreases from46.236 at joint3 to44.233 at joint6, although its full-Train
loss falls from3.087354 to2.965817. Both Dev evaluations cover all4,834 frames
with the same normalization, image size and tracking settings. The current
best model for this arm therefore remains joint3. The minimum12-joint-epoch
budget and shared stopping rule remain unchanged; one later checkpoint does
not establish the shape of the complete learning curve.

| New DiffV2IR | Train loss | Dev HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint3 | 3.087354 | 46.236 | 68.766 | 31.137 | 53.119 | 90.224 | 108 |
| Joint6 | 2.965817 | 44.233 | 68.042 | 28.805 | 52.285 | 89.652 | 105 |
| Joint6 minus joint3 | -0.121537 | -2.003 | -0.724 | -2.332 | -0.834 | -0.572 | -3 |

Old IR has also completed joint6 with Train loss2.942829; its joint6 Dev
replay is queued. The RGB matched control is continuing toward the same
budget. Comparisons between different completed epoch counts are not used
as evidence of an input benefit. No final selected-model Test has run yet.

![Current training and Dev curves](PSEUDO_IR_BUDGET_EXTENSION_20261121_CURVES.png)

The current
[`RESULTS.json`](PSEUDO_IR_BUDGET_EXTENSION_20261121_RESULTS.json),
[`tables`](PSEUDO_IR_BUDGET_EXTENSION_20261121_TABLES.md) and
[`curve CSV`](PSEUDO_IR_BUDGET_EXTENSION_20261121_CURVES.csv)
are explicitly `INCOMPLETE`. They contain completed training epochs and
integrated Dev checkpoints only; pending Dev points are not fabricated.

The first paired comparison contains three auxiliary calibration epochs and
only three joint detector epochs. It starts from an existing visible detector,
so these are adaptation epochs rather than training from scratch. The final
joint training losses still decreased: old IR 3.1141 to 3.0611 and new IR
3.1400 to 3.0874. These observations do not establish convergence. The short
comparison remains available in `PSEUDO_IR_COMPARISON_20261120_RESULTS.json`.
The subsequently completed RGB control reaches Dev HOTA47.579, above the
new-input46.236 and old-input45.764. This negative auxiliary-input control is
retained; a gain over the original unadapted RGB checkpoint alone does not
isolate the effect of adding pseudo IR.

The user requested a check of whether six epochs suffice. The extension plan
was frozen and pushed in commit `daced61` before extended training and before
the short new-input Test evaluation completed:
[`pseudo_ir_budget_extension_20261121.json`](../../configs/pseudo_ir_budget_extension_20261121.json).
Previously observed short outcomes are listed in the plan; this is not a
claim that all earlier Test results were unseen.
All eight short main evaluations and two historical-reference replays passed
the complete coverage/protocol audit before continuation began. The
independent-session launcher started new-input training on GPU2 and old-input
training on GPU3 at2026-11-21 01:53 CST. Both resume the full original
three-calibration-plus-three-joint checkpoint; the first stage adds joint
epochs4–6. The RGB control receives the same minimum/conditional joint budget
as GPU slots become available. CLI maximum24 is the predeclared upper bound,
not a claim that24 epochs have already completed.
The first added joint epoch has now completed all27,251 Train frames in both
paired arms. Old/new mean loss changes from3.061076/3.087354 at joint3 to
3.038350/3.063256 at joint4. Both saved checkpoints report three calibration
plus four joint epochs,23,843 successful EMA updates, the full optimizer and
all five RNG state groups. The original six-epoch histories are preserved
exactly. All1,404 floating tensors in each raw model and each EMA are finite.
Each arm skipped two AMP updates in this added epoch. These are training and
recovery checks; the next full Dev evaluation is scheduled at joint6.

All three adapted arms resume their original full checkpoint: RGB only, V3
with archived legacy IR, and V3 with the November DiffV2IR images. Each must
complete at least 12 joint epochs. The paired arms also retain their original
three calibration epochs. Every third joint epoch has a complete 4,834-frame
Dev evaluation. At joint12 and18, if any arm's best Dev HOTA improved by more
than 0.2 percentage points over its best through six joint epochs earlier,
all arms receive another six epochs. The shared upper bound is24. Reaching
that bound while still improving is reported as a budget cap, not convergence.

The highest Dev HOTA selects the model, with earlier epochs breaking ties.
Each selected model is evaluated once on the complete 15,985-frame Test
split, using the same frozen detection and ByteTrack settings as the short
campaign. Test metrics do not control training length or model selection.
All short outcomes remain visible, including negative metrics.

This is an explicit continuation with a warm restart, not a relabeled
from-scratch 12-epoch cosine run. The first three joint epochs retain their
original effective batch8 learning rates 0.000025/0.000020/0.000010. Additional
joint epochs4–12 restart at0.000025 and decay within a nine-epoch segment.
Conditional epochs13–18 and19–24 start at0.000010 and0.000005 respectively.
Each segment uses the same cosine floor0.2. Every arm uses the same schedule,
seed, data order policy, normalization, augmentation, optimizer, frozen BN
statistics, EMA and epoch-boundary recovery states.

`tools/launch_pseudo_ir_extension.py` schedules one worker on each of physical
GPU2 andGPU3. No other GPU is used. It waits for all eight short main replays
and two historical V3 reference replays before starting the extension. Each
arm stores one atomic full latest checkpoint and one best EMA checkpoint in
FP16, which is the same conversion already used by inference. No epoch
snapshot series or durable image dataset copy is made. Expected additional
checkpoint storage is about6 GiB, with at least40 GiB local free space kept.
A separate Train-only tmpfs cache is limited to10 GiB and byte-verified;
evaluation reads the original NAS, and the cache is removed after completion.
Cache preparation completed:27,251 PNGs /7,748,699,851 bytes, all source
hashes identical to the short comparison's recorded Train bytes. It occupies
7.22 GiB of tmpfs and makes no durable dataset copy.

Validation so far:19 local tests passed, including protection against silent
epoch, LR, seed, batch and prefix-coverage changes. A GPU3 one-batch resume
from the old-input full checkpoint completed joint epoch4's smoke step with
finite loss2.694551, effective LR0.000025, zero skipped AMP updates and
14.464 GiB peak allocated memory. It was labeled `BENCHMARK` and saved no
model checkpoint. CPU scheduling simulations covered both a plateau at12
and continued improvement through the24-epoch cap. These checks validate the
continuation machinery; they do not count as experimental training results.
The compact best-EMA retention also passed a four-frame GPU3 replay check:
the saved frame IDs, offsets,47 boxes and47 scores matched the original
full-checkpoint batch exactly. Its temporary0.35 GiB checkpoint was removed.
The bounded checks and scope are recorded in
[`PSEUDO_IR_BUDGET_EXTENSION_20261121_VALIDATION.json`](PSEUDO_IR_BUDGET_EXTENSION_20261121_VALIDATION.json).

The final export uses `tools/summarize_pseudo_ir_extension.py` to audit the
common budgets, preserved short prefixes, piecewise LR histories, every Dev
selection and stopping decision, final checkpoint hashes, normalization,
tracker settings and contiguous frame coverage. The export includes joint
epoch counts read from each actual source checkpoint rather than the unused
training defaults on the inference command line. It also includes
loss/Dev HOTA/Dev IDF1 curves, all selected metrics, all seven Test sequences
and new-minus-old/RGB differences. The original pretraining, historical
Dev/Test exposure and single-seed limitations in the short report still apply.
The old/new paired arms share their loader seed and phase counts. The RGB
control matches RGB/head epochs and the augmentation policy, but has no
auxiliary calibration passes, so its joint-phase minibatch ordering and EMA
update history are not identical to the paired arms.
