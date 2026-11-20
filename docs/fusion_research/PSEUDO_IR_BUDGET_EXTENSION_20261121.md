# Dev-driven extension of the pseudo-IR training budget

Status: **RUNNING — two paired continuation workers started on physical GPU2/GPU3.**

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
