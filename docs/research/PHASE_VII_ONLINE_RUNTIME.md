# Phase VII online runtime

Status: IN_PROGRESS. Current measurements time native association/output on frozen V3/RGB caches; no new complete raw online FPS is claimed.

The disabled replay covers684SN-001frames and6,570SN-015frames, with exact historical prediction-byte parity. Its association/output time is4.79s/49.19s; SN-015 p50/p95is7.35/8.96ms. Joint bounded adds new full-set CPU head work and reaches97.54swith p50/p95=14.63/16.58ms; full learned joint reaches111.92swith p50/p95=16.86/19.46ms. The untrained common cost control takes69.62son SN-015with p50/p95=10.49/11.94ms; even its wrapper currently recomputes observable GRU appearance, so it is not a minimum-overhead threshold implementation. These figures are single-run CPU association measurements, not complete-system speed or a statistically confirmed performance comparison.

Training the small heads used two previously idle A100devices and finished in137.80s(joint)/92.25s(pair). PyTorch peak allocated memory was25.14/25.55MB; nvidia-smi showed about540MiBper process including CUDA context. The allocated model has150,533parameters in both modes, with142,084/83,845participating optimizer parameters. This does not include visual encoders, detector, IR construction or raw decode.

The inherited roughly1.2FPS complete-system bottleneck remains a required investigation. A new raw all-frame profiler must include decode, causal IR construction, preprocessing, V3 detector/batch8, RGB crop/encoding/padding128, memory/model work, assignment, writes and output. Report FPS, p50/p95, memory, parameters, countable operations and incremental overhead on the same complete inputs. Every batching/precision/cache optimization first requires input/output/numerical parity; cached replay speed cannot substitute for ProtocolC. All full raw comparisons and new visual-encoder costs remain pending.

## Sparse visual pilot costs

Sparse extraction produced47,751observations and3.462GiBfeatures plus9.38GBnormalized crop cache. Four frozen-ImageNet adapter fits take about19–24seconds each; two genuine backbone fits take99.23seconds(global)/101.68seconds(learned). Precise per-run seconds, allocated/gradient parameters and peak PyTorch allocated memory are in VISUAL_PILOT_MEASURED. AMP overflow skips and effective optimizer updates are explicitly recorded. These costs exclude full raw video decode, causal IR/V3detector and complete crop encoding; they are training/cache measurements, not online FPS. Only idle physical GPU0–3 were used; other users' GPU4–9 jobs were untouched. Free space remains above40GiB.

Global/local backbone controls now complete all three registered seeds at the first fold, with a real epoch2stop/newprocessresume equivalence proof. Runtime remains training/cache-only; no raw onlineFPS claim. Current free space is about148GiB after the crop/features and checkpoint assets; all our GPU0–3 processes have completed, other users' GPU4–9 work is untouched.

Reference auditing used only bounded source text/CPU/network, with no foreign model execution, weight download or GPU allocation. V-JEPA clip recomputation and CoTracker3 chunkdelay/prefix handling must be measured if adopted; no whole-video cached feature/point prediction may be credited as free strict-online work. Current free space remains about147GiB, and GPU4–9 belong to other live jobs.

DINOv2 frozen47,751crop extraction completed with synchronized per-batch cache encoding timings stored outsideGit. Reading input-cache hashes, encoding, serialization and post-write hashes are separate from fullraw ProtocolC; none of these crop timings is reported as complete detector/onlineMOT FPS. New encoder native video integration/fullrawprofiling remains pending.
