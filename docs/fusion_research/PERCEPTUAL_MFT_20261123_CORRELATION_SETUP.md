# Shared-encoder local correlation repair: validated setup, research ongoing

The requested >=34.529 Test HOTA and2-3 independently positive contributions
are not established. The fixed direct RGB baseline is YOLOX-X + ByteTrack,
29.529 Test HOTA; the prior three-seed new-IR point control averages31.710.
No current correlation method has been evaluated on Test.

The complete earlier controls show two distinct failures: whole-image IR
attention can select generated tank borders, and direct native-IR point flow
is less reliable than RGB physical correspondence. Retain all negative evidence
in the adjacent foveation and semantic-point reports. A single priority
pipeline reaches48.461 Dev, but uses RGB-guided crops on a paired global
detector; it is not an IR-attention or equal-parameter contribution claim.

The new proposed modules are: local5x5 cosine-correlation readout to tolerate
IR geometry errors; an agreement/uncertainty gate and original-observation
priority; and wrong-auxiliary counterfactual consistency within each Train
video. These are candidate mechanisms, pending complete removal experiments.

All methods use the same frozen original RGB encoder and head. Only217619
adapter parameters are fitted; the RGB-repeat and normalized grayscale RGB
controls have the same capacity. Gray and IR each require an auxiliary
encoder pass with shared weights. RGB-repeat can reuse its identical features
and is cheaper; end-to-end costs must be reported separately. No separately
pretrained IR encoder gives this campaign a hidden fitting-budget advantage.

Train-only calibration uses the same192 uniformly spaced frames in each of
the6 original Train videos,1152 total. Frozen FP16 FPN arrays stay on the
independent work disk, under the64GiB cap. No source image dataset is copied.
Both allowed GPUs passed actual-data gradient checks with exactly4 updates,
equal parameter counts and equal sample digests. Initial one-step checks are
retained separately, not misreported as four-step checks.

On both Dev videos, all first8 fresh RGB proposal arrays match the original
baseline cache exactly: detection count, coordinates and scores. All three
zero-initialized auxiliary adapters also preserve these predictions exactly.
The active fusion layer's output is independent of other samples in a batch.
Original eval BN is fixed; batch4 contains independent image samples, never
cross-image statistics or cross-frame attention. Buffering can add up to3
frame intervals of latency, as in the original batch4 detector; any final
end-to-end timing must include it. The training-only head helper
has identical box decoding; applying sigmoid after concatenation can differ
by one float32 ulp, tested with tolerance1e-7. Inference uses the unchanged
original head and passes exact first8-frame actual-image proposal parity.
Full-production checks found4830/4834 bit-exact fresh RGB frames, with all
4834 detection counts unchanged. A separate complete numerical audit finds
the differences only in the final2 frames of each video: maximum box change
0.675 pixels and score change0.001323, with identical0.6/0.7 threshold
membership. This is consistent with different tail batch shapes in per-video
versus concatenated inference. Full-sequence bit-exact parity is not claimed.
The baseline tracking replay uses the unchanged canonical proposals.

Seed0 RGB-repeat, gray and actual-IR pilots all receive24 epochs and the same
epoch-by-epoch draws, preserving6/12/18/24 snapshots. All3 fits completed24 epochs and6912 optimizer steps, with identical24-epoch
sample digests. Before any full Dev inference/accuracy, the evaluation was
extended symmetrically to retain both already computed raw and final-priority
outputs for all12 snapshots, plus the exact baseline:25 official Dev replays.
This requires no extra model forward. The earlier13-row registration remains
archived as superseded before inference; it is not presented as an executed
accuracy result. A
promising method still requires all3 seeds and every module removal before
a contribution claim. Training loss alone is not evidence of HOTA improvement.

All129 maintained tests pass, including3 Train-curriculum validity tests.
The [conditional repair](../../configs/perceptual_correlation_recoverability_20261123.json)
has all6 Train eligibility caches, but has not been activated or fitted.
Its17270 labeled instances contain16437 heuristic eligible instances; this
does not prove that every generated IR object is correct. RGB and gray controls
receive the exact same IR-conditioned Train eligibility and perturbations.
Existing RGB pretraining included current Dev
videos, and historical Test was previously exposed; this is exploratory
research, not a fresh blind benchmark. No new Test chooses current parameters.
