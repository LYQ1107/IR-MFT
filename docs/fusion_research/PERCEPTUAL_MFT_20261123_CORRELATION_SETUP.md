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
original head and passes exact actual-image proposal parity.

Seed0 RGB-repeat, gray and actual-IR pilots all receive24 epochs and the same
epoch-by-epoch draws, preserving6/12/18/24 snapshots. Complete12-snapshot
Dev production and13 official replays are registered before accuracy. A
promising method still requires all3 seeds and every module removal before
a contribution claim. Training loss alone is not evidence of HOTA improvement.

All126 maintained tests pass. Existing RGB pretraining included current Dev
videos, and historical Test was previously exposed; this is exploratory
research, not a fresh blind benchmark. No new Test chooses current parameters.
