# Genuine Fish-JEPA training — ongoing results

Status: IN_PROGRESS. Round1 is still running; future-target and third targeted rounds are pending. All values below are measured, not forecasts.

Eight epochs, 21,443 fit queries from five train videos, 10,955 calibration queries from SN-015, batch128,168steps/epoch. Six-video frozen encoder exposure and detector development exposure are disclosed in the baseline audit. Identical query orders across first-round variants; augmentation streams are separate. Teacher target outputs stop gradient; EMA updates only follow successful optimizer steps. Finalepoch8 student export excludes teacher/optimizer.

An actual own-child SIGTERM after epoch3 and resume to epoch8 exactly reproduced student/teacher/optimizer/scaler/EMA updates/all RNG states/epoch losses/query orders; training seconds are excluded from equality. Proof: local `phase5/jepa_J3_actual_resume_v1.proof.json`. Initial full120tests and seven targeted runtime/model tests were executed; verify the individual log for its final result.

## Completed models and training-calibration diagnostics

|Model|Params|EMA updates|Clean rank1|Actual rank1|Actual N01−N10|
|---|---:|---:|---:|---:|---:|
|jepa_v1_J3_seed0|281344|1344|99.206|61.543|-82|
|jepa_v1_J4_seed0|281344|1344|99.498|61.807|-53|
|jepa_v1_J5_seed0|281344|1343|63.058|39.233|-2526|
|jepa_v1_J6_seed0|281344|1344|81.314|50.735|-1266|
|jepa_v1_J7_seed0|282210|1344|82.939|53.784|-932|
|jepa_v1_frozen_encoder_seed0|281344|1343|97.618|60.393|-208|
|jepa_v1_no_ema_seed0|281344|0|89.521|55.491|-745|
|jepa_v1_ordinary_gru_error_seed0|281344|0|58.895|38.622|-2593|
|jepa_v1_ordinary_gru_seed0|281344|0|59.900|37.590|-2706|
|jepa_v1_ordinary_transformer_seed0|289296|0|76.440|45.742|-1813|

Query rank is retrospective fixed-history diagnosis, not actual online corrections. Actual-history mean baseline rank1 is62.291%; clean99.498%. Artificially masked original diagnostic used an unmasked-mean reference and is not a fair masking intervention: retain original fields as legacy evidence, and recompute with equal visible inputs before using stress comparisons.

## Completed fixed-proposal tracking results

Same CFP lifecycle and current matched measurement boxes for all. Original ordinaryGRU seed0 at.3 HOTA53.104; original three-seed ordinary measurement output mean53.591. No per-video thresholds. Fixed appearanceweights.1/.3/.5; no checkpoint selection.

|Model/weight|HOTA|AssA|DetA|IDF1|MOTA|IDSW|
|---|---:|---:|---:|---:|---:|---:|
|jepa_replay_v1_J3_s0_w0.5|50.2103|34.9635|72.1611|57.3392|93.6829|87.0000|
|jepa_replay_v1_J3_s0_w0.3|52.6317|37.7247|73.4848|60.1641|93.7173|80.0000|
|jepa_replay_v1_J3_s0_w0.1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|jepa_replay_v1_ordinary_gru_s0_w0.1|51.1018|35.5073|73.6051|59.2604|93.6527|82.0000|
|jepa_replay_v1_ordinary_gru_s0_w0.3|52.8618|38.0680|73.4672|61.9129|93.7044|80.0000|
|jepa_replay_v1_ordinary_gru_s0_w0.5|50.2453|34.8988|72.4104|57.9068|93.6140|103.0000|
|jepa_replay_v1_ordinary_transformer_s0_w0.3|55.4745|41.8174|73.6347|64.0967|93.7109|79.0000|
|jepa_replay_v1_ordinary_transformer_s0_w0.1|51.9850|36.7649|73.5570|60.8170|93.6786|83.0000|
|jepa_replay_v1_ordinary_transformer_s0_w0.5|46.8343|30.0689|73.0244|51.5389|93.6657|108.0000|
|jepa_replay_v1_J4_s0_w0.3|52.6318|37.7248|73.4848|60.1641|93.7152|81.0000|
|jepa_replay_v1_J4_s0_w0.1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|jepa_replay_v1_J4_s0_w0.5|50.3123|34.7757|72.8481|57.3432|93.6872|87.0000|
|jepa_replay_v1_J5_s0_w0.1|51.0061|35.4056|73.5407|59.2610|93.6506|84.0000|
|jepa_replay_v1_J5_s0_w0.3|53.4877|38.9630|73.4832|62.3325|93.6958|81.0000|
|jepa_replay_v1_J5_s0_w0.5|50.3505|35.1325|72.2423|54.6945|93.6958|92.0000|

Each video and complete metric vector/provenance are in PHASE_V_JEPA_INITIAL_MEASURED.json. OrdinaryTransformer seed0 gain is an ordinary capacity/architecture baseline, not JEPA evidence; requires three seeds. Strict online prefix currently running is only an integration probe, not final FPS.
