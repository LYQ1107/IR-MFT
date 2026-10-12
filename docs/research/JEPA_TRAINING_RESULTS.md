# Genuine Fish-JEPA training and controlled tracking results

Status: EXPERIMENTS_COMPLETE. This table is measured; research validation and full online stages can still be pending.

Five fit train videos:21,443queries. SN015 calibration:10,955queries. Eightepochs,128batch,168steps/epoch, equal query order and separate augmentation RNG streams. Training-only clean current/future teacher outputs are stopped; actual EMA steps follow successful optimizer updates. All inference exports exclude teacher and optimizer. No backbone retraining or new featurebank.

Frozen RGB encoders used sixtrain videos; V3 detector had development-video exposure. No whole-system blind-test claim. Historical7video tunedtest is not used for selection. Three seeds measure initialization/encoder variation on two development videos, not scene-generalization uncertainty.

## Training-calibration diagnostics

|Model|Student params|EMA updates|Clean rank1|Actual rank1|Actual N01−N10|
|---|---:|---:|---:|---:|---:|
|jepa_v1_J3_seed0|281344|1344|99.206|61.543|-82|
|jepa_v1_J3_seed1|281344|1344|99.571|61.716|-66|
|jepa_v1_J3_seed2|281344|1344|99.626|61.880|-47|
|jepa_v1_J4_seed0|281344|1344|99.498|61.807|-53|
|jepa_v1_J5_seed0|281344|1343|63.058|39.233|-2526|
|jepa_v1_J6_seed0|281344|1344|81.314|50.735|-1266|
|jepa_v1_J6_seed1|281344|1344|94.158|57.919|-482|
|jepa_v1_J6_seed2|281344|1344|96.248|60.657|-181|
|jepa_v1_J7_seed0|282210|1344|82.939|53.784|-932|
|jepa_v1_frozen_encoder_seed0|281344|1343|97.618|60.393|-208|
|jepa_v1_no_ema_seed0|281344|0|89.521|55.491|-745|
|jepa_v1_no_ema_seed1|281344|0|94.678|58.512|-417|
|jepa_v1_no_ema_seed2|281344|0|92.214|57.380|-540|
|jepa_v1_ordinary_gru_error_seed0|281344|0|58.895|38.622|-2593|
|jepa_v1_ordinary_gru_seed0|281344|0|59.900|37.590|-2706|
|jepa_v1_ordinary_transformer_error_seed0|289296|0|69.512|46.837|-1693|
|jepa_v1_ordinary_transformer_seed0|289296|0|76.440|45.742|-1813|
|jepa_v1_ordinary_transformer_seed1|289296|0|72.031|42.072|-2218|
|jepa_v1_ordinary_transformer_seed2|289296|0|66.326|47.686|-1602|
|jepa_v2_future_J3_seed0|281344|1344|98.749|61.369|-101|
|jepa_v2_future_J6_seed0|281344|1344|78.247|50.397|-1303|
|jepa_v2_future_frozen_encoder_seed0|281344|1343|96.741|59.827|-270|
|jepa_v2_future_no_ema_seed0|281344|0|87.951|54.633|-839|
|jepa_v3_gru_latent_identity_seed0|281344|1343|74.797|44.473|-1952|
|jepa_v3_gru_no_ema_latent_identity_seed0|281344|0|76.157|48.197|-1544|
|jepa_v3_gru_ordinary_latent_identity_seed0|281344|0|57.252|37.937|-2668|
|jepa_v3_transformer_J6_seed0|289296|1344|81.351|51.666|-1164|
|jepa_v3_transformer_latent_identity_seed0|289296|1342|87.960|55.993|-690|
|jepa_v3_transformer_ordinary_latent_identity_seed0|289296|0|72.707|47.805|-1587|

Fixed-history retrospective query ranking is not actual online identity correction. Original artificially masked diagnostics used unmaskedmean as comparator; use separate equal-visible stress results for a fair recovery comparison. Allocated student parameter counts include maskedtraininghead; runtime does not execute that head. Samearchitecture noEMA controls preserve the identical architecture/activeauxiliary losses. GRU/Transformer nominalcounts differ<3%; inferenceactive counts differ about5%, so no exact activecapacity equality claim.

## Fixed-proposal development tracking

All methods share frozen proposals, seedmatched RGB coordinates, CFP lifecycle and actual current matchedmeasurement output. Original ordinaryGRU currentmeasurement three-seedmean HOTA53.591386 is a stronger output baseline than its previous51.885 Kalmanbox output. No per-video threshold.

|Variant/weight|Seeds|HOTA|AssA|DetA|IDF1|MOTA|IDSW|
|---|---:|---:|---:|---:|---:|---:|---:|
|r1_J3_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r1_J3_w0.3|3|52.8215 ± 0.1743|37.9934 ± 0.2245|73.4947 ± 0.0754|61.3025 ± 0.8076|93.7109 ± 0.0077|80.0000 ± 1.6330|
|r1_J3_w0.5|1|50.2103|34.9635|72.1611|57.3392|93.6829|87.0000|
|r1_J4_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r1_J4_w0.3|1|52.6318|37.7248|73.4848|60.1641|93.7152|81.0000|
|r1_J4_w0.5|1|50.3123|34.7757|72.8481|57.3432|93.6872|87.0000|
|r1_J5_w0.1|1|51.0061|35.4056|73.5407|59.2610|93.6506|84.0000|
|r1_J5_w0.3|1|53.4877|38.9630|73.4832|62.3325|93.6958|81.0000|
|r1_J5_w0.5|1|50.3505|35.1325|72.2423|54.6945|93.6958|92.0000|
|r1_J6_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r1_J6_w0.3|3|52.8635 ± 0.5279|38.0534 ± 0.8184|73.5073 ± 0.1403|61.5024 ± 1.1103|93.7037 ± 0.0067|81.6667 ± 0.9428|
|r1_J6_w0.5|1|50.8125|35.3205|73.1658|57.8480|93.6958|88.0000|
|r1_J7_w0.1|1|50.8025|35.1154|73.5588|59.2542|93.6506|86.0000|
|r1_J7_w0.3|1|50.6911|34.9137|73.6657|57.7094|93.6721|84.0000|
|r1_J7_w0.5|1|49.2678|33.3195|72.9190|55.4569|93.6872|92.0000|
|r1_frozen_encoder_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r1_frozen_encoder_w0.3|1|53.4998|38.9646|73.5136|62.3240|93.7065|79.0000|
|r1_frozen_encoder_w0.5|1|50.0154|34.2373|73.1239|56.7638|93.7130|83.0000|
|r1_no_ema_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r1_no_ema_w0.3|3|52.9518 ± 0.3990|38.1790 ± 0.5849|73.5047 ± 0.0871|61.8612 ± 0.4645|93.7030 ± 0.0167|81.6667 ± 3.8586|
|r1_no_ema_w0.5|1|48.0149|31.7136|72.7748|53.6115|93.6506|96.0000|
|r1_ordinary_gru_error_w0.1|1|50.8025|35.1154|73.5588|59.2542|93.6506|86.0000|
|r1_ordinary_gru_error_w0.3|1|51.8271|36.5903|73.4698|60.0991|93.7130|78.0000|
|r1_ordinary_gru_error_w0.5|1|51.2843|36.2520|72.6003|58.2958|93.6721|95.0000|
|r1_ordinary_gru_w0.1|1|51.1018|35.5073|73.6051|59.2604|93.6527|82.0000|
|r1_ordinary_gru_w0.3|1|52.8618|38.0680|73.4672|61.9129|93.7044|80.0000|
|r1_ordinary_gru_w0.5|1|50.2453|34.8988|72.4104|57.9068|93.6140|103.0000|
|r1_ordinary_transformer_error_w0.1|1|50.7961|35.1249|73.5201|59.2593|93.6183|87.0000|
|r1_ordinary_transformer_error_w0.3|1|50.8759|35.2699|73.4512|58.7530|93.6743|83.0000|
|r1_ordinary_transformer_error_w0.5|1|48.8267|32.7509|72.8647|54.0788|93.6549|100.0000|
|r1_ordinary_transformer_w0.1|1|51.9850|36.7649|73.5570|60.8170|93.6786|83.0000|
|r1_ordinary_transformer_w0.3|3|53.5336 ± 1.3727|39.0417 ± 1.9644|73.5024 ± 0.1246|62.2049 ± 1.3535|93.7008 ± 0.0206|81.6667 ± 1.8856|
|r1_ordinary_transformer_w0.5|1|46.8343|30.0689|73.0244|51.5389|93.6657|108.0000|
|r2_future_J3_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r2_future_J3_w0.3|1|52.6309|37.7253|73.4815|60.1635|93.7087|83.0000|
|r2_future_J3_w0.5|1|50.1647|34.5931|72.8027|57.3367|93.6786|91.0000|
|r2_future_J6_w0.1|1|50.8010|35.1137|73.5578|59.2520|93.6506|86.0000|
|r2_future_J6_w0.3|1|52.9364|38.1825|73.4525|61.7983|93.6958|81.0000|
|r2_future_J6_w0.5|1|51.0369|35.6677|73.0986|59.1695|93.6958|92.0000|
|r2_future_frozen_encoder_w0.1|1|50.9050|35.2525|73.5683|59.2559|93.6657|85.0000|
|r2_future_frozen_encoder_w0.3|1|53.4999|38.9646|73.5138|62.3240|93.7065|79.0000|
|r2_future_frozen_encoder_w0.5|1|48.7644|32.5487|73.1224|55.1513|93.7173|90.0000|
|r2_future_no_ema_w0.1|1|50.9047|35.2523|73.5679|59.2537|93.6614|85.0000|
|r2_future_no_ema_w0.3|1|53.4055|38.8574|73.4575|62.3240|93.6958|82.0000|
|r2_future_no_ema_w0.5|1|48.0157|31.7126|72.7793|53.6019|93.6527|97.0000|
|r3_gru_latent_identity_w0.1|1|50.8010|35.1137|73.5578|59.2520|93.6506|86.0000|
|r3_gru_latent_identity_w0.3|1|52.8535|38.0736|73.4320|61.8028|93.6958|84.0000|
|r3_gru_latent_identity_w0.5|1|48.9327|32.8449|72.9574|55.4601|93.6915|94.0000|
|r3_gru_no_ema_latent_identity_w0.1|1|50.9047|35.2522|73.5681|59.2559|93.6657|85.0000|
|r3_gru_no_ema_latent_identity_w0.3|1|53.8429|39.4967|73.4568|62.7794|93.7216|83.0000|
|r3_gru_no_ema_latent_identity_w0.5|1|49.8240|34.0733|72.9228|56.9057|93.6893|95.0000|
|r3_gru_ordinary_latent_identity_w0.1|1|50.7989|35.1127|73.5539|59.2471|93.6441|86.0000|
|r3_gru_ordinary_latent_identity_w0.3|1|53.9974|39.7547|73.3922|62.9798|93.7087|82.0000|
|r3_gru_ordinary_latent_identity_w0.5|1|49.8889|34.0874|73.0859|56.0966|93.6829|95.0000|
|r3_transformer_J6_w0.1|1|50.9065|35.2542|73.5691|59.2559|93.6614|85.0000|
|r3_transformer_J6_w0.3|1|54.4027|40.4237|73.2719|64.3937|93.6872|84.0000|
|r3_transformer_J6_w0.5|1|48.9723|33.0709|72.5915|54.4226|93.6700|97.0000|
|r3_transformer_latent_identity_w0.1|1|50.8998|35.2634|73.5302|59.2610|93.6291|86.0000|
|r3_transformer_latent_identity_w0.3|1|54.2952|40.0831|73.6009|64.1240|93.6936|82.0000|
|r3_transformer_latent_identity_w0.5|1|47.5078|30.8699|73.1929|52.4585|93.6570|97.0000|
|r3_transformer_ordinary_latent_identity_w0.1|1|50.8010|35.1148|73.5554|59.2514|93.6484|86.0000|
|r3_transformer_ordinary_latent_identity_w0.3|1|52.7645|37.9539|73.4159|62.0512|93.6915|79.0000|
|r3_transformer_ordinary_latent_identity_w0.5|1|47.8550|31.3392|73.1472|52.5371|93.6614|100.0000|

Full metric vectors and BT005/SN013 results, individualseed outputs, model/source/input hashes are in the adjacent measured JSON. Means combine official per-seed COMBINED_SEQ results; no averaging of video scores to invent a combined metric.

Actual own-child epoch3 interruption/resume proof is retained in local phase5/jepa_J3_actual_resume_v1.proof.json: all student/teacher/optimizer/scaler/RNG/EMA/epochloss/queryorder states exactly match continuous training, excluding elapsedseconds. The proof applies to the original audited trainer version; thirdround trainer is separately snapshotted.

Reproduction: run the committed frozen training plans with tools/cfp_run_jepa_training.py, then corresponding replay plans with tools/cfp_run_recovery_campaign.py. Largeweights and traces remain local under /data3/liuyeqiang/IR-MFT_assets/cfp/phase5. Full onlinetiming uses tools/cfp_run_jepa_online.py and independent manifests; cached association timing is never end-to-end FPS.
