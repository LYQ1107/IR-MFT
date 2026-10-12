# Fish-JEPA failure analysis — in progress

J3 has a genuine trained EMA teacher, but first complete seed0 tracking atweight.3 HOTA52.631715 is below the existing ordinaryGRU current-box control53.104. J4 masking produces52.631763, an essentially identical trajectory-level result. Training latent cosine loss declines, which does not validate identity recovery.

J3 SN015 clean rank99.205% versus mean99.498%; actual61.543% versusmean62.291%, N01=51/N10=133/net−82. Teacher target variance sum.227533 and mean norm.878901 show substantial shared direction, but not exact constant collapse. Student actual variance.206632 also nonzero. A current teacher against its own current candidates ranks100%, a self-consistency sanity check only. It does not demonstrate future identity prediction.

Training-only future5..30 target diagnostic changes10914of10955targets and yields10670correct currentcandidate ranks(97.40%). Round2 therefore tests futureidentitytargets with J3/J6/noEMA/frozenencoder atidentical architecture,8epochs/data/sampling; future observations never enter student contexts. If future targets fail, the target-time explanation alone is unsupported.

J5 introduces raw-coordinate candidateCE and sharply worsens calibration clean/actual ranks; ordinary samearchitecture CE also overfits. J6 corruption partially reduces that deficit but does not reverse it. A targeted third round must examine whether latent prediction is missing direct identity discrimination rather than increasing modelsize blindly. Complete first-round tracking is pending; no final conclusion yet.

Original synthetic-mask diagnostics compare the maskedmodel to an unmaskedhistorymean. That comparator has more evidence. Preserve historical values but do not interpret their N01/N10 as an equal-information recovery gain/loss. Add visible-only mean and report the extra-information comparator separately. Actual and clean no-mask comparisons remain valid.

All no-known-correct SN015 contexts include unknown pastlabels. Do not equate unknown with wrong, or claim recovery impossible. Causal online choices cannot read offline physical identity/GT or CLEAR labels. DenseJEPA, multi-hypothesis, and syntheticIR identity extensions remain conditional on validated core evidence.

## Second and third targeted rounds

Round2 futureJ3weight.3 HOTA52.630910, futureJ6 52.936413; noEMA53.405525 andfreeze53.499933. Futuretarget time alone does not address the first-round deficit. Raw features of the future training target remain identity informative, but its prediction is not superior online.

Round3 adds direct latent candidateCE(.5) against stoppedteacher/currentstudent candidate embeddings. A separately snapshotted trainer preserves the original resume-proof source. Same architecture/noEMA and ordinarycontrastive controls distinguish direct discrimination from EMA. Allsixmodels trained8epochs; full18tracking settings still running. FirstGRUlatentCE HOTA52.853525 at.3; samearchitecture noEMA53.842890, so no positive EMA attribution from these seed0 results. Transformerfactorial tests remain pending.

Equal-visible masking correction is now complete for11models. J3model rank52.487% versus visiblemean59.042%, N01=47,N10=765,net−718; J4net−708. Originalunmaskedmean60.329% hadextra information and must not be treated as a fair paired intervention. Correcting the comparator does not reverse the negative J3/J4 stress result. Raw legacy fields remain intact.

Training dispatcher source-validation bug: saved trainer key was absolute while new checker indexed relativepath. The initial thirdround aggregate FAILED with KeyError; allsix actual model manifests remain COMPLETE. Repair resolves the key and independently verifiesallmodels/exports in jepa_training_seed0_v3_guardrepair_v2.json, without retraining or changingweights. Originalfailurelog/manifest/source snapshots retained.

OrdinaryTransformerseed0 HOTA55.474455 is neither EMA nor recovery validation. It has poor static calibration rank but better endogenoustrack histories. Fixed-score and puregeometry controls have been frozen to check cost-rescaling/acceptance explanations; three seeds are pending. Decisionrecording reruns exactly reproduce J3/ordinaryTransformer two-video predictions before event analysis. Do not derive causal HOTA percentages from overlapping error proxies.

Allthree targeted JEPArounds now have fulltwo-video tracking:21seed0model trainings and63unique fixedsettings. Round3 TransformerJ6 at.3 givesHOTA54.402713/AssA40.4237/IDF164.3937/IDSW84; latentCE variant54.295177, ordinarycontrastive52.764480. The original nominalcapacitymatched unaugmentedordinaryTransformer seed0is55.474455. Its seed1drops to52.595827, so no bestseed or stablegain claim. Transformer J6/currentlatent, directlatentCE, ordinarycontrastive, ordinaryerror-augmentation and identicalnoEMA are now registered for three-seedformal comparisons, allfixed.3.

Identity-independent controls are complete for three seeds: constantzero at.1 HOTA52.370034; at.3 48.743272; constanthalf at.3 52.117691; pureCFPgeometry50.792426. All three seeds produce the same corresponding numbers. This excludes those particular constant-score substitutions as the seed0Transformer improvement explanation, but does not prove recoverability or eliminate every score-calibration confound. The actual query-level netcorrection deficits remain relevant.

Full strictonline4834frame J3/ordinaryTransformer seed0runs are underway. The completed64frame-per-video prefixes match the cached control bytes, but prefixFPS and cachedassociation timings are not final end-to-end speed. Fullpipeline timing includesdecode,causalIRgeneration,V3/NMS,currentRGBcrops/encoder,student/context,associationandoutput. Further seeded online/candidate experiments remain pending.

## First complete three-seed comparison

At frozenweight.3: ordinaryTransformer HOTA53.533551±1.372687; J3 EMA52.821507±.174272; J6 EMA52.863515±.527899; exactJ6 noEMA52.951765±.398978. J6EMA does not improve HOTA orIDF1 over its identical-architecture/data/mask/corruption/loss noEMA control in this three-seed comparison. J3meanIDSW80 is lower than the old ordinaryGRU87.333, but HOTA/AssA/IDF1 alldecline, so do not hide the tradeoff or attribute overallidentitysuccess to that one count. OriginalordinaryGRU current-box HOTA53.591386 remainshigher than this newordinaryTransformermean. These are development results, not newblindtest orcrossscene confidence intervals.

Remaining investigation is the thirdround Transformer factorial: its seed0J6 andlatentCE variants exceed54 but three-seed results/noEMA contrast are pending. Further fullonline timings are running; no FINAL_EVALUATED orvalidatedcore status yet.
