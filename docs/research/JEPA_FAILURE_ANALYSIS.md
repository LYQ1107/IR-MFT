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
