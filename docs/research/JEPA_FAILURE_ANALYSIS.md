# Fish-JEPA failure analysis — in progress

J3 has a genuine trained EMA teacher, but first complete seed0 tracking atweight.3 HOTA52.631715 is below the existing ordinaryGRU current-box control53.104. J4 masking produces52.631763, an essentially identical trajectory-level result. Training latent cosine loss declines, which does not validate identity recovery.

J3 SN015 clean rank99.205% versus mean99.498%; actual61.543% versusmean62.291%, N01=51/N10=133/net−82. Teacher target variance sum.227533 and mean norm.878901 show substantial shared direction, but not exact constant collapse. Student actual variance.206632 also nonzero. A current teacher against its own current candidates ranks100%, a self-consistency sanity check only. It does not demonstrate future identity prediction.

Training-only future5..30 target diagnostic changes10914of10955targets and yields10670correct currentcandidate ranks(97.40%). Round2 therefore tests futureidentitytargets with J3/J6/noEMA/frozenencoder atidentical architecture,8epochs/data/sampling; future observations never enter student contexts. If future targets fail, the target-time explanation alone is unsupported.

J5 introduces raw-coordinate candidateCE and sharply worsens calibration clean/actual ranks; ordinary samearchitecture CE also overfits. J6 corruption partially reduces that deficit but does not reverse it. A targeted third round must examine whether latent prediction is missing direct identity discrimination rather than increasing modelsize blindly. Complete first-round tracking is pending; no final conclusion yet.

Original synthetic-mask diagnostics compare the maskedmodel to an unmaskedhistorymean. That comparator has more evidence. Preserve historical values but do not interpret their N01/N10 as an equal-information recovery gain/loss. Add visible-only mean and report the extra-information comparator separately. Actual and clean no-mask comparisons remain valid.

All no-known-correct SN015 contexts include unknown pastlabels. Do not equate unknown with wrong, or claim recovery impossible. Causal online choices cannot read offline physical identity/GT or CLEAR labels. DenseJEPA, multi-hypothesis, and syntheticIR identity extensions remain conditional on validated core evidence.
