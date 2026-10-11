# PLIM experiments

Status: NOT_STARTED. No predictive identity model has been trained or evaluated. Initial lightweight GRU/causal Transformer, history lengths 4/8/16, frozen future target and cosine/ranking losses are prospective.

Compare clean GT histories and actual online histories separately, plus last/EMA/FIFO, ordinary GRU, capacity-matched Transformer, motion prediction and RGB/IR variants. Check collapse/feature diversity, gaps and post-crossing ID recovery on training/dev. Prediction benefit and safe-write benefit require separate toggles and ablations. All scoring uses history before current writes. Do not call generic temporal averaging a successful PLIM experiment.
