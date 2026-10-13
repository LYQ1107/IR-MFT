# Phase VII object-centric Dense JEPA experiments

Status: IN_PROGRESS. No local-token JEPA model has yet been trained; inherited512-dimensional PLIM is not renamed DenseJEPA.

A0 shows why prediction-loss optimization alone is insufficient: frozen GRU pure retrieval loses to retained global features on clean SN-015 histories, while fixed native geometry scores hide much of that difference. Joint probes also show that apparent bounded gains can be reproduced by a constant cost edit. A future DenseJEPA result must add identity evidence under matched current candidate support, not merely lower latent error or move the acceptance threshold.

The required implementation uses current object crop local/spatial tokens from verified prior/current observations, visible past-only online context and an EMA target encoder used only during training. Future training target tokens may be stop-gradient labels but cannot enter history, online memory, masks, reliability or decisions. Masked target tokens cannot leak through teacher pooling or context construction. Target masks must correspond to annotated/observable validity; missing physical identity is not a negative.

Matched local TemporalTransformer, contrastive temporal learning, noEMA/noPrediction/noID and simple local aggregation controls remain required. Input coordinate, producer, crop budget, candidate pool, training video split and parameter/compute budget must be recorded. Paired both-crop feature foundation, complete I-JEPA/V-JEPA2/GOT source audits, lawful small weights and independent identity retrieval are pending before large training. No dense predictive novelty or performance claim is retained at this stage.

## I-JEPA source audit completed

Exactcommit52c1ae95d05f743e000e8f10a1f3a79b10cff048 has16verifiedsource/config/license files andall11fields inREPOSITORY_REGISTRY. Code is [CCBYNC4](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE), notMIT/Apache; no officialcodecopied/executed orhugeweightsdownloaded. Staticimagecontext/maskedregionSmoothL1/stoppedEMA doesnot providefishidentitysupervision. Actualmasker relaxesdisjointness afterretries andrectangularpositioninterpolation isunfit for128x256; independentlyimplemented DenseJEPA musttesttargetexclusion/current-futurecausality andrectangularlocalshapes explicitly. This audit preparesB; no actuallocalDenseJEPAtraining resultis claimed.
