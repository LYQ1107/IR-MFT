# Prior-art boundaries for the correlation hypothesis

No new mechanism in this campaign is an established contribution yet.
Every claim still requires same-capacity RGB/gray controls, module removals,
and complete-sequence official metrics on the fixed setting. The user's latest
directive fixes future work to seed0 and cancels all further seed replication;
already completed multi-seed evidence is retained, with no future seed-robustness claim.

[OGM-GE, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html)
studies imbalanced multimodal optimization and modulates gradients to improve
learning of a weaker modality. Its audio-visual findings motivate checking
optimization imbalance; they do not diagnose this fish detector's failure.

[On-the-fly Modulation, 2024](https://arxiv.org/abs/2410.11582)
includes modulation that drops dominant-modality features. Ordinary modality
dropout or weakening RGB therefore cannot be claimed as a new contribution.

[Deformable DETR](https://arxiv.org/abs/2010.04159)
already uses sparse attention near reference locations. The current fixed
5x5 cosine readout is a simpler distinct implementation, but locality alone
is insufficient novelty.

The task-specific hypotheses concern recoverability of actual generated IR,
protection of a strong RGB observation graph, and class-valid counterfactual
training when two frames both contain foreground. Each remains provisional.
The pseudo-IR is generated from RGB: any improvement is evidence about an
additional representation, not evidence of independent physical sensing.

[ViPT, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Visual_Prompt_Multi-Modal_Tracking_CVPR_2023_paper.html)
already adapts a frozen RGB foundation tracker through a small number of
modality-relevant prompt parameters. Freezing the RGB model and training
only a small multimodal adapter is therefore not an independent novelty claim.

[SDSTrack, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_SDSTrack_Self-Distillation_Symmetric_Adapter_Learning_for_Multi-Modal_Visual_Object_Tracking_CVPR_2024_paper.html)
uses symmetric multimodal adaptation and complementary masked-patch
self-distillation to address RGB dominance and degraded or missing inputs.
Ordinary masking plus teacher preservation is already related work. The
current hypothesis instead tests generated-IR recoverability-conditioned
target placement and class-validity of mismatched foreground; their value
must be demonstrated beyond matched-count ordinary weakening and explicit
removals. This distinction remains a hypothesis, not an assertion of originality.

These visual-object-tracking studies use their own datasets and metrics;
their reported numbers are not substitutes for HOTA on this fish MOT protocol.
The searches are a focused prior-art check, not an exhaustive novelty proof.

## Additional verified boundaries for the semantic/geometry diagnosis

The static choice to use RGB localization and fused classification has prior
art outside tracking. [Learning Rich Features for Image Manipulation Detection,
CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf)
fuses RGB/noise evidence for classification and uses RGB for box regression.
Copying RGB coordinates alone therefore earns no standalone novelty credit.

[Efficient RGB-T Tracking via Cross-Modality Distillation, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Efficient_RGB-T_Tracking_via_Cross-Modality_Distillation_CVPR_2023_paper.pdf)
already transfers multimodal features and response supervision. Ordinary
freezing, distillation or separating classification/regression is insufficient.

[PMATrack, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jin_Progressive_Multi-cue_Alignment_for_Unaligned_RGBT_Tracking_CVPR_2026_paper.html)
progressively estimates unaligned RGB/TIR geometry and selects alignment cues.
Local offset handling cannot be called new merely because this data is synthetic.

[Cross-Modal Ranking with Soft Consistency and Noisy Labels, ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Chenglong_Li_Cross-Modal_Ranking_with_ECCV_2018_paper.html)
already treats imperfect cross-modal consistency and noisy foreground queries.
Generic background suppression or consistency weighting is prior art.

[ESMTrack, arXiv2026](https://arxiv.org/abs/2609.37162)
uses unimodal/fused branches, reliability estimates and training-time modality
decoupling; [Self-Supervised RGB-T Tracking with Cross-Input Consistency,
2023](https://arxiv.org/abs/2301.11274) already studies reweighted consistency.
Dominant-modality mitigation alone is not a new contribution.

[Synthetic RGB-T tracking, ICCVW2019](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Zhang_Multi-Modal_Fusion_for_End-to-End_RGB-T_Tracking_ICCVW_2019_paper.html)
uses translated paired TIR data. Synthetic auxiliary data itself is not a first.
These papers address other tasks/protocols; their published single-object scores
are not interchangeable with HOTA on the current seven MFT Test sequences.

The current50.515 seed0 Dev result is an empirical candidate, with mixed
per-video effects, not an established original algorithm. Required evidence
remains actual-versus-unavailable/misaligned IR, strongest matched RGB/gray and
ordinary confidence-calibration controls, original-parent module removals,
the complete frozen Test. The user's later directive cancels further seed
replication; future experiments stay at seed0. This is a targeted related
work check, not an exhaustive proof that any future mechanism is original.

## Classical structural correspondence diagnostic

[MIND, Medical Image Analysis2012](https://pubmed.ncbi.nlm.nih.gov/22722056/)
compares local same-image patch self-similarities to represent structure across
modalities. [Self-similarity context, MICCAI2013](https://iacl.ece.jhu.edu/proceedings/miccai2013/papers/8149/81490187.pdf)
instead compares neighboring patches without relying on the central patch and
uses inverse-consistent multimodal registration. These are classical published
methods, not new descriptors or tracking innovations from this campaign.

The new small2D8-neighbor Train-only diagnostic borrows the patch-SSD/local
variance idea. It does not reproduce the full published3D registration method.
Known synthetic grayscale displacements test the implementation; current versus
wrong actualIR tests structural discrimination. Bounding-box foreground labels
do not supply true RGB/IR pixel correspondences, so even a positive diagnostic
cannot establish corrected fish geometry or a HOTA contribution. All24 fields
are frozen before GT is opened; no full cache or new fit is activated automatically.

## Relative-response correction is a hypothesis, not a new subtraction rule

[Counterfactual VQA, CVPR2021](https://arxiv.org/abs/2006.04315)
already subtracts a direct language effect from the total effect to reduce
language bias. [Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
contrasts output distributions under original and distorted images to reduce
object hallucinations without training. These are different tasks, but they
establish that counterfactual subtraction and original-versus-distorted output
contrasts are existing ideas.

## Direct multimodal MOT references checked on2026-11-23

[PFTrack / VT-MOT](https://arxiv.org/abs/2408.00969) already performs progressive
temporal and cross-modal fusion for multiple-object tracking. Its temporal
attention does not assume strict adjacent-frame spatial alignment. Generic
temporal attention plus complementary fusion is therefore established prior art.

[MCTrack publisher abstract](https://ieeexplore.ieee.org/abstract/document/11175188/)
describes temporal salient interaction, bidirectional cross-modality interaction
and complementary masking. The [official repository](https://github.com/ydhcg-BoBo/MCTrack)
identifies the released implementation and its dependence on PFTrack. The older
RGB-T-MOT repository link redirects there. Only the publisher's indexed abstract
and the official README were verified here; a full-paper fetch was unavailable.

[DM3T, CVPR2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_DM3T_Harmonizing_Modalities_via_Diffusion_for_Multi-Object_Tracking_CVPRF_2026_paper.html)
already uses diffusion-inspired iterative cross-modal feature harmonization,
a feature refiner and hierarchical confidence handling. Its
[author preprint](https://arxiv.org/abs/2511.22896) is another primary source.
It addresses visible/thermal MOT; merely applying diffusion or confidence-based
hierarchy cannot be claimed as a new multimodal tracking contribution here.

These are relevant algorithmic comparisons, not MFT reproduction results.
Published VT-MOT HOTA cannot be copied into the current seven-sequence fish table.
No checkpoint, image dataset or new training environment was downloaded during
this literature check. The present experiment continues with actual November
RGB-derived pseudo-IR, matched RGB/gray controls and fixed seed0. It tests
representation-level benefit, not an independent physical thermal sensor.

The structural-relative model's current-versus-shifted auxiliary logit difference
therefore receives no novelty credit by itself. The provisional task-specific
question is whether structural correspondence validation and a bounded semantic
correction at unchanged RGB geometry produce a measurable auxiliary-content
benefit beyond matched RGB/gray and module removals. No positive answer or
exhaustive originality claim is assumed. All new fits use the fixed seed0.
