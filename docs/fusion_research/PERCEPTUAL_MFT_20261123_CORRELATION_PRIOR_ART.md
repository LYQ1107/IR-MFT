# Prior-art boundaries for the correlation hypothesis

No new mechanism in this campaign is an established contribution yet.
Every claim still requires same-capacity RGB/gray controls, module removals,
multiple seeds, and complete-sequence official metrics on the fixed setting.

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
