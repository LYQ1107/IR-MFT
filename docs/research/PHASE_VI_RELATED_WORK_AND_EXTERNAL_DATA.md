# Phase VI primary-source research and external-data screen

Checked 2026-10-13. These are mechanism comparisons and eligibility checks, not novelty or independent validation claims. No model/dataset download was performed.

| Primary source | What needs controlling in our research |
|---|---|
| [MOTIP, CVPR 2025](https://github.com/MCG-NJU/MOTIP) | History-context identity prediction already exists. Candidate-JEPA must beat a matched supervised ID decoder; its gain cannot be inferred from adding attention. |
| [GOT-JEPA](https://github.com/chenshihfang/GOT) | Predictive tracking and occlusion reasoning already have an implementation. Track-state candidate relations require separate task and target definitions. |
| [V-JEPA 2 / 2.1](https://github.com/facebookresearch/vjepa2) | EMA prediction is established machinery. We must isolate its contribution against equal-input, equal-budget ordinary models. No large video model is trained here. |
| [Hybrid-SORT, AAAI 2024](https://github.com/ymzis69/HybridSORT) | Weak geometric/confidence cues already support online association. Risk features need ordinary cue and threshold controls. |
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | Preserve the original appearance/motion baseline; candidate recalls and lifecycle changes cannot be attributed solely to learning. |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | Use the pinned existing evaluator, frozen GT and complete metric dictionaries. |
| [UTrack preprint, 2024](https://arxiv.org/abs/2408.17098) | Detection uncertainty and uncertainty-aware association are existing directions; selective risk needs explicit discrimination/calibration and final assignment evidence. |
| [DualTTrack, 2026](https://doi.org/10.1016/j.ipm.2026.104843) | The primary publisher describes persistent ambiguous trajectory evidence and recovery. Our K=2 experiment must distinguish branch-specific memories from ordinary delayed writes or potential trajectories. Full code fidelity has not been established. |
| [Conditional motion hypotheses and contrastive prototypes, 2026](https://www.nature.com/articles/s41598-026-50731-8) | The primary article combines motion ambiguity and appearance persistence. Combining these concepts alone does not establish novelty. No faithful benchmark was run. |

External fish tracking candidates:

| Candidate / primary source | Checked facts | Eligibility still unresolved |
|---|---|---|
| [FishTrack23, WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/papers/Dawkins_FishTrack23_An_Ensemble_Underwater_Dataset_for_Multi-Object_Tracking_WACV_2024_paper.pdf) | Optical-spectrum underwater subsets, RGB/grayscale variation, tracking CSV/COCO variants; annotations may be 5/10 Hz rather than native image cadence. | Dataset/subset-specific license, exact accessible sample sizes, dense sequence time mapping, physical-ID definition and overlap with MFT26 require verification. The paper being public is not a dataset license. |
| [Fraunhofer fish tracking dataset](https://cvg.hhi.fraunhofer.de/FishTracking.htm) | Institution page lists two shoal videos, boxes, masks and consistent track IDs. | Full license and download-size/overlap metadata remain unverified; subsequent page fetch failed. |
| [FISHTRAC author repository](https://github.com/tmandel/fish-detrac) | Fish tracking benchmark code and dataset links are available. | Software licenses differ by integrated tracker and do not establish the data license. Data permissions/size/overlap must be checked independently. |
| [SFISHTRACK author repository](https://github.com/JosepSanchezCano/SFISHTRACK/) | README describes natural marine video, COCO annotations, temporal identities and a Drive archive. | No explicit dataset permission/overlap confirmation or measured archive size obtained. No archive downloaded. |

Inference from this screen: there are plausible independent optical fish tracking resources, but none has yet passed the project's licensing, overlap, identity and cadence audit. No new blind validation is claimed. Continue small metadata/license checks before committing storage to acquisition. Frozen detector and identity encoder exposures within MFT26 remain disclosed even if a later association module is video-heldout.

Further primary-source check: [FISHTRAC author README](https://github.com/tmandel/fish-detrac#running-trackers-on-datasets) states a non-harmful-use condition for data, describes three public training videos and an11-video test split whose download link is still marked forthcoming. Harness CC BY-NC-SA is separate from data terms. Public archive access timed out and size/overlap was not established. [SFISHTRACK README](https://github.com/JosepSanchezCano/SFISHTRACK) still supplies a Drive link but no explicit license text found. Fraunhofer page fetch timed out again. No data archive was downloaded and no independent final dataset has passed eligibility.
