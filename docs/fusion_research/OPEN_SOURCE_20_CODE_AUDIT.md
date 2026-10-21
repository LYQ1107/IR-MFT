# Open-source multimodal fusion code audit (20 pinned repositories)
This is a source-level audit at the exact commits in the task book. The files below were fetched from the pinned raw GitHub paths on 2026-10-21; the recorded bytes/SHA-256 and symbol lines are the evidence used for the design. No third-party source was copied into IR-MFT. A missing root license is recorded as a boundary, not inferred.
## Fixed source inventory
| ID | Repository / commit | License evidence | Pinned source paths and observed interfaces |
|---|---|---|---|
| R01 | [PFTrack](https://github.com/wqw123wqw/PFTrack) @ `73d4dc4c5e69e530f7a77d974bff7d2040720728` | MIT | `src/lib/model/networks/transformer_fusion_v3_vt.py` (OK, 5290 B, 6b4a38cbab37feb138d122f5ac6d2a621c69b03d2088c5c92bfd271c5cc62699) — class Transformer_Fusion(nn.Module):; def __init__(self, d_model=64, nhead=8, num_fusion_encoder_layers=2, dim_feedforward=2048,; def _reset_parameters(self):; def with_pos_embed(self, tensor, pos):; def forward(self, src_v,src_t, pos_embed):<br>`src/lib/model/networks/dla_fusion_mid.py` (OK, 22800 B, cde8f9f32bd14f5695bb2386518611a85a22d79e05cc2c2f64a30a47b7010242) — def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):; def conv3x3(in_planes, out_planes, stride=1):; class BasicBlock(nn.Module):; def __init__(self, inplanes, planes, stride=1, dilation=1):; def forward(self, x, residual=None):<br>`src/lib/model/networks/generic_network.py` (OK, 3961 B, 5a65b39c767b85559db771635d5b4dc4470fd6d37c02253937d9b4ad7cd28f8d) — def fill_fc_weights(layers):; class GenericNetwork(nn.Module):; def __init__(self, num_layers, heads, head_convs, num_stacks=1, opt=None):; def forward(self, x, pre_img=None, pre_hm=None): |
| R02 | [DM3T](https://github.com/vranlee/DM-3-T) @ `662790ba213b4460f19a1bdbefb9714416eb2b5c` | Apache-2.0 | `src/lib/model/networks/tf_fusion_multimodal.py` (OK, 5292 B, da2e3d55480efe1bf6dfca01f23ceb218e0a0a48ec9326e934ed6753041ed715) — class Transformer_Fusion(nn.Module):; def __init__(self, d_model=64, nhead=8, num_fusion_encoder_layers=2, dim_feedforward=2048,; def _reset_parameters(self):; def with_pos_embed(self, tensor, pos):; def forward(self, src_v,src_t, pos_embed):<br>`src/lib/model/networks/tf_fusion_refined.py` (OK, 7843 B, d9288a174ad6fcbbf6715b2da1d18c880584117de07ffa2e845a5e5ea59f9d39) — class Transformer_Fusion(nn.Module):; def __init__(self, d_model=64, nhead=8, num_fusion_encoder_layers=2, dim_feedforward=2048,; def _reset_parameters(self):; def with_pos_embed(self, tensor, pos):; def forward(self, src_v,src_t, pos_embed):<br>`src/lib/model/networks/dla_rgbt_main.py` (OK, 30912 B, ee9d32c3ee12c363fcf418d615bc151bb37e183596734806420322cd250997e5) — class CrossModalDiffusionFusion(nn.Module):; def __init__(self, channels, steps=3, noise_level=0.01):; def _make_denoise_net(self, in_channels, out_channels):; def forward(self, x_rgb, x_t):; def save_feature_map(tensor, name, output_dir='feature_maps_output'):<br>`src/lib/model/networks/generic_network.py` (OK, 3961 B, 5a65b39c767b85559db771635d5b4dc4470fd6d37c02253937d9b4ad7cd28f8d) — def fill_fc_weights(layers):; class GenericNetwork(nn.Module):; def __init__(self, num_layers, heads, head_convs, num_stacks=1, opt=None):; def forward(self, x, pre_img=None, pre_hm=None): |
| R03 | [SMSTracker](https://github.com/Leezed525/SMSTracker) @ `ab85a8a6e6f1793a0d0ac20c268aae687f1abcab` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `lib/models/backbone/vit_sigma.py` (OK, 12953 B, 4d2f67189c0b2080d2192a15439ef2f8ced3111ef9b6999ddcc4880a0c9cc1c4) — class Attention(nn.Module):; def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):; def forward(self, x, return_attention=False):; class Block(nn.Module):; def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,<br>`lib/models/layer/score.py` (OK, 3873 B, 2ac149e9f4127a0cf2cd32320af3ff02b22583d6a371935bc70c0934e691c33b) — class ScoreLayerUseConv(nn.Module):; def __init__(self, threshold1=0.33, threshold2=0.66, embed_dim=768, num_heads=12, ffn_dim=96, initial_value=1, heads_range=3):; def forward(self, x):; class PLScoreLayerUseConv(nn.Module):; def __init__(self, threshold1=0.33, threshold2=0.66, embed_dim=768):<br>`lib/models/SMSTracker/score_os_sigma_center.py` (OK, 4511 B, 11c068b30a97a93c0cc7254fa8c2fd3975d6b547c00cd00380be83ed0e1ba1f5) — class ScoreOSCENTER(nn.Module):; def __init__(self, backbone, box_head, cfg: CfgLoader = None, head_type='CENTER'):; def forward(self, template: torch.Tensor,; def forward_head(self, cat_feature, gt_score_map=None):; def build_score_os_sigma_center(cfg, training=True): |
| R04 | [FlexTrackV2](https://github.com/supertyd/FlexTrackV2) @ `30a5ff1b39b8f3004dde4018018b901e6cc1bf54` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `lib/models/flextrackv2/moe_fusion.py` (OK, 28952 B, 8c16213f24bb8524774210f04e5d243a561d23b5b05af504248e31efd2b1309f) — class adapter(nn.Module):; def __init__(self, embed_dim=512, hidden_dim=32):; def forward(self, x):; class adapter_down(nn.Module):; def __init__(self, embed_dim=768, hidden_dim=32):<br>`lib/models/flextrackv2/encoder.py` (OK, 3105 B, a95d1f4ac25ed41720696477ff1c01bb0e84ce8b04a3bda6fb7439017d573091) — class EncoderBase(nn.Module):; def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):; def forward(self, template_list, search_list, template_anno_list):; class Encoder(EncoderBase):; def __init__(self, name: str,<br>`lib/models/flextrackv2/flextrackv2.py` (OK, 23081 B, a421d5887282a7c1424b5d2b22dbd2a86fd69502bc3391ed4886335d9f335338) — def apply_mask(tensor1, tensor2, mask_list, rgb_prompt=None, aux_prompt=None):; def random_mask(tensor, mask_prob, num_tokens_to_mask):; class LinearAttention_moe(nn.Module):; def __init__(self, embed_size, hidden_size, patch_num_x, patch_num_z, moe_type=None, moe_cfg=None):; def forward(self, x, missing_mask=None): |
| R05 | [XTrack](https://github.com/supertyd/XTrack) @ `8a606f00c5e98b5393254f6ee1891447b662407f` | MIT | `lib/models/layers/MoE.py` (OK, 17990 B, 868cbbad5d08c2a6eba6b2e488e9f59a74c03322664d871299d63bfe8c10bf14) — class PatchEmbed(nn.Module):; def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, norm_layer=None, flatten=True):; def forward(self, x):; class PatchEmbed_1(nn.Module):; def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):<br>`lib/models/layers/moe_lora.py` (OK, 17272 B, e1623efb1a433e6166650d0a9365d33ef8de758aecbb5efd19b3dd3e8d430ee8) — class SparseDispatcher(object):; def __init__(self, num_experts, gates):; def dispatch(self, inp):; def combine(self, expert_out, top_gedits, multiply_by_gates=True):; def expert_to_gates(self):<br>`lib/models/xtrack/ostrack_adapter.py` (OK, 7286 B, 45a5b3061c3dcda26df1ccbfdbfc04e8c4c87e66bede61ff1623e75c14c2536f) — class XTrack(nn.Module):; def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):; def forward(self, template: torch.Tensor,; def forward_head(self, cat_feature, gt_score_map=None):; def build_xtrack(cfg, training=True): |
| R06 | [MOSSTrack](https://github.com/lyssyl157/MOSSTrack) @ `95f33ab2c927c396cad9bdb8ee213fa04a49e0a4` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `lib/models/mosstrack/fusion_layer.py` (OK, 2395 B, 2e96259e7b19d825cfa438c8991608d7584edf4a547e1971a9809fb4623ae8db) — class FusionLayer(nn.Module):; def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,; def forward(self, x_v, x_i, temporal_query_RGB, temporal_query_TIR, lens_z):<br>`lib/models/mosstrack/mosstrack.py` (OK, 6915 B, 64bee88cb51edfefc440f13a5a7926c03ede4b8998426ad518043819a1d173e6) — class MoSSTrack(nn.Module):; def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):; def forward(self, template_list: torch.Tensor,; def forward_head(self, opt, gt_score_map=None):; class GradCAMWrapper(nn.Module): |
| R07 | [CADTrack](https://github.com/IdolLab/CADTrack) @ `93d0a67d3eb328617751b18510bdab072705c1a0` | MIT | `lib/models/layers/CDA_Module.py` (OK, 19729 B, 98b4709934a946b57aa11f0aea7719378c572ea7fa2aa58ead9a899c85504628) — class DAttentionBaseline(nn.Module):; def __init__(; def _init_weights(self, m):; def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):; def off_set_shared(self, data, reference):<br>`lib/models/layers/cross_layer.py` (OK, 3969 B, 5008a012c53aaa0f5b7bb567b95b6118cef138644be9955e566cdf8c298d114e) — def candidate_elimination(tokens: torch.Tensor, attn: torch.Tensor, template_mask: torch.Tensor, lens_q: int, lens_t: int, keep_ratio: float):; class Up_Down(nn.Module):; def __init__(self, dim, xavier_init=False):; def forward(self, x): |
| R08 | [PFGF](https://github.com/liting1018/PFGF) @ `df2a3c03cc217204ae26f6d0afb8b0e0a628ec3f` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `PFGF-full/mmdet/models/necks/graphmamba_utils/tir_graph.py` (OK, 6080 B, 1bb1353af5c1d0c45f7cb8c887da4ab180fa2e807e7168bba296d4151c93fd2a) — class PPM(nn.Module):; def __init__(self, chnn_in, rd_sc, dila):; def forward(self, inputs):; class GraphReasoning(nn.Module):; def __init__(self, chnn_in, rd_sc, dila, n_iter, num_inter_graphmambas, num_intra_graphmambas):<br>`PFGF-full/mmdet/models/necks/yolox_pafpn_tirgraphmamba.py` (OK, 6543 B, 80449f4919a4881d63b8eb25139f17fe6198512568bef0548b713848982153fa) — class YOLOXPAFPN_TIRGraphMamba(BaseModule):; def __init__(self,; def forward(self, inputs): |
| R09 | [M2D-LIF](https://github.com/Zhao-Tian-yi/M2D-LIF) @ `210c7ca22b7670e7d4629cca995427bf7e74653f` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `ultralytics/nn/modules/block.py` (OK, 21694 B, bba7913200b3201c569baeefc97395205de5ac462252c68629e3be7cac10a148) — class LIF(nn.Module):; def __init__(self):; def forward(self, x):; class LIFAdd(nn.Module):; def __init__(self, layer):<br>`model_yaml/yolov8-LIF.yaml` (OK, 2456 B, 05456d29cdd507fc9f5b7ff43e4d00c0541b1214e5845f4dfa8141a503cd7877) —  |
| R10 | [ModTr](https://github.com/heitorrapela/ModTr) @ `cace11eeb51f1421149ff4740f14e3b38f03d58c` | MIT | `src/engine/utils/fusion_strategy.py` (OK, 1366 B, d51cb53200819e9e790c95e32d847545656861af1f6077b4e7f566bf9a2b3cbd) — def addition_fusion(tensor1, tensor2):; def attention_fusion_weight(tensor1, tensor2):; def spatial_fusion(tensor1, tensor2, spatial_type='sum'):; def spatial_attention(tensor, spatial_type='sum'): |
| R11 | [SDSTrack](https://github.com/hoqolo/SDSTrack) @ `822d9852fee66bdd6d9df4dcb679eebecd85e91c` | MIT | `lib/models/sdstrack/vit_ce.py` (OK, 15766 B, c8c5b8b719ec3b28208fec6a85db5d31b82ff550dea80e06b3a7e38ff7bd399b) — class VisionTransformerCE(VisionTransformer):; def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,; def forward_features(self, z, x, mask_z=None, mask_x=None,; def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,; def _create_vision_transformer(pretrained=False, **kwargs):<br>`lib/models/sdstrack/sdstrack.py` (OK, 6203 B, b40d631db4bd05fc6bf60ed3cfe7b92c03e398907a57acf2e3df25e3ceebbc73) — class SDSTrack(nn.Module):; def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):; def forward(self, template: torch.Tensor,; def forward_head(self, cat_feature, gt_score_map=None):; def build_sdstrack(cfg, training=True): |
| R12 | [SUTrack](https://github.com/chenxin-dlut/SUTrack) @ `d65052d1ba3fcf55010e1fb3665ee6616c139a2c` | MIT | `lib/models/sutrack/encoder.py` (OK, 3833 B, d89cf7c8646b2e862b6f2e758faa382f7dbcbcf3ef996e496e84a612de4ceda5) — class EncoderBase(nn.Module):; def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):; def forward(self, template_list, search_list, template_anno_list, text_src, task_index):; class Encoder(EncoderBase):; def __init__(self, name: str,<br>`lib/models/sutrack/sutrack.py` (OK, 5320 B, 2d88cd3473eb7f3265086f9fba36fd8f19ccee4c58b3e9694bf036c98ee19834) — class SUTRACK(nn.Module):; def __init__(self, text_encoder, encoder, decoder, task_decoder,; def forward(self, text_data=None,; def forward_textencoder(self, text_data):; def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index): |
| R13 | [UnTrack](https://github.com/Zongwei97/UnTrack) @ `8eec76ec912c19e326e2b0020444f8f20c7d4355` | ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only. | `lib/models/layers/MoE_prompt.py` (OK, 5149 B, 355e3158427dd0c9abaeed7937e53d35748952e966f8773d7b34f19a0c1f48f8) — class Expert(nn.Module):; def __init__(self):; def forward(self, x):; class Router(nn.Module):; def __init__(self, num_experts):<br>`lib/models/untrack/vit_prompt.py` (OK, 12310 B, 68e1de540b1342aa6658a7eec13169b59bc7b8ba14d2f72a68496e214cce43a4) — class Attention(nn.Module):; def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):; def forward(self, x, return_attention=False):; class Block(nn.Module):; def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., |
| R14 | [GOLA](https://github.com/MelanTech/GOLA) @ `339c737cda6a24be667b6e5abdc721e8d6046f05` | Apache-2.0 | `trackit/models/methods/GOLA/modules/gola/apply.py` (OK, 1206 B, be74eca916c50bb4e60ce89a68a28a6f05bae57a6573140cb0f8b39b302858c9) — def find_all_frozen_nn_linear_names(model):; def apply_lora(model: nn.Module, lora_module_names: List[str],<br>`trackit/models/methods/GOLA/modules/lora/apply.py` (OK, 1206 B, fc1093e27a7086a9af3c07119a754f14f6badc402a0f7bc6366c6f7a36479a29) — def find_all_frozen_nn_linear_names(model):; def apply_lora(model: nn.Module, lora_module_names: List[str],<br>`trackit/models/methods/GOLA/gola.py` (OK, 6337 B, 62c15e8482b8ae61d2b6a6fa4ed73d77cec8a03613f08e4e3ed07efdfeadc311) — class GOLA_DINOv2(nn.Module):; def __init__(self, vit: DinoVisionTransformer,; def forward(self, z: torch.Tensor, x: torch.Tensor, d: torch.Tensor,; def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):; def _d_feat(self, d: torch.Tensor, d_feat_mask: torch.Tensor): |
| R15 | [PURA](https://github.com/MelanTech/PURA) @ `428c60ab4e8e1c692967d95f0c55b79c940efa3f` | MIT | `lib/test/tracker/tta/pura.py` (OK, 8076 B, 865d0860e6b9e1835da439e3e45bcbc4a135fb371a5b592f8862e0d8d0b8f242) — class ParamDict(OrderedDict):; def __init__(self, *args, **kwargs):; def _prototype(self, other, op):; def __add__(self, other):; def __rmul__(self, other): |
| R16 | [MOTIP](https://github.com/MCG-NJU/MOTIP) @ `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` | Apache-2.0 | `models/motip/id_decoder.py` (OK, 11908 B, a16cf34bb16fa14a17d715231af1439cb3f1c1fa15d18341e2a84c61cdf59d81) — class IDDecoder(nn.Module):; def __init__(; def forward(self, seq_info, use_decoder_checkpoint):; def _forward_a_layer(; def id_label_to_embed(self, id_labels):<br>`models/motip/trajectory_modeling.py` (OK, 1403 B, 0bdf92640347ec2b719ac56082aec307eeb6c3fd39c1cc87ea26e197eca58166) — class TrajectoryModeling(nn.Module):; def __init__(; def forward(self, seq_info): |
| R17 | [TrackTrack](https://github.com/kamkyu94/TrackTrack) @ `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` | MIT | `3. Tracker/trackers/tracker.py` (OK, 5594 B, b3289f19060502d31dccebf88a2c98986afb61e1be8c3f13258726c34a725f62) — Tracker.init_tracks; Tracker.update (verified URL-encoded path) |
| R18 | [HybridSORT](https://github.com/ymzis69/HybridSORT) @ `396f8d30db13304c0cbaf1dcf2e16ded93ce1701` | MIT | `trackers/hybrid_sort_tracker/hybrid_sort.py` (OK, 24461 B, b6fdd1b1a2f5fe9fa234e82ef09f45d3d3d66367bec27b97153335295ca32a35) — def k_previous_obs(observations, cur_age, k):; def convert_bbox_to_z(bbox):; def convert_x_to_bbox(x, score=None):; def speed_direction(bbox1, bbox2):; def speed_direction_lt(bbox1, bbox2):<br>`trackers/hybrid_sort_tracker/association.py` (OK, 31610 B, 696c195f805dc1dee2bafa670820cb297ffbea9135a9dca1b6428c4f1ea24bd8) — def intersection_batch(bboxes1, bboxes2):; def box_area(bbox):; def iou_batch(bboxes1, bboxes2):; def cal_score_dif_batch(bboxes1, bboxes2):; def cal_score_dif_batch_two_score(bboxes1, bboxes2): |
| R19 | [BoT-SORT](https://github.com/NirAharon/BoT-SORT) @ `251985436d6712aaf682aaaf5f71edb4987224bd` | MIT | `tracker/bot_sort.py` (OK, 16778 B, ba6dd32ad4b96ee49d6f70654159269bd45ce15fd0fe5007f3a1b81bc257edf0) — class STrack(BaseTrack):; def __init__(self, tlwh, score, feat=None, feat_history=50):; def update_features(self, feat):; def predict(self):; def multi_predict(stracks):<br>`tracker/matching.py` (OK, 6435 B, 020b2215c20467b31f7d8a977a2f557b8d1838f8baa5a5be54c0b8d14bbee4e1) — def merge_matches(m1, m2, shape):; def _indices_to_matches(cost_matrix, indices, thresh):; def linear_assignment(cost_matrix, thresh):; def ious(atlbrs, btlbrs):; def tlbr_expand(tlbr, scale=1.2): |
| R20 | [OC-SORT](https://github.com/noahcao/OC_SORT) @ `8462e7e729a93ccd3bd995c0a79a890336cb3a0b` | MIT | `trackers/ocsort_tracker/association.py` (OK, 13384 B, c009a78eaa1c35c1d61217506255051ba882a7b99f9505cdec64da474ff0fbd7) — def iou_batch(bboxes1, bboxes2):; def giou_batch(bboxes1, bboxes2):; def diou_batch(bboxes1, bboxes2):; def ciou_batch(bboxes1, bboxes2):; def ct_dist(bboxes1, bboxes2): |

## Interpretation, complexity and IR-MFT decision
The exact per-repository record above captures the callable class/forward surfaces. The tensor shapes below are the source-level contract used for the migration; where a repository does not state a shape in annotations, the shape is marked as inferred from its reshape/conv/attention operations rather than presented as a measured runtime result. Parameter/FLOP numbers are not claimed unless the source declares them; the new IR-MFT modules will report their own parameter and runtime measurements.
### R01 — PFTrack

- **Observed interface / cost boundary:** Token-level two-stream Transformer_Fusion; forward(src_v, src_t, pos_embed) returns cross-modal encoder output. The fusion is progressive/attention-based and costs several encoder layers. Adopt only as a source-level comparison; do not transplant the detector.
- **IR-MFT migration decision:** The closest baseline for ordinary cross-modal attention. Our F1/F2 keeps FPN geometry and adds causal quality/complementarity gates.
- **License/reuse boundary:** MIT
### R02 — DM3T

- **Observed interface / cost boundary:** Transformer fusion plus CrossModalDiffusionFusion/feature refinement; image feature maps are [B,C,H,W] around the diffusion block, with additional denoising steps and noise/refinement parameters.
- **IR-MFT migration decision:** Use as a contrastive audit and cost upper bound. No full diffusion or stochastic evaluation path is adopted.
- **License/reuse boundary:** Apache-2.0
### R03 — SMSTracker

- **Observed interface / cost boundary:** ScoreLayerUseConv/PLScoreLayerUseConv produce modal reliability/score features around ViT tokens. Token tensors are [B,N,C] and score heads add attention/FFN cost.
- **IR-MFT migration decision:** Source for F1 quality scoring; reimplemented as pooled FPN quality MLP, avoiding single-object template semantics.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R04 — FlexTrackV2

- **Observed interface / cost boundary:** MoE/adapters and LinearAttention_moe operate on token sequences with optional missing-modal masks. Experts add routed projections and capacity cost.
- **IR-MFT migration decision:** Source for missing-modality treatment; no full HMoE transplant, because detector FPN controls need matched capacity.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R05 — XTrack

- **Observed interface / cost boundary:** SparseDispatcher and LoRA adapters route [B,N,C] tokens to selected experts and avoid equal use of every modality.
- **IR-MFT migration decision:** Source for sparse routing rationale only. QRA uses one light gate per FPN level, not a copied Transformer.
- **License/reuse boundary:** MIT
### R06 — MOSSTrack

- **Observed interface / cost boundary:** FusionLayer.forward(x_v, x_i, temporal_query_RGB, temporal_query_TIR, lens_z) performs temporal-query cross-modal token fusion.
- **IR-MFT migration decision:** Source for F3 temporal state. We use a causal GRU over pooled FPN evidence; no template update or future token is used.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R07 — CADTrack

- **Observed interface / cost boundary:** DAttentionBaseline/CDA use deformable cross-attention over feature/token maps and learned offsets. Cost grows with sampled neighborhoods and projection layers.
- **IR-MFT migration decision:** Only a negative/conditional reference: RGB-derived auxiliary is nominally aligned, so deformable alignment is enabled only after measured misregistration.
- **License/reuse boundary:** MIT
### R08 — PFGF

- **Observed interface / cost boundary:** GraphReasoning/GraphModel perform intra/inter feature graph updates in a YOLOX PAFPN neck; feature maps are [B,C,H,W] with graph projections and iterative blocks.
- **IR-MFT migration decision:** Source for local complementarity reasoning; our spatial gate is a small 1x1 + depthwise 3x3 module.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R09 — M2D-LIF

- **Observed interface / cost boundary:** LIF/LIFAdd/MF implement feature interaction blocks for [B,C,H,W] maps. The repository is AGPL-licensed at the Ultralytics-derived path.
- **IR-MFT migration decision:** Audit-only. No AGPL code is copied; the implementation is independently written with compatible tensor semantics.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R10 — ModTr

- **Observed interface / cost boundary:** addition_fusion, attention_fusion_weight and spatial_fusion provide simple tensor fusion/attention controls.
- **IR-MFT migration decision:** Required control family, not a novelty claim. Static/duplicate-RGB controls are retained.
- **License/reuse boundary:** MIT
### R11 — SDSTrack

- **Observed interface / cost boundary:** VisionTransformerCE/SDSTrack use symmetric modal adapters and candidate elimination/masks on token sequences.
- **IR-MFT migration decision:** Reference for modality dropout and stability checks. No full ViT is added to YOLOX.
- **License/reuse boundary:** MIT
### R12 — SUTrack

- **Observed interface / cost boundary:** Unified Encoder receives template/search lists and task/modality indices, sharing encoder structure across tasks.
- **IR-MFT migration decision:** Reference for shared representation. Our two existing branches remain explicit to preserve the V3 checkpoint.
- **License/reuse boundary:** MIT
### R13 — UnTrack

- **Observed interface / cost boundary:** Expert/Router/MoE and prompt ViT operate on token features and route prompt/expert choices for missing modalities.
- **IR-MFT migration decision:** Reference for missing auxiliary intervention; implemented controls are zero, shuffle, repeat-RGB and grayscale.
- **License/reuse boundary:** ROOT LICENSE NOT FOUND at the pinned raw paths; source was inspected only.
### R14 — GOLA

- **Observed interface / cost boundary:** GOLA applies parameter-efficient LoRA/adapters and a multimodal DINO-style fusion path; source is Apache-2.0 with inherited dependencies.
- **IR-MFT migration decision:** Adapter efficiency is relevant to future shared-backbone work, but this round keeps the validated FPN channel budget.
- **License/reuse boundary:** Apache-2.0
### R15 — PURA

- **Observed interface / cost boundary:** PURA fast_parameter_update and parameter_decomposition_recovery update test-time parameters.
- **IR-MFT migration decision:** Explicitly excluded from fusion experiments: no Test-GT-driven TTA and no replacement for the causal multimodal question.
- **License/reuse boundary:** MIT
### R16 — MOTIP

- **Observed interface / cost boundary:** IDDecoder.forward consumes sequence identity information and trajectory_modeling returns learned trajectory features; this is token/sequence-level identity decoding.
- **IR-MFT migration decision:** Reserved for F4 memory only after visual fusion evidence; not used to turn an association gain into a fusion claim.
- **License/reuse boundary:** Apache-2.0
### R17 — TrackTrack

- **Observed interface / cost boundary:** Tracker.init_tracks/update maintain target-level tracks, NMS and CMC state. Inputs are detections/track objects rather than feature maps.
- **IR-MFT migration decision:** Reference for track-level fusion memory. The V3 BYTETracker baseline remains unchanged; no tracker transplant in F1–F3.
- **License/reuse boundary:** MIT
### R18 — HybridSORT

- **Observed interface / cost boundary:** HybridSORT association uses IoU, geometry, score, direction and size cues on detection arrays.
- **IR-MFT migration decision:** Association control only. It cannot be labelled a multimodal improvement.
- **License/reuse boundary:** MIT
### R19 — BoT-SORT

- **Observed interface / cost boundary:** STrack.update_features maintains an EMA feature history; matching.py computes IoU/embedding distances and assignment.
- **IR-MFT migration decision:** Reference for the later FusionMemoryTrack update rule; appearance weight zero must reproduce BYTETracker.
- **License/reuse boundary:** MIT
### R20 — OC-SORT

- **Observed interface / cost boundary:** speed_direction_batch and associate_detections_to_trackers provide motion-consistency and assignment controls over detection/track arrays.
- **IR-MFT migration decision:** Motion control only; causal temporal fusion is evaluated separately from motion association.
- **License/reuse boundary:** MIT

## Adopted ideas and explicit non-adoptions

- **Adopted as independent implementations:** pooled modal quality evidence (SMSTracker), lightweight routed/conditional capacity reasoning (FlexTrackV2/XTrack), spatial local interaction (PFGF/CADTrack audit), causal temporal query semantics (MOSSTrack), modality dropout controls (SDSTrack/UnTrack), and EMA-style memory diagnostics (BoT-SORT).
- **Controls only:** ModTr addition/attention/spatial fusion, HybridSORT, OC-SORT and BoT-SORT association cues. These are not labelled multimodal novelty.
- **Not copied:** PFTrack/DM3T full Transformer/diffusion, M2D-LIF AGPL implementation, large ViT/Mamba backbones, Test-Time Adaptation, and full ID-decoder trackers.
- **Scientific boundary:** the archived auxiliary images are RGB-derived; the strict-online path uses `CausalAuxiliaryGenerator` and records `physical_sensor: false`.
