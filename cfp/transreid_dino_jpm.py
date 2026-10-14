"""Source-matched JPM mechanism on the same licensed DINO crop encoder.

Not official TransReID: DINO patch14, no camera/view SIE; fit-only video-key
labels and temporal same-video positives. Prefix CLS has already mixed all
patches. JPM parts are shuffled token subsets, not anatomical/local-only evidence.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from third_party.transreid_jpm.shuffle import shuffle_unit
from .dinov2_identity import resized_observations
from .local_identity_training import local_identity_objective


class DinoJointPatchReID(nn.Module):
    def __init__(self, backbone, identities, mode='jpm'):
        super().__init__()
        if mode not in ('global', 'jpm', 'cls_capacity') or identities < 2:
            raise ValueError('Registered JPM mode and fit-only identity vocabulary required')
        if len(backbone.blocks) != 12 or backbone.num_register_tokens != 0 or backbone.embed_dim != 384:
            raise ValueError('Exact nonregister DINOv2S required')
        self.mode = mode
        self.backbone = backbone.requires_grad_(False)
        # Last original block/norm remain immutable; executed branches are copies.
        self.backbone.blocks[10].requires_grad_(True)
        self.global_branch = nn.Sequential(copy.deepcopy(backbone.blocks[11]), copy.deepcopy(backbone.norm)).requires_grad_(True)
        self.local_branch = None if mode == 'global' else nn.Sequential(
            copy.deepcopy(backbone.blocks[11]), copy.deepcopy(backbone.norm)).requires_grad_(True)
        branches = 1 if mode == 'global' else 5
        self.necks = nn.ModuleList([nn.BatchNorm1d(384) for _ in range(branches)])
        self.classifiers = nn.ModuleList([nn.Linear(384, identities, bias=False) for _ in range(branches)])
        for neck in self.necks:
            nn.init.normal_(neck.weight, 1., .02)
            nn.init.zeros_(neck.bias); neck.bias.requires_grad_(False)
        for classifier in self.classifiers:
            nn.init.normal_(classifier.weight, std=.001)

    def forward(self, images, content):
        prepared, fraction = resized_observations(images, content)
        with torch.no_grad():
            tokens = self.backbone.prepare_tokens_with_masks(prepared)
            for block in self.backbone.blocks[:10]:
                tokens = block(tokens)
        tokens = self.backbone.blocks[10](tokens)
        valid = fraction.flatten(1).sum(-1) > 0
        raw = [self.global_branch(tokens)[:, 0]]
        part_fraction = fraction.new_zeros(len(images), 4)
        if self.local_branch is not None:
            # Exact source shift5/group2, floor162/4=40, final2tokens omitted.
            shuffled = shuffle_unit(tokens, shift=5, group=2)
            reordered_fraction = shuffle_unit(torch.cat([fraction.new_zeros(len(images), 1, 1),
                fraction.flatten(1)[..., None]], 1), shift=5, group=2).squeeze(-1)
            for part in range(4):
                pieces = shuffled[:, part*40:(part+1)*40]
                if self.mode == 'cls_capacity':
                    pieces = tokens[:, :1].expand(-1, 40, -1)
                raw.append(self.local_branch(torch.cat([tokens[:, :1], pieces], 1))[:, 0])
                part_fraction[:, part] = reordered_fraction[:, part*40:(part+1)*40].mean(-1)
        neck = [module(value) for module, value in zip(self.necks, raw)]
        # Source uses TEST.NECK_FEAT=after, each local descriptor divided by4.
        feature = torch.cat([neck[0], *[x/4 for x in neck[1:]]], -1).float()
        return dict(embedding=F.normalize(feature, dim=-1)*valid[:, None],
                    raw_global=F.normalize(raw[0].float(), dim=-1)*valid[:, None],
                    raw_branches=raw, logits=[m(x).float() for m, x in zip(self.classifiers, neck)],
                    crop_valid=valid, part_content_fraction=part_fraction)

    def immutable_base(self):
        state = self.state_dict()
        return {'backbone.'+n: state['backbone.'+n].detach().cpu().clone()
                for n, p in self.backbone.named_parameters() if not p.requires_grad}


def jpm_identity_objective(output, labels, frames, videos, minimum_gap=25,
                           triplet_weight=1., same_frame_weight=.2):
    valid = output['crop_valid']
    if not valid.any():
        raise ValueError('No actual content in sampled batch')
    losses, details = [], []
    for raw, logits in zip(output['raw_branches'], output['logits']):
        # Shared fish loss/support across modes; source half-global/half-local
        # head weighting retained, same-video temporal metric replaces camera exclusion.
        loss, values = local_identity_objective(dict(embedding=F.normalize(raw[valid].float(), dim=-1),
            logits=logits[valid]), labels[valid], frames[valid], videos[valid],
            minimum_gap, triplet_weight, same_frame_weight)
        losses.append(loss); details.append(values)
    weights = [1.] if len(losses) == 1 else [.5, .125, .125, .125, .125]
    result = {k: sum(w*d[k] for w, d in zip(weights, details)) for k in details[0]}
    return sum(w*x for w, x in zip(weights, losses)), result
