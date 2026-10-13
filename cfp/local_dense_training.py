"""Two objectives: local prediction and explicit instance identity evidence."""
import torch
from torch.nn import functional as F

from .local_dense_jepa import dense_prediction_loss, known_candidate_loss


def local_dense_objective(student, teacher, scene, label_map, target_mask,
                          prediction_weight=1., relation_weight=.5):
    output = student(**scene['inputs'])
    zero = output['predicted'].sum()*0
    classification, relation, prediction, contrastive = zero, zero, zero, zero
    supervised_rows, classified = 0, 0
    if student.mode != 'no_id':
        # Clean GT is an ordinary *training supervision* branch. It is never
        # supplied to history_context/predict or online relation inference.
        current_clean = student.current(scene['target_local'], scene['target_fraction'], scene['target_global'])
        clean_valid = current_clean['valid'].any(1)
        candidate_valid = (scene['candidate_ids']>=0) & output['current']['valid'].any(1)
        labels = [label_map[(scene['video'], int(identity))] for identity in scene['track_ids'][clean_valid]]
        labels += [label_map[(scene['video'], int(identity))] for identity in scene['candidate_ids'][candidate_valid]]
        features = torch.cat([current_clean['feature'][clean_valid], output['current']['feature'][candidate_valid]])
        classified = len(labels)
        if classified:
            labels = torch.tensor(labels, dtype=torch.int64, device=features.device)
            classification = F.cross_entropy(student.identity_logits(features), labels, label_smoothing=.1)
        relation, supervised_rows = known_candidate_loss(output['scores'], scene['track_ids'],
            scene['candidate_ids'], output['history_available'])
    target = teacher(scene['target_local'], scene['target_fraction'])
    masked_tokens = 0
    if student.mode in ('full', 'no_ema', 'no_id'):
        prediction, masked_tokens = dense_prediction_loss(output['predicted'], target,
            target_mask, output['history_available'])
    if student.mode == 'local_contrastive':
        available = output['history_available'] & target['valid'].any(1)
        query = F.normalize(output['predicted'].mean(1)[available], dim=-1)
        clean = F.normalize(target['feature'][available].detach(), dim=-1)
        if len(query)>1:
            contrastive = F.cross_entropy(query @ clean.T/.1, torch.arange(len(query), device=query.device))
    identity = classification+relation_weight*relation
    loss = identity+prediction_weight*(prediction+contrastive)
    parts = dict(total=float(loss.detach()), classification=float(classification.detach()),
        relation=float(relation.detach()), prediction=float(prediction.detach()),
        contrastive=float(contrastive.detach()), classified_observations=classified,
        known_supported_relation_rows=supervised_rows, masked_target_tokens=masked_tokens,
        tracks=len(scene['track_ids']), candidates=len(scene['candidate_ids']),
        unknown_candidates=int((scene['candidate_ids']<0).sum()),
        target_variance=float(F.normalize(target['tokens'], dim=-1).flatten(0, 1).var(0, unbiased=False).mean()),
        predicted_variance=float(output['predicted'].flatten(0, 1).var(0, unbiased=False).mean()))
    return loss, parts
