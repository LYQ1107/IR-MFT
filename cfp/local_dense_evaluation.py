"""Same-current-scene physical diagnostics, separate from native tracking."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .identity_evidence import pair_discrimination, query_retrieval
from .provenance import sha256, write_json


def scene_query_plan(bank, video, cap=64, maximum_age=1500):
    plan = defaultdict(list)
    frames = bank.videos[video]['sampled_frames']
    for identity, history in sorted(bank.videos[video]['histories']['gt'].items()):
        eligible = [frame for frame in frames if any(frame-maximum_age <= t < frame for t, i in history)]
        if eligible:
            selected = np.unique(np.linspace(0, len(eligible)-1, min(cap, len(eligible)), dtype=int))
            for position in selected:
                plan[eligible[position]].append(identity)
    return dict(sorted(plan.items()))


def summarize_queries(rows):
    with_target = [r for r in rows if r['target_present']]
    supported = [r for r in with_target if r['history_available']]
    return dict(queries=len(rows), target_present=len(with_target), target_absent=len(rows)-len(with_target),
        history_absent=sum(not r['history_available'] for r in rows),
        supported_history_and_target=len(supported),
        rank1_percent=100*np.mean([r['rank1'] for r in supported]) if supported else None,
        rank5_percent=100*np.mean([r['rank5'] for r in supported]) if supported else None,
        mAP_percent=100*np.mean([r['average_precision'] for r in supported]) if supported else None,
        unknown_selected=sum(r['unknown_selected'] for r in supported),
        rank1_including_nohistory_percent=100*np.mean([r['rank1'] for r in with_target]) if with_target else None,
        hard_known_negative_margin_mean=float(np.mean([r['hard_negative_margin'] for r in supported if r['hard_negative_margin'] is not None]))
            if any(r['hard_negative_margin'] is not None for r in supported) else None)


def measured_pair_scores(positive, negative):
    result = pair_discrimination(positive, negative)
    # A learned relation logit is not cosine similarity.
    return {name.replace('cosine', 'score'): value for name, value in result.items()}


def evaluate_local_scenes(student, bank, video, config, audit_directory=None):
    student.eval()
    plan = scene_query_plan(bank, video, config['query_cap_per_physical_identity'], config['maximum_history_age_frames'])
    report = dict(query_frames=len(plan), protocol={}, sampled_frames=len(bank.videos[video]['sampled_frames']),
        empty_both_sampled_frames=bank.videos[video]['sampled_empty_both_frames'],
        scope='Clean physical offline inventory includes currently absent previously seen fish. Past-only GT and verified-detector histories are separate, not actual native association memories. Full current proposal rows retained; unknowns not labeled negatives. No GT/teacher/IDs enter student.')
    with torch.inference_mode():
        for history_view in ('gt', 'detector'):
            report['protocol'][history_view] = {}
            for candidate_view in ('gt', 'detector'):
                report['protocol'][history_view][candidate_view] = {}
                for gap in config['minimum_past_gaps']:
                    rows, positive, negative, audit = [], [], [], []
                    timings, input_candidates, observable_candidates, unknowns = [], 0, 0, 0
                    predicted_values = []
                    for frame, identities in plan.items():
                        scene = bank.scene(video, frame, identities, history_view, candidate_view,
                            config['history_length'], config['maximum_history_age_frames'], minimum_gap=gap)
                        start = time.monotonic()
                        output = student(**scene['inputs'])
                        timings.append(time.monotonic()-start)
                        legal = output['current']['valid'].any(1)
                        labels = scene['candidate_ids'][legal].numpy()
                        scores = output['scores'][:, legal].numpy()
                        input_candidates += len(scene['candidate_ids'])
                        observable_candidates += int(legal.sum())
                        unknowns += int((scene['candidate_ids'][legal]<0).sum())
                        if output['history_available'].any():
                            predicted_values.append(output['predicted'][output['history_available']].mean(1).cpu().numpy())
                        for j, identity in enumerate(identities):
                            edge_labels = np.where(labels<0, -1, (labels!=identity).astype(np.int8))
                            result = query_retrieval(scores[j], edge_labels)
                            result['history_available'] = bool(output['history_available'][j])
                            rows.append(result)
                            if result['history_available']:
                                positive.extend(scores[j, edge_labels==0].tolist())
                                negative.extend(scores[j, edge_labels==1].tolist())
                            audit.append(dict(frame=frame, query_identity=identity,
                                history_available=result['history_available'], target_present=result['target_present'],
                                input_candidates=len(scene['candidate_ids']), observable_candidates=int(legal.sum()),
                                unknown_candidates=int((labels<0).sum()), history_indices=scene['history_indices'][j],
                                current_indices=scene['current_indices']))
                    values = np.concatenate(predicted_values, axis=0) if predicted_values else None
                    support = audit
                    if audit_directory is not None:
                        path = Path(audit_directory)/(video+'_'+history_view+'_'+candidate_view+'_gap'+str(gap)+'.json')
                        if path.resolve().is_relative_to(Path.cwd().resolve()):
                            raise ValueError('Physical query annotation traces must stay outsideGit')
                        write_json(path, dict(scope=report['scope'], rows=audit))
                        support = dict(path=str(path), sha256=sha256(path), rows=len(audit))
                    report['protocol'][history_view][candidate_view][str(gap)] = dict(
                        retrieval=summarize_queries(rows), discrimination=measured_pair_scores(positive, negative),
                        input_candidate_observations=input_candidates, observable_candidate_observations=observable_candidates,
                        unknown_candidate_observations=unknowns, frame_model_seconds=sum(timings),
                        frame_model_p50_p95_ms=(1000*np.percentile(timings, [50, 95])).tolist() if timings else None,
                        predicted_feature_variance=float(values.var(0).mean()) if values is not None else None,
                        query_support=support,
                        score_units='Unthresholded identity pair logits; EER diagnostic only. Repeated fish/frames are dependent samples. CPU cached-feature evaluation, not raw online FPS.')
    return report
