"""Offline physical labels and finite error taxonomy; never imported by inference."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .provenance import sha256
from .supervision import independent_proposal_labels


def gt_by_video(annotation, protocol):
    # Identical physical annotation policy to the existing frozen query builder.
    names = {video['id']: video['file_name'] for video in annotation['videos']}
    images = {image['id']: image for image in annotation['images']}
    excluded = {(row['sequence'], row['frame_id'], row['track_id']) for row in protocol['ambiguous_identity_keys']}
    result = defaultdict(lambda: defaultdict(list))
    for entry in annotation['annotations']:
        image = images[entry['image_id']]
        seq, frame, identity = names[image['video_id']], int(image['frame_id']), int(entry['track_id'])
        if (seq, frame, identity) in excluded or entry.get('conf', 1) <= 0 or entry.get('iscrowd', 0):
            continue
        x, y, w, h = entry['bbox']
        if w > 0 and h > 0:
            result[seq][frame].append((identity, [x, y, x + w, y + h]))
    return result


def physical_bank(sequence, partition, protocol_path='configs/cfp/protocol_v2.json'):
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if sequence not in protocol[partition] or partition not in ('train', 'dev'):
        raise ValueError('Only frozen train/dev physical diagnosis permitted')
    annotation = Path('/data3/liuyeqiang/datasets/mft26-ir-causal-v1/annotations/train.json')
    if sha256(annotation) != protocol['train_annotation_sha256']:
        raise ValueError('Annotation drift')
    gt = gt_by_video(json.loads(annotation.read_text()), protocol)[sequence]
    bank_path = root / 'phase0' / f'causal_v3_{partition}_detections' / (sequence + '.npz')
    manifest = json.loads(bank_path.with_name('manifest.json').read_text())
    if sha256(bank_path) != manifest['caches'][sequence]['sha256']:
        raise ValueError('Proposal bank drift')
    with np.load(bank_path, allow_pickle=False) as bank:
        frames, offsets, boxes, scores = [bank[key] for key in ('frame_ids', 'offsets', 'boxes', 'scores')]
    labels = np.full(len(boxes), -1, np.int64)
    for i, frame in enumerate(frames):
        identities, truth = zip(*gt[int(frame)]) if gt[int(frame)] else ([], [])
        lo, hi = offsets[i:i + 2]
        labels[lo:hi], _ = independent_proposal_labels(boxes[lo:hi], truth, identities)
    return dict(frames=frames, offsets=offsets, boxes=boxes, scores=scores, labels=labels, gt=gt,
                frame_of_proposal=np.repeat(frames, np.diff(offsets)), detection_sha256=sha256(bank_path), annotation_sha256=sha256(annotation))


def history_view(track, maximum=8):
    history = [(int(t), int(index)) for t, index in track['history'] if index is not None]
    first = track.get('initial_proposal')
    if first is not None and all(index != first for _, index in history):
        history = [(int(track['initial_frame']), int(first))] + history[-(maximum - 1):]
    return history[-maximum:]


def modal_identity(indices, labels, minimum_consensus=.6):
    known = [int(labels[index]) for index in indices if labels[index] >= 0]
    if not known:
        return None, 0.
    counts = Counter(known)
    common = counts.most_common()
    fraction = common[0][1] / len(known)
    if fraction < minimum_consensus or (len(common) > 1 and common[0][1] == common[1][1]):
        return None, fraction
    return common[0][0], fraction


def classify_row(row, position, labels, association, features=None):
    """Operational, overlapping retrospective categories, not ground-truth causes.

    The modal known past identity is an analysis reference, never an immutable
    first-observation truth. Unknown/ambiguous histories remain unclassified.
    """
    track = row['tracks'][position]
    history = history_view(track)
    target, consensus = modal_identity([index for _, index in history], labels)
    if target is None:
        return dict(target=None, consensus=consensus, categories=[], known_history=0)
    proposals = row['proposals']
    candidates = labels[proposals]
    correct = np.flatnonzero(candidates == target)
    cost = np.asarray(row['costs'], float).reshape(len(row['tracks']), len(proposals))[position]
    geometry = np.asarray(row['geometry'], float).reshape(len(row['tracks']), len(proposals))[position]
    matches = {int(a): int(b) for a, b in row['matches']}
    chosen = matches.get(position)
    known = [int(labels[index]) for _, index in history if labels[index] >= 0]
    contamination = float(np.mean(np.asarray(known) != target)) if known else None
    categories = []
    if contamination and history[-1][0] - history[0][0] >= 30:
        categories.append('long_term_memory_pollution')
    gate = .5 if association == 'bot' else .9
    excluded = [int(j) for j in correct if geometry[j] > gate and cost[j] > row['threshold']]
    if excluded:
        categories.append('true_candidate_geometry_excluded')
    wrong = chosen is not None and candidates[chosen] >= 0 and candidates[chosen] != target
    margin = None
    if chosen is not None:
        alternatives = np.delete(cost, chosen)
        margin = min(float(alternatives.min()) if len(alternatives) else row['threshold'], row['threshold']) - cost[chosen]
    if wrong:
        categories.append('identity_inconsistent_match')
        if track['state'] != 1:  # Both native trackers use Tracked=1.
            categories.append('wrong_reactivation')
        if row['scores'][chosen] >= .8 and margin >= .05:
            categories.append('high_confidence_wrong_match')
        if features is not None and correct.size:
            clean = [index for _, index in history if labels[index] == target]
            prototype = features[clean].mean(0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
            bad_similarity = float(prototype @ features[proposals[chosen]])
            good_similarity = max(float(prototype @ features[proposals[j]]) for j in correct)
            if bad_similarity >= good_similarity - .02:
                categories.append('appearance_similar_swap_proxy')
        if correct.size and np.any(geometry[correct] <= .9):
            chosen_box = np.asarray(row['boxes'][chosen])
            for j in correct:
                true_box = np.asarray(row['boxes'][j])
                overlap = np.maximum(np.minimum(chosen_box[2:], true_box[2:]) - np.maximum(chosen_box[:2], true_box[:2]), 0).prod()
                union = np.prod(chosen_box[2:] - chosen_box[:2]) + np.prod(true_box[2:] - true_box[:2]) - overlap
                if overlap / max(union, 1e-8) >= .05:
                    categories.append('crossing_overlap_wrong_match_proxy')
                    break
    return dict(target=int(target), consensus=consensus, categories=categories, known_history=len(known),
                history_contamination=contamination, assigned_column=chosen,
                assigned_physical_identity=int(candidates[chosen]) if chosen is not None else None,
                true_columns=correct.tolist(), geometry_excluded_columns=excluded, row_margin=float(margin) if margin is not None else None,
                true_candidate_recall=bool(correct.size), correct_match=not wrong if chosen is not None and candidates[chosen] >= 0 else None)
