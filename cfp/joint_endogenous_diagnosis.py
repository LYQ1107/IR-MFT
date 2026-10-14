"""Offline physical-history/cue/NULL/dictionary diagnosis, never inference."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from .centered_joint_identity import CenteredJointIdentity
from .joint_identity_decoder import decode_joint_assignment
from .joint_identity_supervision import joint_identity_targets
from .recoverability_diagnosis import EarlyReference
from .recovery_supervision import history_view, modal_identity


def classify_history(track, labels, reference):
    history = history_view(track)
    indices = [index for _, index in history]
    values = labels[indices]
    known = values >= 0
    recent = labels[[int(index) for _, index in track['history'] if index is not None]]
    result = dict(observations=len(indices), known=int(known.sum()), unknown=int((~known).sum()))
    if reference is None:
        return dict(result, state='unknown_early_reference', known_correct=None, known_wrong=None,
                    recent_known_correct=None, latest_disagrees=None, anchor_disagrees=None, modal_disagrees=None)
    correct = values == reference
    wrong = known & ~correct
    modal, _ = modal_identity(indices, labels)
    state = 'mixed_known_identity' if wrong.any() and correct.any() else (
        'known_no_correct_cue' if wrong.any() else ('known_consistent_with_unknowns' if (~known).any() else 'all_known_consistent'))
    anchor = track.get('initial_proposal')
    latest = int(recent[-1]) if len(recent) else -1
    return dict(result, state=state, known_correct=int(correct.sum()), known_wrong=int(wrong.sum()),
                recent_known_correct=int((recent == reference).sum()),
                latest_disagrees=bool(latest >= 0 and latest != reference),
                anchor_disagrees=bool(anchor is not None and labels[anchor] >= 0 and labels[anchor] != reference),
                modal_disagrees=bool(modal is not None and modal != reference))


def classify_selection(selected, target, n):
    if target < 0:
        return 'unsupervised_unknown_or_nonunique'
    if target == n:
        return 'correct_known_null' if selected < 0 else 'false_existing_for_known_null'
    if selected < 0:
        return 'false_null_for_known_existing'
    return 'correct_existing' if selected == target else 'wrong_existing'


def diagnose_native_rows(folder, physical, fit, config):
    labels = physical['labels']
    with gzip.open(folder/'model_observations.jsonl.gz', 'rt') as stream:
        observations = {row['frame']: row for row in map(json.loads, stream)}
    refs = EarlyReference()
    counts = Counter()
    cue_counts = defaultdict(Counter)
    by_history = defaultdict(Counter)
    margins = defaultdict(list)
    supervised = {}
    with gzip.open(folder/'decisions.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            frame = row['frame']
            for track in row['tracks']:
                for t, index in history_view(track):
                    if t >= frame or physical['frame_of_proposal'][index] != t:
                        raise ValueError('Offline analysis history is not actual strictly past accepted proposals')
            identities = [refs.update(track, labels) for track in row['tracks']]
            if row['stage'] != 'high' or frame not in observations:
                continue
            observed = observations[frame]
            tracks, current = row['tracks'], np.asarray(row['proposals'], dtype=np.int64)
            n, m = len(tracks), len(current)
            if (physical['frame_of_proposal'][current] != frame).any():
                raise ValueError('Current proposal alignment drift')
            logits = np.asarray([[float('-inf') if x is None else x for x in v] for v in observed['logits']], dtype=np.float64)
            if logits.shape != (m, n+1):
                raise ValueError('Model observation and native high-stage set differ')
            physical_tracks = torch.tensor([-1 if x is None else x for x in identities], dtype=torch.int64)
            supervision = joint_identity_targets('inner', physical_tracks, torch.from_numpy(labels[current].copy()))
            targets = supervision['target'].numpy()
            supervised[frame] = (targets, n)
            matches = {int(j): int(i) for i, j in row['matches']}
            if any(matches.get(j, -1) != selected for j, selected in enumerate(observed['selected'])):
                raise ValueError('Recorded model choice did not equal actual native high-stage matching')
            history = [classify_history(track, labels, identity) for track, identity in zip(tracks, identities)]
            counts.update(frames=1, detections=m, track_slots=n)
            for h in history:
                counts['track_history_'+h['state']] += 1
                if h['known_wrong'] is not None:
                    counts['known_history_observations'] += h['known']
                    counts['known_wrong_history_observations'] += h['known_wrong']
                    for key in ['latest_disagrees', 'anchor_disagrees', 'modal_disagrees']:
                        counts[key] += int(h[key])
            for j, (target, reason, selected) in enumerate(zip(targets, supervision['reason'], observed['selected'])):
                counts['target_'+reason] += 1
                choice = classify_selection(selected, target, n)
                counts[choice] += 1
                if target < 0 or target == n:
                    continue
                h = history[target]
                by_history[h['state']][choice] += 1
                by_history[h['state']]['supported_existing'] += 1
                counts['existing_without_recent_correct_cue'] += int(not h['recent_known_correct'])
                counts['existing_without_any_correct_cue'] += int(not h['known_correct'])
                old = logits[j, :-1]
                alternative = np.delete(old, target)
                margins['correct_minus_null'].append(float(old[target]-logits[j, -1]))
                if len(alternative):
                    margins['correct_minus_best_wrong'].append(float(old[target]-alternative.max()))
                cue_counts['model_existing_logits']['queries'] += 1
                cue_counts['model_existing_logits']['rank1'] += int(np.argmax(old) == target)
                for cue in ['fifo_cost', 'last_cost', 'gru_cost', 'geometry']:
                    cost = np.asarray(observed[cue], dtype=np.float64)[:, j]
                    available = np.asarray(observed['history_available']) if cue != 'geometry' else np.ones(n, dtype=bool)
                    if not available[target]:
                        cue_counts[cue]['missing_target_history'] += 1
                        continue
                    cue_counts[cue]['queries'] += 1
                    cue_counts[cue]['rank1'] += int(np.argmin(np.where(available, cost, np.inf)) == target)
    result = dict(counts=dict(counts), by_target_history={k: dict(v) for k, v in by_history.items()},
                  local_same_set_existing_rank1={k: dict(v, percentage=100*v['rank1']/max(v['queries'], 1)) for k, v in cue_counts.items()},
                  logit_margins={k: dict(count=len(v), mean=float(np.mean(v)), p10_p50_p90=np.quantile(v, [.1, .5, .9]).tolist()) for k, v in margins.items()},
                  supervision='Frozen strictly-past >=3observation EarlyReference consensus, not unconditional physical track truth. Unknown/nonunique and unknown old coverage ignored; absent verified match is not verified birth. All physical labels are independently matched same-frame proposals after all output parity checks.',
                  local_cue_scope='Same endogenous candidate/track sets only. Local per-candidate ranks omit one-to-one/NULL/lifecycle; not counterfactual native tracker gains.')
    checkpoint = torch.load(Path(fit)/'epoch8.pt', map_location='cpu', weights_only=False)
    meta = json.loads((Path(fit)/'manifest.json').read_text())
    model = CenteredJointIdentity(config['dimension'], config['hidden'], config['heads'], config['capacity'], meta['guard']['architecture']).eval()
    model.load_state_dict(checkpoint['model'])
    packets = torch.load(folder/'sparse_observable_inputs.pt', map_location='cpu', weights_only=False)
    perturbations = defaultdict(Counter)
    with torch.inference_mode():
        for packet in packets:
            inputs = packet['inputs']; target, n = supervised[packet['frame']]
            original = model(**inputs)['logits']
            if not torch.equal(original, packet['logits']):
                raise ValueError('Sparse offline original inputs failed exact model-logit parity')
            baseline = decode_joint_assignment(original, inputs['detection_valid']).numpy()
            for variant in [1001, 1002, 1003, 1004, 'zero_dictionary']:
                changed = dict(inputs)
                if variant == 'zero_dictionary':
                    saved = model.dictionary.weight.clone(); model.dictionary.weight.zero_()
                else:
                    rng = np.random.default_rng(variant+packet['frame'])
                    changed['codes'] = torch.from_numpy(rng.permutation(config['capacity'])[:n].astype(np.int64))
                altered = model(**changed)['logits']
                selection = decode_joint_assignment(altered, inputs['detection_valid']).numpy()
                if variant == 'zero_dictionary':
                    model.dictionary.weight.copy_(saved)
                c = perturbations[str(variant)]; c.update(frames=1, detections=len(target), selected_assignment_changes=int((selection != baseline).sum()))
                for selected, wanted in zip(selection, target):
                    c[classify_selection(selected, wanted, n)] += 1
            perturbations['original']['frames'] += 1
            perturbations['original']['detections'] += len(target)
            for selected, wanted in zip(baseline, target):
                perturbations['original'][classify_selection(selected, wanted, n)] += 1
    result['offline_dictionary_sensitivity'] = {k: dict(v) for k, v in perturbations.items()}
    result['dictionary_scope'] = 'Fixed stride5 inner observed sets, four predeclared alternative distinct-code assignments and isolated zero-dictionary intervention. No retraining/threshold selection, no altered online outputs or claimed full tracker gain; arbitrary local dictionary embedding dependence is measured as a nuisance sensitivity.'
    return result
