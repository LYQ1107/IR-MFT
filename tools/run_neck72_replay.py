#!/usr/bin/env python3
"""Causal seed0 neck72 replay; all inference finishes before official GT eval."""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.neck72_experiment import registered, STUDY
from tools.neck72_evidence import load_neck72_evidence
from tools.neck72_runtime import read, require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.ordered_identity_memory import OrderedIdentityMemoryTracker
from yolox.tracker.consensus_identity_memory import ConsensusIdentityMemoryTracker


class QueryIncrementObserver:
    """Read returned packets once per frame; do not change any decision/state."""
    def __init__(self, tracker):
        self.frame, self.seen = None, set()
        self.values = dict(packets=0, pair_cells=0, nonzero_delta_cells=0,
            available_cells=0, sum_delta=0., sum_abs_delta=0., max_abs_delta=0.,
            branch_disabled=bool(tracker.config.get('disable_native_increment', False)),
            scope='All cells of actual cached query matrices, no inference padding; '
                  'bounded logit increments, not sigmoid gates or object-only averages.')
        self.original = tracker.query_posterior

    def __call__(self, frame, context):
        if frame != self.frame:
            self.frame, self.seen = frame, set()
        packet = self.original(frame, context)
        key = id(packet)
        if key not in self.seen:
            self.seen.add(key)
            delta = np.asarray(packet['delta'], np.float64)
            assert np.isfinite(delta).all() and np.max(abs(delta), initial=0) <= 4.00001
            self.values['packets'] += 1; self.values['pair_cells'] += delta.size
            self.values['nonzero_delta_cells'] += int(np.count_nonzero(delta))
            self.values['sum_delta'] += float(delta.sum())
            self.values['sum_abs_delta'] += float(abs(delta).sum())
            self.values['max_abs_delta'] = max(self.values['max_abs_delta'], float(np.max(abs(delta), initial=0)))
            if self.values['branch_disabled']:
                assert not np.any(delta)
            else:
                self.values['available_cells'] += int(np.count_nonzero(packet['available']))
        return packet


def load_sequence(plan, work, split, sequence, config):
    arm = config['detection_arm']
    if config['backend'] == 'byte':
        source = Path(plan['frozen_detection_sources'][('new_diffv2ir' if arm == 'new' else 'rgb')+'_'+split]) / 'detections' / (sequence+'.npz')
    else:
        source = work / 'features' / split / sequence / (arm+'.npz')
        complete = read(source.parent / 'complete.json')
        assert complete['status'] == 'COMPLETE' and sha256(source) == complete['outputs'][arm]['sha256']
    with np.load(source) as saved:
        data = {name: saved[name] for name in saved.files}
    metadata = dict(path=str(source), sha256=sha256(source))
    np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids'])+1))
    assert data['offsets'][-1] == len(data['boxes'])
    if config['backend'] != 'byte':
        path = work / 'transport' / split / sequence / (arm+'.npz')
        complete = read(path.parent / 'complete.json')
        assert complete['status'] == 'COMPLETE' and sha256(path) == complete['outputs'][arm]['sha256']
        with np.load(path) as saved:
            data['foreground_transport'] = {name: saved[name] for name in saved.files}
        np.testing.assert_array_equal(data['foreground_transport']['frame_ids'], data['frame_ids'])
        np.testing.assert_array_equal(data['foreground_transport']['detection_offsets'], data['offsets'])
        metadata['transport'] = dict(path=str(path), sha256=sha256(path))
        metadata['native_evidence'] = load_neck72_evidence(data, work, split, sequence, config)
    return data, metadata


def tracker_for(config):
    BaseTrack._count = 0
    if config['backend'] == 'byte':
        return BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False), frame_rate=25)
    factory = {'ordered_query': OrderedIdentityMemoryTracker, 'consensus_query': ConsensusIdentityMemoryTracker}
    if config['backend'] not in factory or config['native_seed'] != 0:
        raise ValueError('Only registered neck72 seed0 backends')
    return factory[config['backend']](config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    args = parser.parse_args()
    plan, protocol, work, study, _ = registered()
    config_path = Path(args.config); config = read(config_path)
    if args.split == 'test':
        freeze = read(work / 'neck72_Test_selection_frozen.json')
        assert freeze['study_sha256'] == sha256(STUDY)
        assert any(c['config'] == config for c in freeze['cases'])
        assert freeze['complete_selected_Dev_before_Test'] and not freeze['Test_accuracy_read_for_selection']
    sequences = protocol['dev' if args.split == 'dev' else 'historical_test']
    output = work / ('neck72_'+args.split) / config['id']
    assert not (output / 'metrics.json').exists(), 'Preserve complete experiments'
    require_host_memory(4*2**30, maximum_workers=2); require_storage(work)
    sources, coverage, statistics, increments = {}, {}, {}, {}
    started = time.monotonic(); head_parameters = None
    for sequence in sequences:
        data, sources[sequence] = load_sequence(plan, work, args.split, sequence, config)
        tracker = tracker_for(config); observer = None
        if config['backend'] != 'byte':
            assert tracker.native_metadata['seed'] == 0
            if not config.get('disable_native_increment', False):
                assert sources[sequence]['native_evidence']['alignment']['sha256'] == tracker.native_metadata['alignment_sha256']
            observer = QueryIncrementObserver(tracker); tracker.query_posterior = observer
            head_parameters = dict(native_query=sum(p.numel() for p in tracker.native_model.parameters()),
                                   frozen_RGB_pair_scorer=sum(p.numel() for p in tracker.model.parameters()))
        rows = []
        for frame in data['frame_ids']:
            if config['backend'] == 'byte':
                lo, hi = data['offsets'][frame-1:frame+1]
                detections = np.c_[data['boxes'][lo:hi], data['scores'][lo:hi]].astype(np.float32)
                rows.extend((int(frame), t.tlwh.copy(), t.track_id, t.score)
                            for t in tracker.update(detections, (1, 1), (1, 1)) if t.tlwh[2]*t.tlwh[3] > 100)
            else:
                rows.extend(tracker.update(int(frame), data))
        path = output / 'track_results' / (sequence+'.txt'); write_sequence_results(str(path), rows)
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']),
                                 output_rows=len(rows), output_sha256=sha256(path))
        statistics[sequence] = getattr(tracker, 'statistics', {})
        if observer is not None:
            increments[sequence] = observer.values
    seconds = time.monotonic()-started
    frames = sum(v['frames'] for v in coverage.values())
    assert frames == (4834 if args.split == 'dev' else 15985)
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config,
        config_path=str(config_path), config_sha256=sha256(config_path), study_sha256=sha256(STUDY),
        split=args.split, sequences=sequences, frames=frames, sources=sources, coverage=coverage,
        statistics=statistics, query_increment_statistics=increments, head_parameters=head_parameters,
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        tracking_and_ID_counter_reset_each_sequence=True,
        replay_seconds=seconds, replay_FPS=frames/seconds,
        runtime_scope='Instrumented cached CPU replay, including loading verified vectors, transport '
            'and ByteTrack/query decisions; detector, pixel encoders, point production and IR generation '
            'excluded. Not full-system FPS.', source_sha256=sha256(__file__))
    dump(output / 'inference_manifest.json', manifest)
    values = run_trackeval(str(output / 'track_results'), plan['evaluation']['GT_root'], str(output),
        sequences, plan['evaluation']['TrackEval_root'], data_split='train' if args.split == 'dev' else 'test')
    assert len(values) == 39
    dump(output / 'metrics.json', dict(status='COMPLETE', trackeval=values, manifest=manifest))
    print(config['id'], args.split, 'COMPLETE', frames, flush=True)


if __name__ == '__main__':
    main()
