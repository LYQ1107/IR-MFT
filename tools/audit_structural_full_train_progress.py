#!/usr/bin/env python3
"""Read coherent live checkpoint bytes and verify actual Train-address delivery.

This is a CPU-only progress audit. It does not evaluate a model, read held-out
accuracy, copy a weight file or change any active training source.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-any-full-epoch', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'Do not overwrite a published progress snapshot'
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    registry_path = work / 'structural_full_train_pipeline_registered.json'
    registry = json.loads(registry_path.read_text())
    assert all(sha256(ROOT / p) == h for p, h in registry['source_hashes'].items())
    schedule_path = work / 'structural_full_train_schedule/complete.json'
    schedule = json.loads(schedule_path.read_text())
    addresses_path = Path(schedule['address_file']['path'])
    assert sha256(addresses_path) == schedule['address_file']['sha256']
    with np.load(addresses_path) as saved:
        addresses = saved['addresses']
    assert addresses.shape == (72, 288, 4, 2)
    sequences, lengths = schedule['sequence_order'], schedule['lengths']
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    assert sequences == protocol['train'] and sum(lengths.values()) == 27251
    config_path = ROOT / 'configs/perceptual_structural_fulltrain_20261123.json'
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    state = json.loads((work / 'structural_full_train_fit_state.json').read_text())
    rows = []
    for name, job in state['jobs'].items():
        if job['status'] != 'FIT_RUNNING':
            continue
        pid = job['pid']
        proc = Path('/proc') / str(pid)
        assert b'tools/train_structural_full_train.py' in (proc / 'cmdline').read_bytes()
        environment = (proc / 'environ').read_bytes().split(b'\0')
        gpu = [v.decode().split('=', 1)[1] for v in environment if v.startswith(b'CUDA_VISIBLE_DEVICES=')]
        assert gpu == [job['physical_gpu']] and gpu[0] in ['4', '5']
        path = work / 'structural_full_train_fit' / name / 'latest.pt'
        # Atomic writer replacement cannot mutate these already-read bytes.
        snapshot = path.read_bytes()
        saved = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=False)
        completed = saved['completed_epochs']
        meta, history = saved['metadata'], saved['history']
        assert 0 < completed <= 72 and len(history) == len(meta['live_frame_passes']) == completed
        assert saved['config_sha256'] == meta['config_sha256'] == sha256(config_path)
        assert meta['full_train_study_sha256'] == sha256(study_path)
        assert meta['full_train_address_sha256'] == schedule['semantic_address_sha256']
        assert (meta['cue'], meta['variant'], meta['seed']) == (job['cue'], job['components'], 0)
        assert all(sha256(ROOT / p) == h for p, h in meta['source_hashes'].items())
        assert sum(v.numel() for v in saved['adapter'].values()) == 145600
        assert all(torch.isfinite(v).all() for v in saved['adapter'].values())
        digests = []
        for index, (record, passed) in enumerate(zip(history, meta['live_frame_passes'])):
            digest = hashlib.sha256()
            for batch in addresses[index]:
                draw = [(sequences[int(i)], int(frame)) for i, frame in batch]
                digest.update(json.dumps(draw, separators=(',', ':')).encode())
            expected = digest.hexdigest()
            assert record['epoch'] == passed['pass_index'] == index + 1
            assert record['steps'] == passed['prepared_batches'] == 288 and passed['complete_input_pass']
            assert passed['logical_Train_epoch'] == index // 24 + 1
            assert expected == record['sample_digest'] == passed['actual_frame_draw_sha256'] == passed['expected_frame_draw_sha256']
            assert np.isfinite(record['loss']).all()
            for i, seq in enumerate(sequences):
                batch = addresses[index]
                count = len(np.unique(batch[..., 1][batch[..., 0] == i]))
                assert passed['unique_frames'][seq] == count
            digests.append(expected)
        for epoch in range(completed // 24):
            block = addresses[epoch * 24:(epoch + 1) * 24]
            for i, seq in enumerate(sequences):
                np.testing.assert_array_equal(np.unique(block[..., 1][block[..., 0] == i]),
                                              np.arange(1, lengths[seq] + 1))
        block = addresses[:completed]
        coverage = {seq: len(np.unique(block[..., 1][block[..., 0] == i])) for i, seq in enumerate(sequences)}
        rows.append(dict(id=name, physical_gpu=gpu[0], observed_live_pid=pid,
            completed_macro_passes=completed, full_Train_epochs_completed=completed // 24,
            successful_updates=sum(r['steps'] for r in history), sample_exposures=completed * 288 * 4,
            unique_Train_frames_seen=sum(coverage.values()), per_video_unique_frames=coverage,
            all_actual_provider_address_digests_exact=True, actual_draw_digests=digests,
            live_checkpoint_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), last_training_loss=history[-1]['loss']))
    assert 0 < len(rows) <= 2 and len({r['physical_gpu'] for r in rows}) == len(rows)
    if args.require_any_full_epoch:
        assert any(r['full_Train_epochs_completed'] >= 1 for r in rows), 'No completed full traversal yet'
    report = dict(status='COMPLETE_LIVE_TRAIN_ADDRESS_DELIVERY_PROGRESS_AUDIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
        source_sha256=sha256(__file__), source_registry_sha256=sha256(registry_path),
        schedule_proof_sha256=sha256(schedule_path), pinned_code_and_config_unchanged=True,
        GPU_work=False, Dev_read=False, Test_read=False, goal_met=False, supported_innovations=0,
        scope='Actual consumed Train addresses checked against the registered schedule for every completed '
              'macro pass in coherent in-memory checkpoint snapshots.24macro passes are one complete '
              '27251frame traversal. Mutable latest.pt snapshot hashes are not permanent checkpoint releases. '
              'This is not convergence, completed six-fit training or evidence of HOTA improvement.')
    dump(output, report)
    print(json.dumps(dict(status=report['status'], rows=[{k: r[k] for k in [
        'id', 'completed_macro_passes', 'full_Train_epochs_completed', 'unique_Train_frames_seen']} for r in rows])))


if __name__ == '__main__':
    main()
