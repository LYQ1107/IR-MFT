#!/usr/bin/env python3
"""Independently verify the first two completed current-IR full-Train fits."""
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    output = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_IR_TWO_COMPLETE_TRAIN_AUDIT.json'
    assert not output.exists()
    registry_paths = [work/'structural_full_train_pipeline_registered.json', work/'structural_content_guard_recovery_registered.json']
    for path in registry_paths:
        registration = json.loads(path.read_text())
        assert all(sha256(ROOT/p) == h for p, h in registration['source_hashes'].items())
    schedule_path = work/'structural_full_train_schedule/complete.json'
    schedule = json.loads(schedule_path.read_text())
    address_path = Path(schedule['address_file']['path'])
    assert sha256(address_path) == schedule['address_file']['sha256']
    with np.load(address_path) as value:
        addresses = value['addresses']
    assert addresses.shape == (72, 288, 4, 2)
    sequences, lengths = schedule['sequence_order'], schedule['lengths']
    assert sequences == protocol['train'] and sum(lengths.values()) == 27251
    state = json.loads((work/'structural_full_train_fit_state.json').read_text())
    rows, manifests = [], set()
    for role in ['ir_111_s0', 'ir_011_s0']:
        job = state['jobs'][role]
        assert job['status'] == 'COMPLETE' and not (Path('/proc')/str(job['pid'])).exists()
        folder = work/'structural_full_train_fit'/role
        complete_path = folder/'full_train_complete.json'
        complete = json.loads(complete_path.read_text())
        assert complete['status'] == 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
        assert complete['logical_Train_epochs_completed'] == 3 and complete['epochs'] == 72
        assert complete['successful_steps_this_invocation'] == 20736
        assert complete['seed'] == 0 and complete['Train_frames'] == 27251
        assert complete['full_frame_address_delivery_verified'] and complete['post_loop_source_verified']
        assert complete['Dev_fit_rows'] == 0 and not complete['Test_read']
        assert complete['frozen_original_encoder'] and complete['frozen_original_head']
        assert all(sha256(ROOT/p) == h for p, h in complete['source_hashes'].items())
        snapshot = (folder/'latest.pt').read_bytes()
        saved = torch.load(io.BytesIO(snapshot), map_location='cpu', weights_only=False)
        assert saved['completed_epochs'] == 72 and saved['history'] == complete['history']
        assert saved['metadata']['live_frame_passes'] == complete['live_frame_passes']
        assert len(complete['history']) == len(complete['live_frame_passes']) == 72
        assert sum(v.numel() for v in saved['adapter'].values()) == 145600
        assert all(bool(torch.isfinite(v).all()) for v in saved['adapter'].values())
        for index, (record, actual) in enumerate(zip(complete['history'], complete['live_frame_passes'])):
            digest = hashlib.sha256()
            for batch in addresses[index]:
                draw = [(sequences[int(i)], int(frame)) for i, frame in batch]
                digest.update(json.dumps(draw, separators=(',', ':')).encode())
            assert record['epoch'] == actual['pass_index'] == index+1
            assert record['steps'] == actual['prepared_batches'] == 288 and actual['complete_input_pass']
            assert actual['logical_Train_epoch'] == index//24+1
            assert digest.hexdigest() == record['sample_digest'] == actual['actual_frame_draw_sha256'] == actual['expected_frame_draw_sha256']
            assert np.isfinite(record['loss']).all()
        for epoch in range(3):
            block = addresses[epoch*24:(epoch+1)*24]
            for i, sequence in enumerate(sequences):
                np.testing.assert_array_equal(np.unique(block[..., 1][block[..., 0] == i]), np.arange(1, lengths[sequence]+1))
        manifest = complete['source_image_content_manifest']
        assert sha256(manifest['path']) == manifest['sha256']
        records = json.loads(Path(manifest['path']).read_text())
        expected = {(seq, frame, modality) for seq, n in lengths.items() for frame in range(1, n+1) for modality in ['rgb', 'ir']}
        assert len(records) == len(expected) == 54502
        assert {(r['sequence'], r['frame'], r['modality']) for r in records} == expected
        assert all(len(r['sha256']) == 64 and r['bytes'] > 0 for r in records)
        semantic = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert semantic == complete['content_guard_statistics']['content_manifest_sha256']
        manifests.add(manifest['sha256'])
        rows.append(dict(id=role, complete_file_sha256=sha256(complete_path),
            final_checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(), completed_macro_passes=72,
            complete_Train_traversals=3, optimizer_updates=20736, frame_visits=82944,
            unique_frames_per_traversal=27251, repeated_padding_visits_total=1191,
            full_72_actual_address_digests_exact=True, all3_traversals_complete=True,
            image_content_entries=54502, source_image_content_manifest=manifest,
            original_RGB_encoder_head_frozen=True, adapter_parameters=145600,
            successful_invocation_seconds=complete['seconds'], physical_gpu=complete['physical_gpu'],
            original_worker_exited=True, controller_exit_fields={k: job[k] for k in ['fit_exit_code', 'exit_code'] if k in job},
            no_new_image_or_FPN_disk_cache=complete['no_image_or_FPN_disk_cache']))
    assert len(manifests) == 1
    report = dict(status='COMPLETE_TWO_IR_THREE_TRAVERSAL_ADDRESS_AND_CONTENT_AUDIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
        common_image_manifest_sha256=next(iter(manifests)), both_fits_exact_same_original_RGB_and_November_IR_bytes=True,
        source_code_sha256=sha256(__file__), schedule_proof_sha256=sha256(schedule_path),
        existing_registrations={str(p): sha256(p) for p in registry_paths},
        completed_matched_fits=4, total_required_matched_fits=6,
        remaining_IR_fits={k: dict(status=state['jobs'][k]['status'], pid=state['jobs'][k].get('pid'),
            physical_gpu=state['jobs'][k]['physical_gpu']) for k in ['ir_101_s0', 'ir_110_s0']},
        scope='Actual consumed addresses and recorded first-read byte hashes during the two completed fits; '
            'not a fresh rehash of the entire external dataset, convergence or held-out accuracy. '
            'Successful invocation timing excludes preserved earlier failed attempts; six-fit pipeline is not complete.',
        GPU_work=False, Dev_Test_accuracy_read=False, new_HOTA=False, goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], common_manifest=next(iter(manifests)), complete_fits=4)), flush=True)


if __name__ == '__main__':
    main()
