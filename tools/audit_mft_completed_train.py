#!/usr/bin/env python3
"""Independent all-six completed address/content audit, no NN or held-out metrics."""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from tools.neck72_runtime_study import exited
from tools.run_pseudo_ir_comparison import dump,sha256

ROLES=('rgb_111_s0','gray_111_s0','ir_111_s0','ir_011_s0','ir_101_s0','ir_110_s0')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and not args.output.exists()
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    registrations={}
    for name in ['structural_full_train_pipeline_registered.json','structural_content_guard_recovery_registered.json']:
        p=work/name;registered=json.loads(p.read_text())
        assert all(sha256(ROOT/n)==s for n,s in registered['source_hashes'].items())
        registrations[str(p)]=sha256(p)
    schedule_path=work/'structural_full_train_schedule/complete.json';schedule=json.loads(schedule_path.read_text())
    address_path=Path(schedule['address_file']['path']);assert sha256(address_path)==schedule['address_file']['sha256']
    with np.load(address_path) as source:addresses=source['addresses']
    assert addresses.shape==(72,288,4,2)
    sequences,lengths=schedule['sequence_order'],schedule['lengths']
    assert sequences==protocol['train'] and sum(lengths.values())==27251
    expected_digests=[]
    for block in addresses:
        digest=hashlib.sha256()
        for batch in block:
            draw=[(sequences[int(i)],int(frame)) for i,frame in batch]
            digest.update(json.dumps(draw,separators=(',',':')).encode())
        expected_digests.append(digest.hexdigest())
    coverage=[]
    for epoch in range(3):
        block=addresses[epoch*24:(epoch+1)*24];counts={}
        for i,sequence in enumerate(sequences):
            frames=np.unique(block[...,1][block[...,0]==i]);np.testing.assert_array_equal(frames,np.arange(1,lengths[sequence]+1));counts[sequence]=len(frames)
        coverage.append(dict(logical_Train_epoch=epoch+1,per_video_unique_frames=counts,unique_frames=27251))
    state_path=work/'structural_full_train_fit_state.json';state=json.loads(state_path.read_text())
    assert set(state['jobs'])==set(ROLES) and all(j['status']=='COMPLETE' for j in state['jobs'].values())
    rows=[];ir_manifests=set();reference_sources=set()
    for name in ROLES:
        job=state['jobs'][name];assert exited(job['pid'],command=job['command'])
        folder=work/'structural_full_train_fit'/name;complete_path=folder/'full_train_complete.json';complete=json.loads(complete_path.read_text())
        assert sha256(complete_path)==job['fit_complete_sha256']
        assert complete['status']=='COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
        assert complete['logical_Train_epochs_completed']==3 and complete['epochs']==72
        assert complete['successful_steps_this_invocation']==20736 and complete['seed']==0 and complete['Train_frames']==27251
        assert complete['cue']==job['cue'] and complete['variant']==job['components']
        assert complete['full_frame_address_delivery_verified'] and complete['post_loop_source_verified']
        assert complete['Dev_fit_rows']==0 and not complete['Test_read']
        assert complete['frozen_original_encoder'] and complete['frozen_original_head']
        assert all(sha256(ROOT/p)==s for p,s in complete['source_hashes'].items())
        raw=(folder/'latest.pt').read_bytes();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        assert saved['completed_epochs']==72 and saved['history']==complete['history']
        assert saved['metadata']['live_frame_passes']==complete['live_frame_passes']
        assert len(complete['history'])==len(complete['live_frame_passes'])==72
        assert sum(v.numel() for v in saved['adapter'].values())==145600
        assert all(bool(torch.isfinite(v).all()) for v in saved['adapter'].values())
        for index,(record,actual) in enumerate(zip(complete['history'],complete['live_frame_passes'])):
            assert record['epoch']==actual['pass_index']==index+1
            assert record['steps']==actual['prepared_batches']==288 and actual['complete_input_pass']
            assert actual['logical_Train_epoch']==index//24+1
            assert expected_digests[index]==record['sample_digest']==actual['actual_frame_draw_sha256']==actual['expected_frame_draw_sha256']
            assert np.isfinite(record['loss']).all()
        manifest=complete.get('source_image_content_manifest');content_entries=None
        if job['cue']=='ir':
            assert manifest is not None and sha256(manifest['path'])==manifest['sha256']
            records=json.loads(Path(manifest['path']).read_text())
            expected={(s,f,m) for s,n in lengths.items() for f in range(1,n+1) for m in ['rgb','ir']}
            assert len(records)==len(expected)==54502
            assert {(v['sequence'],v['frame'],v['modality']) for v in records}==expected
            assert all(len(v['sha256'])==64 and v['bytes']>0 for v in records)
            semantic=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert semantic==complete['content_guard_statistics']['content_manifest_sha256']
            ir_manifests.add(manifest['sha256']);content_entries=54502
        reference_sources.add(complete['source_RGB_sha256'])
        rows.append(dict(id=name,physical_gpu=complete['physical_gpu'],original_worker_exited=True,
            controller_recorded_exit_codes={k:job[k] for k in ['fit_exit_code','exit_code'] if k in job},
            absent_exit_codes_remain_unknown=True,complete_file_sha256=sha256(complete_path),
            final_checkpoint_snapshot_sha256=hashlib.sha256(raw).hexdigest(),completed_macros=72,
            complete_Train_traversals=3,optimizer_updates=20736,frame_visits=82944,unique_frames_per_traversal=27251,
            repeated_padding_visits=1191,all72_actual_address_digests_exact=True,all3_full_frame_coverages_exact=True,
            content_manifest=manifest,content_entries=content_entries,
            byte_content_scope='Original first-read hashes during content-verified fitting' if manifest else 'Original RGB/gray reader recorded metadata only; no retroactive training-byte certification',
            original_RGB_encoder_head_frozen=True,adapter_parameters=145600,
            successful_invocation_seconds=complete['seconds'],successful_invocation_peak_gpu_GiB=complete['peak_gpu_GiB'],
            timing_excludes_preserved_earlier_failed_attempts=True,no_new_image_or_FPN_disk_cache=complete['no_image_or_FPN_disk_cache']))
    assert len(ir_manifests)==len(reference_sources)==1
    result=dict(status='COMPLETE_ALL_SIX_MATCHED_THREE_TRAVERSAL_ADDRESS_AND_FOUR_IR_CONTENT_AUDIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,coverage_each_fit=coverage,
        common_IR_image_manifest_sha256=next(iter(ir_manifests)),all4_IR_used_exact_same_recorded_RGB_and_November_IR_bytes=True,
        RGB_gray_training_byte_hashes_not_recorded=True,gray_worker_numeric_exit_code_not_invented=True,
        common_frozen_RGB_source_sha256=next(iter(reference_sources)),source_code_sha256=sha256(__file__),
        source_fit_state_sha256=sha256(state_path),schedule_proof_sha256=sha256(schedule_path),
        existing_registrations=registrations,completed_matched_fits=6,total_required_matched_fits=6,
        total_successful_optimizer_updates=124416,total_frame_visits=497664,full_Train_traversals_across_all_fits=18,
        scope='All72 actual delivered address digests perfit and complete three-frame-set traversals. '
            'FourIR first-read byte manifests identical,not fresh externaldataset rehash or retrofitRGBgraybyteproof. '
            'Missing historicalgray numeric exit remainsunknown. Successful invocation costs exclude earlier preservedfailures. '
            'No convergence,completeDev/TestorHOTA/innovationclaim.',new_GPU_work=False,new_weights_or_cache=False,
        Dev_Test_accuracy_read=False,new_HOTA=False,goal_met=False,supported_innovations=0)
    dump(args.output,result)
    print(json.dumps(dict(status=result['status'],complete_fits=6,updates_each=20736,
        common_IR_manifest=result['common_IR_image_manifest_sha256'],new_HOTA=False)),flush=True)


if __name__=='__main__':main()
