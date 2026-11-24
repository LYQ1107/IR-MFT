#!/usr/bin/env python3
"""CPU-only F4 mathematics/ROI checks and complete Dev zero-weight replay."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torchvision

from tools.fusion_memory_features import pool_p3
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig,FusionMemoryTracker

SPEC=ROOT/'configs/perceptual_fusion_track_memory_prototype_20261124.json'


class NoFeatures:
    def __array__(self,*args,**kwargs):raise AssertionError('Zero path accessed a feature')
    def __getitem__(self,key):raise AssertionError('Zero path accessed a feature')


def state_digest(core):
    digest=hashlib.sha256()
    for label,group in [('tracked',core.tracked_stracks),('lost',core.lost_stracks),('removed',core.removed_stracks)]:
        digest.update(label.encode());digest.update(np.asarray([len(group)],dtype='<i8').tobytes())
        for track in group:
            digest.update(np.asarray([track.track_id,track.state,track.frame_id,track.start_frame,
                track.tracklet_len,int(track.is_activated)],dtype='<i8').tobytes())
            digest.update(str(np.asarray(track.score).dtype).encode())
            digest.update(np.asarray([track.score],dtype='<f8').tobytes())
            for value in [track._tlwh,track.mean,track.covariance]:
                digest.update(np.asarray(value,dtype='<f8').tobytes())
    return digest.digest()


def append_tracks(digest,frame,tracks):
    count=0
    for track in tracks:
        x,y,w,h=map(float,track.tlwh)
        if w*h<=100:continue
        line=f'{frame},{track.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(track.score):.6f},-1,-1,-1\n'
        digest.update(line.encode());count+=1
    return count


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    specification=json.loads(SPEC.read_text());assert sha256(plan_path)==specification['main_plan_sha256']
    assert specification['status']=='PREREGISTERED_CONDITIONAL_F4_ENGINEERING_NOT_FORMAL_LAUNCH_READY'
    output=work/'fusion_memory_CPU_proof.json';assert not output.exists()
    require_host_memory(2*2**30,maximum_workers=1);require_storage(work)
    names=['yolox/tracker/fusion_memory_tracker.py','yolox/tracker/byte_tracker.py',
        'yolox/tracker/matching.py','yolox/tracker/kalman_filter.py','yolox/tracker/basetrack.py',
        'tools/fusion_memory_features.py','tools/probe_fusion_memory.py','tests/test_fusion_memory.py',
        str(SPEC.relative_to(ROOT)),str(plan_path.relative_to(ROOT)),plan['protocol']]
    sources={n:sha256(ROOT/n) for n in names};started=time.monotonic()
    log=work/'fusion_memory_CPU_pytest.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fusion_memory.py'],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    match=re.search(r'\b(\d+) passed\b',log.read_text());assert match and int(match.group(1))>=11
    x=(torch.arange(12,dtype=torch.float32)[None,:]+.5).expand(8,12)*8
    y=(torch.arange(8,dtype=torch.float32)[:,None]+.5).expand(8,12)*8
    pooled,valid=pool_p3(torch.stack([x,y])[None],np.array([[4,2,20,10]],np.float32),
                         (32,48),(64,96),2.,normalize=False)
    np.testing.assert_array_equal(pooled.numpy(),[[24.,12.]]);assert valid.all()
    source=Path(plan['frozen_detection_sources']['rgb_dev']);metadata=source/'metrics.json'
    original=json.loads(metadata.read_text());p=original['protocol']
    assert p['seed']==0 and p['variant']=='rgb' and p['split']=='dev'
    assert p['conf']==.1 and p['nms']==.7 and p['track_thresh']==.6 and p['match_thresh']==.9 and p['track_buffer']==30
    args=SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)
    rows=[];BaseTrack._count=0;total=0
    for sequence in protocol['dev']:
        path=source/'detections'/(sequence+'.npz');packet_sha=sha256(path)
        with np.load(path,allow_pickle=False) as value:
            data={k:value[k] for k in ['frame_ids','offsets','boxes','scores']}
        np.testing.assert_array_equal(data['frame_ids'],np.arange(1,len(data['frame_ids'])+1))
        assert data['offsets'][0]==0 and data['offsets'][-1]==len(data['boxes'])
        assert data['offsets'].shape==(len(data['frame_ids'])+1,) and (np.diff(data['offsets'])>=0).all()
        def current(frame):
            lo,hi=map(int,data['offsets'][frame-1:frame+1])
            return np.c_[data['boxes'][lo:hi],data['scores'][lo:hi]].astype(np.float32)
        initial_id=BaseTrack._count
        baseline=BYTETracker(args,frame_rate=25);expected=[];baseline_mot=hashlib.sha256()
        trace=hashlib.sha256();baseline_rows=0
        for frame in range(1,len(data['frame_ids'])+1):
            tracks=baseline.update(current(frame),(1,1),(1,1))
            baseline_rows+=append_tracks(baseline_mot,frame,tracks)
            state=state_digest(baseline);expected.append(state);trace.update(state)
        after_reference=BaseTrack._count
        tracker=FusionMemoryTracker(FusionMemoryConfig(appearance_weight=0),args,identity_start=initial_id)
        actual_mot=hashlib.sha256();actual_trace=hashlib.sha256();actual_rows=0
        for frame in range(1,len(data['frame_ids'])+1):
            tracks=tracker.update(sequence,frame,current(frame),NoFeatures())
            state=state_digest(tracker.core);assert state==expected[frame-1],(sequence,frame)
            actual_trace.update(state);actual_rows+=append_tracks(actual_mot,frame,tracks)
        assert BaseTrack._count==after_reference,'Independent F4 must not change the global baseline ID counter'
        assert actual_mot.hexdigest()==baseline_mot.hexdigest() and actual_trace.hexdigest()==trace.hexdigest()
        published=source/'track_results'/(sequence+'.txt')
        assert published.exists() and sha256(published)==baseline_mot.hexdigest(),sequence
        assert sha256(path)==packet_sha and actual_rows==baseline_rows
        rows.append(dict(sequence=sequence,frames=len(data['frame_ids']),detection_packet=str(path),
            detection_packet_sha256=packet_sha,initial_identity_counter=initial_id,
            original_global_counter_after_reference=after_reference,output_rows=actual_rows,
            source_track_file=str(published),literal_MOT_sha256=actual_mot.hexdigest(),
            original_published_tracks_literal_exact=True,all_frame_tracked_lost_removed_numeric_states_exact=True,
            all_frame_state_trace_sha256=actual_trace.hexdigest(),features_accessed=False))
        total+=len(data['frame_ids']);print(json.dumps(dict(sequence=sequence,frames=len(data['frame_ids']),zero_exact=True)),flush=True)
    assert total==4834 and not torch.cuda.is_initialized()
    assert sources=={n:sha256(ROOT/n) for n in names};require_storage(work)
    roi_source=Path(inspect.getsourcefile(torchvision.ops.roi_align)).resolve()
    result=dict(status='COMPLETE_CPU_F4_ZERO_ALL4834_DEV_AND_ROI_CONTRACT_ENGINEERING',
        source_hashes=sources,specification_sha256=sha256(SPEC),tests_passed=int(match.group(1)),test_log_sha256=sha256(log),
        original_RGB_Dev_protocol_sha256=sha256(metadata),rows=rows,complete_Dev_frames=total,
        torchvision_version=torchvision.__version__,torch_version=torch.__version__,
        compiled_torchvision_ops=bool(torchvision.extension._has_ops()),
        ROI_operator_source=str(roi_source),ROI_operator_source_sha256=sha256(roi_source),
        known_ROI_mean=pooled.tolist(),ROI_expected_mean=[[24.,12.]],GPU_work=False,
        actual_learned_fusion_features_transferred=False,actual_current_IR_fusion_accuracy_validated=False,
        new_image_encoding=False,Train_or_Test_GT_read=False,new_Test_replay=False,
        new_accuracy_evaluation=False,formal_F4_model_fit=False,formal_F4_launch_ready=False,
        existing_source_MOT_files_unmodified=True,fixed_seed=0,additional_seed_work=False,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        seconds=time.monotonic()-started,
        scope='Complete two-video RGB Dev cached detection replay of the zero-appearance path, with exact original published MOT bytes and every tracked/lost/removed KF/numeric state under identical initial IDs. Enabled-memory and P3 tests use synthetic inputs only. No learned fusion feature was associated before visual validation; not F4 efficacy, new HOTA, model training, full Test coverage or innovation.',
        goal_met=False,supported_innovations=0)
    assert result['peak_process_RSS_GiB']<=8
    dump(output,result);print(result['status'],result['seconds'],'seconds',flush=True)


if __name__=='__main__':main()
