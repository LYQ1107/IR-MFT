"""Qualified compact Train candidate banks and ordered frame-only supervision."""
from pathlib import Path
import numpy as np
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.mft_candidate_risk_training_stream import TrainEdgeCollector
from tools.native_pyramid_stream_tracks import mot_line


def qualify_Train(work,protocol):
    work=Path(work);base=work/'candidate_train_bank_20261127'
    execution=read(base/'controller.execution.json');state=read(base/'state.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']
    assert identity(execution['identity']['PID'])is None
    assert state['status']=='COMPLETE_TRAIN_BANK_READY_NO_NEW_MODEL_OR_ACCURACY'
    jobs=[j for rows in state['phases'].values()for j in rows.values()]
    assert len(jobs)==8 and all(j['status']=='COMPLETE'and type(j.get('exit_code'))is int
        and j['exit_code']==0 and j['argv_identity_observed']and identity(j['pid'])is None for j in jobs)
    complete=read(base/'complete_Train.json')
    assert complete['frames']==27251 and complete['sequences']==6 and len(complete['rows'])==6
    assert complete['all8_related_GPU_jobs_actual_exit_zero']and not complete['GT_labels_parsed_by_producers']
    assert set(protocol['train'])==set(LENGTHS['train'])
    rows={r['sequence']:r for r in complete['rows']}
    assert set(rows)==set(LENGTHS['train'])
    result={}
    for sequence,frames in LENGTHS['train'].items():
        meta=read(base/'cache'/(sequence+'.json'))
        assert meta==rows[sequence]and meta['split']=='train'and meta['frames']==frames
        assert not meta['GT_labels_parsed']and meta['all_model_weights_and_BN_unchanged']
        assert meta['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
        path=Path(meta['path'])
        assert path==base/'cache'/(sequence+'.npz')and path.stat().st_size==meta['bytes']and sha256(path)==meta['sha256']
        gt=Path(protocol['derived_gt_root_hint'])/'train'/sequence/'gt/gt.txt'
        expected=protocol['derived_gt_sha256']['train/'+sequence]
        assert sha256(gt)==expected
        result[sequence]=dict(bank_path=str(path),bank_sha256=meta['sha256'],frames=frames,
            GT_path=str(gt),GT_sha256=expected,bank_metadata_path=str(base/'cache'/(sequence+'.json')))
    assert sum(r['frames']for r in result.values())==27251
    return result


def load_arrays(path,frames,expected_sha256):
    assert sha256(path)==expected_sha256
    with np.load(path,allow_pickle=False)as data:
        assert set(data.files)=={'detections','rgb','ir','offsets','frame_ids'}
        arrays={k:data[k]for k in data.files}
    dets,rgb,ir=map(arrays.get,['detections','rgb','ir'])
    assert dets.dtype==np.float32 and dets.shape==(len(dets),5)
    assert rgb.dtype==ir.dtype==np.float16 and rgb.shape==ir.shape==(len(dets),512)
    assert all(np.isfinite(a).all()for a in [dets,rgb,ir])
    offsets=arrays['offsets'];frame_ids=arrays['frame_ids']
    assert offsets.dtype==frame_ids.dtype==np.int64 and offsets.shape==(frames+1,)
    assert offsets[0]==0 and offsets[-1]==len(dets)and(np.diff(offsets)>=0).all()
    np.testing.assert_array_equal(frame_ids,np.arange(1,frames+1,dtype=np.int64))
    assert sum(a.nbytes for a in arrays.values())<=int(1.2*2**30)
    return arrays


def load_GT(path,frames,expected_sha256):
    assert sha256(path)==expected_sha256
    gt=np.loadtxt(path,delimiter=',',ndmin=2)
    assert gt.ndim==2 and gt.shape[1]>=8 and np.isfinite(gt).all()
    assert (gt[:,0]>=1).all()and(gt[:,0]<=frames).all()and(gt[:,0]==gt[:,0].astype(np.int64)).all()
    gt=gt[np.argsort(gt[:,0],kind='stable')]
    boundaries=np.searchsorted(gt[:,0],np.arange(1,frames+2))
    return gt,boundaries


def stream_video(sequence,qualified,speed_upper,sink,*,progress=None):
    assert sequence in LENGTHS['train']and qualified['frames']==LENGTHS['train'][sequence]
    arrays=load_arrays(qualified['bank_path'],qualified['frames'],qualified['bank_sha256'])
    gt,boundaries=load_GT(qualified['GT_path'],qualified['frames'],qualified['GT_sha256'])
    collector=TrainEdgeCollector(speed_upper,sink)
    import hashlib
    digest=hashlib.sha256()
    for i,frame in enumerate(arrays['frame_ids']):
        lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,boundaries[i:i+2])
        rows=collector.step(sequence,int(frame),arrays['detections'][lo:hi],arrays['rgb'][lo:hi],
            arrays['ir'][lo:hi],gt[left:right],split='train')
        for row in rows:digest.update(mot_line(row).encode())
        if progress is not None and(int(frame)%500==0 or int(frame)==qualified['frames']):progress(int(frame))
    assert collector.statistics['frames']==qualified['frames']
    assert collector.statistics['proposals']==len(arrays['detections'])
    assert collector.history.frame==qualified['frames']
    return dict(sequence=sequence,frames=qualified['frames'],proposal_rows=len(arrays['detections']),
        statistics=collector.statistics,teacher_statistics=collector.teacher.statistics,
        original_teacher_MOT_sha256=digest.hexdigest(),GT_rows=len(gt),
        current_frame_only=True,GT_modifies_teacher_costs=False)
