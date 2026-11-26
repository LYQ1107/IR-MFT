import hashlib
import json
import numpy as np
import pytest
from tools.mft_candidate_risk_data import load_arrays,load_GT,qualify_Train


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays():
    cue=np.zeros((2,512),np.float16);cue[:,0]=1
    return dict(detections=np.asarray([[0,0,20,20,.9],[1,0,21,20,.8]],np.float32),
        rgb=cue,ir=cue.copy(),frame_ids=np.arange(1,4,dtype=np.int64),offsets=np.asarray([0,1,1,2],np.int64))


def test_compact_loader_keeps_empty_frame_and_declared_dtypes(tmp_path):
    path=tmp_path/'bank.npz';np.savez_compressed(path,**arrays())
    out=load_arrays(path,3,sha(path))
    np.testing.assert_array_equal(out['offsets'],[0,1,1,2])
    assert out['detections'].dtype==np.float32 and out['rgb'].dtype==np.float16
    with pytest.raises(AssertionError):load_arrays(path,3,'0'*64)
    changed=arrays();changed['frame_ids'][1]=99
    wrong=tmp_path/'wrong.npz';np.savez_compressed(wrong,**changed)
    with pytest.raises(AssertionError):load_arrays(wrong,3,sha(wrong))


def test_gt_reader_delivers_current_frame_only_with_empty_middle_frame(tmp_path):
    path=tmp_path/'gt.txt';path.write_text('3,7,0,0,20,20,1,1,1\n1,7,0,0,20,20,1,1,1\n')
    gt,bounds=load_GT(path,3,sha(path))
    assert bounds.tolist()==[0,1,1,2]
    assert gt[bounds[0]:bounds[1],0].tolist()==[1]
    assert gt[bounds[1]:bounds[2]].shape==(0,9)


def test_train_qualification_rejects_unclosed_producer_controller(tmp_path):
    base=tmp_path/'candidate_train_bank_20261127';base.mkdir()
    (base/'controller.execution.json').write_text(json.dumps(dict(actual_child_exit_code=None,source_unchanged=True)))
    (base/'state.json').write_text(json.dumps(dict(status='RUNNING',phases={})))
    with pytest.raises(AssertionError):qualify_Train(tmp_path,{})
