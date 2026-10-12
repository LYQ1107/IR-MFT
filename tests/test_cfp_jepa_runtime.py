import json
import pytest
import torch
from cfp.fish_jepa import FishJEPAStudent
from cfp.jepa_runtime import load_jepa_student
from cfp.provenance import sha256


def exported(tmp_path, extra=None):
    model=FishJEPAStudent()
    path=tmp_path/'inference_epoch8.pt'
    guard={'seed':0,'sources':{s:sha256(s) for s in ['cfp/fish_jepa.py','cfp/predictive.py']}}
    torch.save(dict(epoch=8,model=model.state_dict(),teacher_removed=True,guard=guard,**(extra or {})),path)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='COMPLETE',inference_checkpoint_sha256=sha256(path),guard=guard,rgb_encoder={'checkpoint_sha256':'matched'},spec={})))
    return path


def test_export_loader_enforces_encoder_and_seed_coordinates(tmp_path):
    path=exported(tmp_path)
    model,metadata=load_jepa_student(path,0,'matched')
    assert not model.training and not any('teacher' in name for name in model.state_dict())
    with pytest.raises(ValueError):load_jepa_student(path,1,'matched')
    with pytest.raises(ValueError):load_jepa_student(path,0,'different')


def test_teacher_and_optimizer_are_rejected_even_in_hash_verified_exports(tmp_path):
    for name in ['teacher','optimizer']:
        path=exported(tmp_path,{name:{}})
        with pytest.raises(ValueError):load_jepa_student(path,0,'matched')
