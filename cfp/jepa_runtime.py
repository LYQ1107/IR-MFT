"""Load only the hash-pinned student export in matching RGB coordinates."""
import json
from pathlib import Path
import torch
from .fish_jepa import FishJEPAStudent
from .recoverable_identity import RecoverableIdentityState
from .provenance import sha256


def load_jepa_student(path, seed, rgb_checkpoint_sha256):
    path=Path(path)
    metadata=json.loads(path.with_name('manifest.json').read_text())
    if metadata['status']!='COMPLETE' or sha256(path)!=metadata['inference_checkpoint_sha256'] or metadata['guard']['seed']!=seed or metadata['rgb_encoder']['checkpoint_sha256']!=rgb_checkpoint_sha256:
        raise ValueError('Completed student/export/encoder provenance differs')
    for source in ['cfp/fish_jepa.py','cfp/predictive.py']:
        if metadata['guard']['sources'][source]!=sha256(source):raise ValueError('Student inference implementation differs from training')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('epoch')!=8 or saved.get('teacher_removed') is not True or any(k in saved for k in ['teacher','optimizer']) or saved['guard']!=metadata['guard']:
        raise ValueError('Only declared last-epoch student exports enter inference')
    spec=metadata['spec']
    model=FishJEPAStudent(spec.get('architecture','gru'),spec.get('latent',128),RecoverableIdentityState() if spec.get('ris') else None,spec.get('freeze_encoder',False))
    model.load_state_dict(saved['model'],strict=True)
    return model.eval(),metadata
