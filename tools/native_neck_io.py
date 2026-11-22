"""Read immutable completed native encoders without altering their fit state."""
import json
from pathlib import Path

import torch
from tools.extract_perceptual_mft_features import AppearanceEncoder
from tools.run_pseudo_ir_comparison import sha256

ROOT=Path(__file__).resolve().parents[1]


def load_native_encoder(cue,seed,work,protocol):
    config_path=ROOT/'configs/perceptual_native_neck_metric_20261123.json';config=json.loads(config_path.read_text())
    if cue=='reference':
        if seed!=0:raise ValueError('Only one frozen RGB reference')
        path=Path(config['initial_checkpoint'])
        if sha256(path)!=config['initial_sha256']:raise ValueError('Frozen RGB reference changed')
    else:
        directory=Path(work)/'native_neck_metric'/f'{cue}_s{seed}';completed=json.loads((directory/'complete.json').read_text())
        if completed['status']!='COMPLETE_FIXED36_NATIVE_NECK_METRIC_FIT' or completed['epochs']!=36 or completed.get('cumulative_native_epochs')!=72 or completed['Dev_fit_rows'] or completed['Test_fit_rows'] or completed['fit_sequences']!=protocol['train']:raise ValueError('Native encoder is not completely Train fitted')
        path=directory/'epoch36.pt'
        if not path.exists():raise ValueError('Wait for exact FP32 native export')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if cue!='reference' and (checkpoint['cue']!=cue or checkpoint['seed']!=seed or checkpoint['epoch']!=36 or checkpoint['config_sha256']!=sha256(config_path)):raise ValueError('Native encoder metadata mismatch')
    model=AppearanceEncoder();model.load_state_dict(checkpoint['model'],strict=True)
    return model.float().eval(),dict(path=str(path),sha256=sha256(path),cue=cue,seed=seed,precision='FP32 including BN state',cumulative_native_epochs=72 if cue!='reference' else 0)
