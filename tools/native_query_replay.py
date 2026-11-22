"""Verified loading of current-image identity vectors for frozen replay."""
import json
from pathlib import Path

import numpy as np

from tools.run_pseudo_ir_comparison import sha256


def load_native_query_evidence(data, work, split, sequence, config):
    work = Path(work)
    arm, cue, seed = config['detection_arm'], config['native_cue'], int(config['native_seed'])
    folder = work/'native_eval_features'/split/sequence
    metadata = {}
    def load(which, keys):
        manifest = json.loads((folder/which/'complete.json').read_text())
        info = manifest['outputs'][arm]
        path = Path(info['path'])
        if (manifest['status'] != 'COMPLETE_NATIVE_CURRENT_FEATURES' or manifest['GT_read']
                or manifest['future_frame_input'] or sha256(path) != info['sha256']):
            raise ValueError('Native observation provenance mismatch')
        with np.load(path) as source:
            for key in ['frame_ids', 'offsets', 'boxes']:
                if not np.array_equal(source[key], data[key]):
                    raise ValueError('Native observation/proposal indexing mismatch: '+key)
            result = {key: source[key].copy() for key in keys}
        metadata[which] = dict(path=str(path), sha256=info['sha256'], checkpoints=manifest['checkpoints'])
        return result, manifest
    reference, _ = load('rgb', ['reference_post'])
    data['native_reference'] = reference['reference_post']
    if not config.get('disable_native_increment', False):
        key = f'native_s{seed}'
        native, manifest = load(cue, [key])
        trained = json.loads((work/'native_train_features'/f'{cue}_s{seed}'/'complete.json').read_text())
        if manifest['checkpoints'][key]['sha256'] != trained['checkpoint_sha256']:
            raise ValueError('Native training/evaluation encoder mismatch')
        path = work/'native_alignment'/f'{cue}_s{seed}.npz'
        alignment_manifest = work/'native_alignment/complete.json'
        alignment = json.loads(alignment_manifest.read_text())
        if (alignment['status'] != 'COMPLETE_TRAIN_ONLY_ORTHOGONAL_ALIGNMENT'
                or alignment['Dev_rows'] or alignment['Test_rows']
                or sha256(path) != alignment['outputs'][f'{cue}_s{seed}']['sha256']):
            raise ValueError('Native alignment provenance mismatch')
        with np.load(path) as source:
            rotation = source['rotation']
        vectors = native[key]
        if not config.get('disable_native_alignment', False):
            vectors = vectors @ rotation
        data['native_auxiliary'] = np.asarray(vectors, np.float32)
        metadata['alignment'] = dict(path=str(path), sha256=sha256(alignment_manifest), rotation_sha256=sha256(path),
                                     applied=not config.get('disable_native_alignment', False))
    metadata['scope'] = 'Current/past crop encodings only; proposal frames preserved; no GT.'
    return metadata
