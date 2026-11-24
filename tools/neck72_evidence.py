"""Load only completed seed0 neck72 vectors with the actual seed_0 layout."""
from pathlib import Path

import numpy as np

from tools.neck72_runtime import read, require_seed0
from tools.run_pseudo_ir_comparison import sha256


def load_neck72_evidence(data, work, split, sequence, config):
    require_seed0(config['native_seed'])
    if config.get('native_feature_family') != 'neck_metric72':
        raise ValueError('Only the72-epoch post-neck encoder family is registered here')
    if split not in {'dev', 'test'} or config['native_cue'] not in {'rgb', 'ir'}:
        raise ValueError('Invalid neck72 evidence split/cue')
    work = Path(work)
    family = config.get('native_eval_root', 'native_neck_eval_features')
    if family not in {'native_neck_eval_features', 'native_neck_seed0_eval_features'}:
        raise ValueError('Unknown completed neck72 observation family')
    folder = work / family / split / sequence
    arm, metadata = config['detection_arm'], {}

    def load(cue, key):
        complete_path = folder / cue / 'seed_0/complete.json'
        complete = read(complete_path)
        if (complete['status'] != 'COMPLETE_NATIVE_CURRENT_FEATURES' or complete['GT_read']
                or complete['future_frame_input'] or complete['selected_encoder_seeds'] != [0]
                or complete['split'] != split or complete['sequence'] != sequence or complete['cue'] != cue):
            raise ValueError('Expected a complete, current, seed0-only neck72 producer')
        info = complete['outputs'][arm]
        path = Path(info['path'])
        if sha256(path) != info['sha256']:
            raise ValueError('Completed neck72 feature bytes changed')
        with np.load(path) as source:
            for name in ['frame_ids', 'offsets', 'boxes']:
                np.testing.assert_array_equal(source[name], data[name])
            if any(name.startswith('native_s') and name != 'native_s0' for name in source.files):
                raise ValueError('New neck72 execution must not load seed1/2 arrays')
            vectors = source[key].copy()
        if vectors.shape != (len(data['boxes']), 512) or not np.isfinite(vectors).all():
            raise ValueError('Malformed neck72 observation vectors')
        metadata[cue] = dict(path=str(path), sha256=info['sha256'],
                             complete_sha256=sha256(complete_path), checkpoints=complete['checkpoints'])
        return vectors, complete

    reference, _ = load('rgb', 'reference_post')
    data['native_reference'] = reference
    if not config.get('disable_native_increment', False):
        cue = config['native_cue']
        vectors, complete = load(cue, 'native_s0')
        trained_path = work / 'native_neck_train_features' / (cue + '_s0/complete.json')
        trained = read(trained_path)
        if (trained['status'] != 'COMPLETE_TRAIN_ONLY_NATIVE_FEATURES'
                or complete['checkpoints']['native_s0']['sha256'] != trained['checkpoint_sha256']):
            raise ValueError('neck72 Train and evaluation encoder endpoints differ')
        alignment_path = work / 'native_neck_alignment/complete.json'
        alignment = read(alignment_path)
        if (alignment['status'] != 'COMPLETE_TRAIN_ONLY_ORTHOGONAL_ALIGNMENT'
                or alignment['Dev_rows'] or alignment['Test_rows']):
            raise ValueError('Expected the original Train-only neck72 alignment')
        info = alignment['outputs'][cue + '_s0']
        rotation_path = Path(info['path'])
        if sha256(rotation_path) != info['sha256']:
            raise ValueError('neck72 alignment bytes changed')
        with np.load(rotation_path) as source:
            rotation = source['rotation']
        if rotation.shape != (512, 512) or not np.isfinite(rotation).all():
            raise ValueError('Malformed neck72 alignment')
        if not config.get('disable_native_alignment', False):
            vectors = vectors @ rotation
        data['native_auxiliary'] = np.asarray(vectors, np.float32)
        metadata['alignment'] = dict(path=str(rotation_path), sha256=sha256(alignment_path),
            rotation_sha256=info['sha256'], applied=not config.get('disable_native_alignment', False))
    metadata.update(feature_family='neck_metric72', fixed_seed=0,
                    scope='Current/past crop encodings and Train-only map; no labels or future observations.')
    return metadata
