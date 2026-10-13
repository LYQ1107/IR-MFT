"""Collect small, hash-linked Phase VII summaries without copying arrays/weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    proofs = {}

    def read(name):
        path = args.assets / name
        raw = path.read_bytes()
        proofs[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    features = read('identity_two_crop_features_pilot_s0_v1/manifest.json')
    actual = read('identity_crop_retrieval_pilot_s0_v1.json')
    paired = read('identity_paired_crop_retrieval_pilot_s0_v1.json')
    crop_cache = read('identity_pilot_crop_cache_v1/manifest.json')
    result = {
        'status': 'COMPLETE_FIRST_FOLD_SEED0_SPARSE_VISUAL_PILOT',
        'research_status': 'IN_PROGRESS',
        'scope': 'Sparse every20frame residues1/2, six training videos. Four adapter and two actual backbone fits: outerSN015, innerBT001, other4fit; fixed last8epochs. Not full-fold/three-seed, full-video MOT, or raw online runtime.',
        'features': {k: features[k] for k in ('status', 'config', 'fish_checkpoint_sha256', 'imagenet_checkpoint_sha256', 'bytes', 'peak_gpu_allocated_bytes')},
        'feature_videos': {v: {k: x[k] for k in ('observations', 'gt', 'detector', 'numeric_parity', 'extraction_seconds')} for v, x in features['videos'].items()},
        'actual_crop_metrics': actual['videos'],
        'paired_crop_metrics': paired['videos'],
        'crop_cache': {k: x for k, x in crop_cache.items() if k not in ('videos', 'source_snapshots')},
        'training': {},
        'source_audit_complete': ['ByteTrack', 'BoT-SORT', 'DINOv2'],
        'proofs': proofs,
        'limitations': [
            'Inherited FishReID was fitted using all six fish videos; only diagnostic reference.',
            'Fresh ImageNet backbone initialization used for all six new fits. Private/external pretraining overlap is not certified zero.',
            'GT versus actual detector pools differ. Paired diagnostic alone controls identical known physical keys, queries and galleries; unknown-pool evaluation remains separate.',
            'K4/K8 are width strips of actual spatial maps, not anatomical annotations. Visibility is observable letterbox image extent, not occlusion supervision.',
            'All pilot features available. Later empty-view/unavailable-gallery helper repair does not change completed pilot scores; historical source snapshots are retained.',
            'New256 coordinates must not reuse old512 GRU. No newly trained representation online improvement is claimed.',
            'Full-state epoch checkpoints exist; actual interrupted-resume equivalence has not yet been demonstrated.',
            'DINOv2 random-initialization source smoke is not pretrained accuracy; all bounded weight transfers failed TLS.',
        ],
    }
    for family, modes in [('local_adapter', ['global', 'dual_global', 'simple', 'learned']), ('visual_identity', ['global', 'learned'])]:
        for mode in modes:
            name = f'{family}_{mode}_SN-015_s0_v1'
            d = read(name + '/manifest.json')
            if d['status'] != 'COMPLETE':
                raise RuntimeError(f'{name} not complete')
            result['training'][name] = {k: x for k, x in d.items() if k not in ('source_snapshots', 'classifier_keys')}
    result['dinov2_architecture_smoke'] = read('dinov2_small_inference_source_smoke_v1.json')
    result['dinov2_alternate_file_provenance'] = read('dinov2_small_hf_pinned_provenance_v1.json')
    result['proofs'] = proofs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'observations': sum(v['observations'] for v in result['feature_videos'].values()), 'training_runs': len(result['training']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
