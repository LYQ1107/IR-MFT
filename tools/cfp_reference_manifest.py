#!/usr/bin/env python3
"""Pin inspected external sources without vendoring their implementation."""
import argparse
import ast
from pathlib import Path
import subprocess

from cfp.provenance import sha256, write_json


SOURCES = {
    'SU-T': ['tools/train.py', 'exps/SU-T-ReID.py', 'trackers/SUT_UKF.py', 'trackers/SUT_reid_UKF.py', 'trackers/association.py', 'trackers/UKF.py', 'fast_reid/fast_reid_interfece.py'],
    'TIDE': ['src/train.py', 'src/lib/models/tidel.py', 'src/lib/models/tides.py', 'src/lib/models/yolo.py', 'src/lib/models/networks/fastvit.py', 'src/lib/trains/mot.py', 'src/lib/datasets/dataset/jde.py', 'src/lib/tracker/matching.py', 'src/lib/tracker/mc_bot_sort.py'],
    'M4FT': ['tools/train.py', 'exps/M4FT_exps.py', 'yolox/models/m4ft.py', 'yolox/models/yolox.py', 'yolox/tracker/byte_tracker.py'],
    'DM3T': ['src/main.py', 'src/lib/trainer.py', 'src/lib/model/networks/dla_rgbt_main.py', 'src/lib/model/networks/tf_fusion_refined.py', 'src/lib/dataset/generic_dataset_rgbt.py', 'src/lib/utils/tracker.py'],
    'TFMFT': ['tf_track.py', 'tracker_multi.py', 'eval_motchallenge.py'],
    'PFTrack': ['src/main.py', 'src/lib/trainer.py', 'src/lib/model/networks/dla_V3_M.py', 'src/lib/model/networks/transformer_fusion_v3.py', 'src/lib/model/networks/base_model.py', 'src/lib/dataset/generic_dataset_rgbt.py', 'src/lib/utils/tracker.py'],
    'MOTIP': ['models/motip/id_decoder.py', 'models/motip/id_criterion.py', 'train.py', 'docs/GET_STARTED.md'],
    'TCEI': ['TCEI/tcei.py', 'submit_and_evaluate.py'],
    'GOT-JEPA': ['pytracking/ltr/models/tracking/tompnet_JEPA.py', 'pytracking/ltr/actors/tracking.py', 'pytracking/ltr/train_settings/tomp/GOT_JEPA_378_pretrain.py', 'pytracking/ltr/train_settings/tomp/GOT_JEPA_378_finetune.py'],
    'GOT-JEPA-paper': ['pytracking/ltr/models/tracking/tompnet_JEPA.py', 'pytracking/ltr/actors/tracking.py', 'pytracking/ltr/train_settings/tomp/GOT_JEPA_378_pretrain.py', 'pytracking/ltr/train_settings/tomp/GOT_JEPA_378_finetune.py'],
    'HybridSORT': ['tools/train.py', 'trackers/hybrid_sort_tracker/hybrid_sort_reid.py', 'trackers/hybrid_sort_tracker/association.py'],
    'BoT-SORT': ['tracker/bot_sort.py', 'tracker/matching.py', 'tracker/gmc.py', 'fast_reid/fast_reid_interfece.py'],
    'ByteTrack': ['tools/train.py', 'yolox/tracker/byte_tracker.py', 'yolox/models/yolo_head.py', 'yolox/data/datasets/mot.py'],
    'FastReID': ['tools/train_net.py', 'fastreid/modeling/meta_arch/baseline.py', 'fastreid/solver/build.py', 'fastreid/config/defaults.py', 'fastreid/data/samplers/triplet_sampler.py'],
    'V-JEPA2': ['app/vjepa/train.py', 'app/vjepa/utils.py', 'src/models/predictor.py', 'src/models/vision_transformer.py', 'src/datasets/video_dataset.py', 'src/masks/multiseq_multiblock3d.py', 'configs/train/vitl16/pretrain-256px-16f.yaml'],
    'DINOv3': ['dinov3/train/train.py', 'dinov3/train/ssl_meta_arch.py', 'dinov3/models/vision_transformer.py', 'dinov3/hub/backbones.py', 'dinov3/data/samplers.py', 'dinov3/configs/ssl_default_config.yaml'],
    'TrackEval': ['trackeval/eval.py', 'trackeval/datasets/mot_challenge_2d_box.py', 'trackeval/metrics/hota.py', 'trackeval/metrics/clear.py', 'trackeval/metrics/identity.py'],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = []
    for name, paths in SOURCES.items():
        root = args.reference_root / name
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        url = subprocess.check_output(['git', '-C', str(root), 'remote', 'get-url', 'origin'], text=True).strip()
        sources = {}
        for rel in paths:
            file = root / rel
            if not file.is_file():
                raise FileNotFoundError(file)
            entry = {'sha256': sha256(file)}
            if file.suffix == '.py':
                try:
                    tree = ast.parse(file.read_text(errors='replace'))
                    entry['classes'] = [x.name for x in ast.walk(tree) if isinstance(x, ast.ClassDef)]
                except SyntaxError as exc:
                    entry['syntax_error'] = str(exc)
            sources[rel] = entry
        licenses = {str(f.relative_to(root)): sha256(f) for f in root.glob('LICENSE*') if f.is_file()}
        for pattern in ('fast_reid/LICENSE*', 'pytracking/LICENSE*'):
            licenses.update({str(f.relative_to(root)): sha256(f) for f in root.glob(pattern) if f.is_file()})
        records.append({'name': name, 'url': url, 'commit': commit, 'sources': sources,
                        'license_files': licenses, 'availability': 'inspected source; training not reproduced'})
    write_json(args.output, {'schema_version': 1, 'references': records})
    print(f'Pinned {len(records)} source trees in {args.output}')


if __name__ == '__main__':
    main()
