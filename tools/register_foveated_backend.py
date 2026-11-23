#!/usr/bin/env python3
"""Freeze two completed Dev observation streams for the unchanged point backend.

The original feature extractor and tracker accept a plan path. Isolated input
aliases let both use the same tested code without changing any earlier producer.
Here `new` is only an input alias; explicit source roles retain its actual
global detector, attention modality, crop count, scale and fusion provenance.
"""
import argparse
import json
from pathlib import Path
import re

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['plain', 'regions', 'causal'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--ir-stream', required=True)
    p.add_argument('--rgb-stream', required=True)
    p.add_argument('--stronger-rgb-control', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.name):
        raise ValueError('Use a simple registration name')
    main_path = ROOT/'configs/perceptual_mft_20261122.json'
    plan = json.loads(main_path.read_text())
    work = Path(plan['work_root'])
    stage, producer, prefix, fovea = {
        'plain': ('foveated_dev', 'foveated_proposals', 'Z_', 'configs/perceptual_foveated_mft_20261123.json'),
        'regions': ('foveated_regions_dev', 'foveated_region_proposals', 'Y_', 'configs/perceptual_foveated_regions_20261123.json'),
        'causal': ('foveated_causal_dev', 'foveated_causal_proposals', 'X_', 'configs/perceptual_foveated_causal_20261123.json')
    }[a.family]
    expected_controls = 70 if a.family == 'causal' else 38
    state = json.loads((work/(stage+'_state.json')).read_text())
    if state['status'] != 'COMPLETE' or len(state['jobs']) != expected_controls:
        raise ValueError('Complete every registered Dev control before selection')
    match = re.fullmatch(r'(rgb|new)_(ir(?:_self|_physical|_consensus)?)_(half|quarter)_k(2|4)_(refine|nms)', a.ir_stream)
    if not match:
        raise ValueError('The IR stream must use actual IR attention')
    front, ir_mode, scale, count, fusion = match.groups()
    expected = f'{front}_rgb_{scale}_k{count}_{fusion}'
    if a.rgb_stream != expected and not a.stronger_rgb_control:
        raise ValueError('Use the exactly matched RGB control or explicitly register a stronger RGB control')
    if not a.rgb_stream.startswith(front+'_') or '_ir_' in a.rgb_stream:
        raise ValueError('Keep the same global frontend and an RGB-only attention control')
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    configs = ROOT/'configs/perceptual_foveated_backend_20261123'
    configs.mkdir(exist_ok=True)
    output = configs/(a.name+'.json')
    run = work/'foveated_backend_runs'/a.name
    if output.exists() or run.exists():
        raise ValueError('Registration already exists; do not overwrite')
    provenance = {}
    for alias, key, role in [('new', a.ir_stream, 'ACTUAL_NEW_IR_GUIDED_RGB_CROP_OBSERVATIONS'),
                              ('rgb', a.rgb_stream, 'RGB_GUIDED_CROP_CONTROL')]:
        metrics_path = work/stage/(prefix+key)/'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        if metrics['status'] != 'COMPLETE' or metrics['manifest']['frames'] != 4834:
            raise ValueError('Complete Dev metrics required')
        sources = {}
        directory = run/'inputs'/alias/'dev'/'detections'
        directory.mkdir(parents=True)
        for sequence in protocol['dev']:
            source_dir = work/producer/'dev'/sequence
            manifest = json.loads((source_dir/'complete.json').read_text())
            info = manifest['outputs'][key]
            source = Path(info['path'])
            if (manifest['GT_read'] or manifest['future_frame_input'] or info['frames'] < 1
                    or info['sha256'] != sha256(source)
                    or manifest['config_sha256'] != sha256(ROOT/fovea)):
                raise ValueError('Observation provenance mismatch')
            (directory/(sequence+'.npz')).symlink_to(source)
            sources[sequence] = dict(path=str(source), sha256=info['sha256'])
        provenance[alias] = dict(role=role, proposal_key=key, global_frontend=front, sources=sources,
                                  Dev_metrics_sha256=sha256(metrics_path), Dev_HOTA=metrics['trackeval']['HOTA'])
    derived = dict(plan)
    derived['work_root'] = str(run)
    derived['frozen_detection_sources'] = {
        prefix+'_'+split: str(run/'inputs'/alias/split)
        for alias, prefix in [('new', 'new_diffv2ir'), ('rgb', 'rgb')] for split in ['dev', 'test']}
    derived['foveated_backend_registration'] = dict(
        name=a.name, original_plan_sha256=sha256(main_path), foveation_plan=fovea,
        foveation_plan_sha256=sha256(ROOT/fovea), roles=provenance,
        alias_warning='new/rgb identify the declared IR-attended/RGB-control streams here; read global_frontend separately.',
        stronger_RGB_control=a.stronger_rgb_control, selection_split='dev', Test_started=False,
        current_goal_met=False, source_Dev_controls=expected_controls, IR_attention_mode=ir_mode,
        feature_extractor='tools/extract_perceptual_mft_features.py',
        feature_extractor_sha256=sha256(ROOT/'tools/extract_perceptual_mft_features.py'),
        point_extractor='tools/extract_foveated_correspondences.py',
        tracker='Unchanged original calibrated/foreground backend with original Train-fitted weights. No new fitting.')
    dump(output, derived)
    print(json.dumps(dict(status='REGISTERED_DEV_BACKEND', plan=str(output), work_root=str(run),
                          IR_stream=a.ir_stream, RGB_control=a.rgb_stream, Test_started=False)))


if __name__ == '__main__':
    main()
