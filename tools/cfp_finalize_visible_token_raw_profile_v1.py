#!/usr/bin/env python3
"""Verify and officially score completed visible-token strict raw profiles."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cfp.provenance import sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--protocol-template', type=Path, required=True)
    ap.add_argument('--trackeval-root', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--derived-gt-root', type=Path, required=True)
    ap.add_argument('--outputs', nargs='+', type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('Finalization must be CPU-only')
    if args.report.exists():
        raise ValueError('Final report already exists; refuse overwrite')
    template = json.loads(args.protocol_template.read_text())
    if template.get('state') != 'FROZEN' or args.video not in template.get('train', []):
        raise ValueError('Protocol template does not contain the declared train video')
    results = []
    for run in args.outputs:
        manifest_path = run / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE':
            raise ValueError(f'{run}: raw manifest is not complete: {manifest.get("status")}')
        if manifest.get('video') != args.video or manifest.get('cached_detection_or_feature_used_for_inference'):
            raise ValueError(f'{run}: wrong video or cached inference input')
        replay = Path(str(run) + '.actual_trace_replay.json')
        if replay.exists():
            raise ValueError(f'{replay}: refuse overwrite')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = ''
        subprocess.run([
            sys.executable, 'tools/cfp_verify_raw_output_trace.py',
            '--run', str(run), '--output', str(replay)
        ], check=True, env=env)
        replay_json = json.loads(replay.read_text())
        if replay_json.get('status') != 'EXACT_ACTUAL_RAW_OUTPUT_TRACE_REPLAY_VERIFIED':
            raise ValueError(f'{run}: independent replay did not verify')
        evaluation = Path(str(run) + '.trackeval')
        evaluation.mkdir()
        protocol = evaluation / 'protocol.json'
        # A raw output directory contains predictions for this outer video
        # only.  Keep the frozen protocol metadata and GT hashes, but scope
        # the official evaluator's train partition to the current video so
        # it never looks for sibling-video prediction files in this run.
        scoped_template = dict(template)
        scoped_template['train'] = [args.video]
        scoped_template['evaluation_scope'] = {
            'partition': 'train',
            'video': args.video,
            'source_protocol_sha256': sha256(args.protocol_template),
        }
        protocol.write_text(json.dumps(scoped_template, ensure_ascii=False, indent=2) + '\n')
        cmd = [
            sys.executable, 'tools/cfp_evaluate.py',
            '--data-root', str(args.data_root),
            '--derived-gt-root', str(args.derived_gt_root),
            '--protocol', str(protocol), '--partition', 'train',
            '--predictions', str(run), '--output', str(evaluation),
            '--trackeval-root', str(args.trackeval_root)
        ]
        subprocess.run(cmd, check=True, env=env)
        metrics_path = evaluation / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        artifact = metrics.get('artifacts', {}).get(args.video, {})
        if artifact.get('prediction_sha256') != manifest.get('prediction_sha256'):
            raise ValueError(f'{run}: TrackEval prediction hash does not match raw manifest')
        results.append({
            'run': str(run),
            'kind': manifest['kind'],
            'mode': manifest['visible_mode'],
            'seed': manifest['visible_seed'],
            'training_checkpoint': manifest['training_checkpoint'],
            'training_checkpoint_sha256': manifest['encoder']['visible_token_checkpoint_sha256'],
            'raw_manifest_sha256': sha256(manifest_path),
            'prediction_sha256': manifest['prediction_sha256'],
            'trace_sha256': manifest['actual_current_output_trace_sha256'],
            'trace_replay': str(replay),
            'trace_replay_sha256': sha256(replay),
            'trackeval_output': str(evaluation),
            'trackeval_protocol_sha256': sha256(protocol),
            'trackeval_metrics': str(metrics_path),
            'trackeval_metrics_sha256': sha256(metrics_path),
            'frames': manifest['frames'],
            'completed_frames': manifest['completed_frames'],
            'output_rows': manifest['output_rows'],
            'whole_raw_FPS': manifest['whole_raw_FPS'],
            'frame_latency_p50_p95_ms': manifest['frame_latency_p50_p95_ms'],
            'elapsed_with_validation_seconds': manifest['elapsed_with_validation_seconds'],
            'parent_peak_gpu_allocated_bytes': manifest['parent_peak_gpu_allocated_bytes'],
            'reference_parity': manifest['reference_parity'],
            'metrics': metrics['metrics'][args.video],
            'official_metrics_artifact': artifact,
            'online_guards': {
                'loaded_GT_annotation_labels': manifest['loaded_GT_annotation_labels'],
                'precomputed_IR_used_for_inference': manifest['precomputed_IR_used_for_inference'],
                'cached_detection_or_feature_used_for_inference': manifest['cached_detection_or_feature_used_for_inference'],
                'no_future_or_teacher_online': True,
                'current_rgb_only_encoder_input': True,
                'proposal_source': 'causal_v3_train_detections_repeated_current_rgb_prefix'
            }
        })
    write_json(args.report, {
        'status': 'COMPLETE_VISIBLE_TOKEN_STRICT_RAW_RUNS_VERIFIED',
        'video': args.video,
        'scope': 'Completed strict current-only RGB -> causal pseudo-IR -> V3 detector -> visible-token encoder -> FIFO lifecycle -> output, followed by independent CPU trace replay and official TrackEval. Results are per declared outer fold/seed/objective; aggregate all-fold or final-method claims require every registered fold.',
        'protocol_template': str(args.protocol_template),
        'protocol_template_sha256': sha256(args.protocol_template),
        'evaluation_protocol_scope': {'partition': 'train', 'video': args.video, 'scoped_protocol_per_run': True},
        'trackeval_root': str(args.trackeval_root),
        'trackeval_commit': json.loads((args.trackeval_root / 'README.md').read_text()).get('commit') if (args.trackeval_root / 'README.md').exists() else None,
        'runs': results,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'feature_banks_and_checkpoints_outside_git': True,
        'raw_outputs_outside_git': True,
        'conclusion': 'Strict online runtime evidence is measured without GT/future/cached inference inputs; compare masked_visible_tokens and visible_target only under the same SN-015 fold, proposals, lifecycle and TrackEval protocol.'
    })


if __name__ == '__main__':
    main()
