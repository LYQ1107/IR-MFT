#!/usr/bin/env python3
"""Fitting/inner-only diagnosis; an outer query request is rejected."""
import argparse
import json
import os
from pathlib import Path

import torch

from cfp.local_dense_data import LocalSceneBank
from cfp.local_dense_evaluation import evaluate_local_scenes
from cfp.local_dense_jepa import LocalDenseStudent
from cfp.local_dense_stage_controls import FixedLocalStage
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('training', 'features', 'index', 'legacy-features', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--videos', nargs='+', required=True)
    args = ap.parse_args()
    manifest = json.loads((args.training/'manifest.json').read_text())
    guard, config = manifest['guard'], manifest['config']
    if (args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or manifest['status']!='COMPLETE' or guard['mode']!='full'
            or any(v not in guard['fit']+[guard['inner']] for v in args.videos)
            or guard['outer'] in args.videos
            or sha256(args.training/'final.pt')!=manifest['checkpoint_sha256']
            or any(sha256(p)!=value for p, value in guard['source_sha256'].items())):
        raise ValueError('Completed exact fixed full model and fitting/inner-only CPU diagnostic required; outer forbidden')
    torch.set_num_threads(2)
    bank = LocalSceneBank(args.features, guard['local_count'], args.index, args.legacy_features)
    if sha256(args.features/'manifest.json')!=guard['feature_manifest_sha256']:
        raise ValueError('Exact observed crop source required')
    state = torch.load(args.training/'final.pt', map_location='cpu', weights_only=False)
    model = LocalDenseStudent(len(manifest['classifier_keys']), guard['mode'], guard['local_count'],
                             config['dimension'], config['attention_heads'])
    model.load_state_dict(state['student'], strict=True)
    model.eval()
    report = dict(status='RUNNING', training_manifest_sha256=sha256(args.training/'manifest.json'),
        training_checkpoint_sha256=manifest['checkpoint_sha256'], videos={}, outer=guard['outer'],
        prohibited_outer_read=True, sources=snapshot_sources(args.output/'sources', [__file__,
            'cfp/local_dense_stage_controls.py', 'cfp/local_dense_jepa.py', 'cfp/local_dense_evaluation.py',
            'cfp/local_dense_data.py', 'cfp/identity_evidence.py', 'cfp/provenance.py']),
        scope='Same fixed learned current candidate encoder and relation scorer, same histories/query/galleries/support. Compare encoder-only FIFO, temporal context pool and regional prediction. No fitting/new hyperparameter or outer query read, no native/MOT causal claim.')
    write_json(args.output/'manifest.json', report)
    for video in args.videos:
        report['videos'][video] = {}
        for stage in ('encoder_fifo', 'context_pool', 'prediction'):
            wrapper = FixedLocalStage(model, stage)
            measured = evaluate_local_scenes(wrapper, bank, video, config, args.output/stage/'query_support')
            report['videos'][video][stage] = measured
            write_json(args.output/'manifest.json', report)
            print(video, stage, 'GT history actual detector gap25', measured['protocol']['gt']['detector']['25']['retrieval'], flush=True)
        supports = [report['videos'][video][s]['protocol']['gt']['detector']['25']['query_support']['sha256']
                    for s in ('encoder_fifo', 'context_pool', 'prediction')]
        if len(set(supports))!=1:
            raise ValueError('Changed stage evaluation support')
    report['status'] = 'COMPLETE_FITTING_INNER_LOCAL_EVIDENCE_STAGE_DIAGNOSIS'
    write_json(args.output/'manifest.json', report)


if __name__=='__main__':
    main()
