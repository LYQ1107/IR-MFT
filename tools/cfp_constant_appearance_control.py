#!/usr/bin/env python3
"""Blind all identity information to distinguish gating from appearance gains."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.source / 'manifest.json').read_text())
    if source['status'] != 'COMPLETE' or source['input_regime'] != 'causal':
        raise ValueError('Expected complete canonical features')
    args.output.mkdir(parents=True)
    manifest = dict(source, status='RUNNING', ir_encoder=None, auxiliary_view=None, videos={},
                    intervention='constant identical unit vector for every extractable RGB crop; no identity information',
                    source_manifest_sha256=sha256(args.source / 'manifest.json'),
                    source_sha256=sha256(__file__), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='Diagnostic identity-blind gating/acceptance control; source checkpoint is unused, not an appearance method',
                    latency_scope='Cached constant-vector control; no measured encoder inference')
    write_json(args.output / 'manifest.json', manifest)
    for sequence in source['videos']:
        path = args.source / (sequence + '.npz')
        meta = json.loads(path.with_suffix('.provenance.json').read_text())
        if meta['status'] != 'COMPLETE' or sha256(path) != meta['features_sha256']:
            raise ValueError('Source changed')
        values = np.load(path)
        feature = np.zeros_like(values['rgb'])
        feature[values['valid_rgb'], 0] = 1
        output = args.output / path.name
        np.savez_compressed(output, rgb=feature, valid_rgb=values['valid_rgb'])
        report = dict(manifest, status='COMPLETE', sequence=sequence, detection_sha256=meta['detection_sha256'],
                      features_sha256=sha256(output), detections=len(feature))
        write_json(output.with_suffix('.provenance.json'), report)
        manifest['videos'][sequence] = dict(features_sha256=report['features_sha256'], detections=len(feature))
    manifest['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
