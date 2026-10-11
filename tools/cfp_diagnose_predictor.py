#!/usr/bin/env python3
"""Frozen predictor diagnosis on canonical dev; never fit or select a model here."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import torch

from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256,write_json,snapshot_sources

spec=importlib.util.spec_from_file_location('cfp_predictor_training_helper',Path(__file__).with_name('cfp_train_predictor.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def main():
    p=argparse.ArgumentParser();p.add_argument('--queries',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--query-config',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Predictive diagnosis outputs are immutable')
    metadata=json.loads((args.model_dir/'manifest.json').read_text());protocol=json.loads(args.protocol.read_text());ckpt=args.model_dir/'epoch8.pt'
    if metadata['status']!='COMPLETE' or metadata['config_sha256']!=sha256(args.config) or metadata['query_config_sha256']!=sha256(args.query_config) or metadata['protocol_sha256']!=sha256(args.protocol) or sha256(ckpt)!=metadata['checkpoint_sha256']:raise ValueError('Matched immutable predictor required')
    rows,features,query_meta=helper.load_queries(args.queries,args.config,args.query_config,'dev')
    if set(query_meta['videos'])!=set(protocol['dev']) or query_meta['protocol_sha256']!=sha256(args.protocol) or query_meta['encoder_seed']!=metadata['seed']:raise ValueError('This diagnosis is canonical dev only')
    train_fm=json.loads((Path(metadata['features_root'])/'manifest.json').read_text());dev_fm=json.loads((Path(query_meta['features_root'])/'manifest.json').read_text())
    if sha256(Path(metadata['features_root'])/'manifest.json')!=metadata['feature_manifest_sha256'] or train_fm['rgb_encoder']['checkpoint_sha256']!=dev_fm['rgb_encoder']['checkpoint_sha256']:raise ValueError('Stopped latent coordinate spaces differ')
    torch.set_num_threads(2);model=LatentIdentityPredictor(512,metadata['architecture']);checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=8:raise ValueError('Fixed epoch8 only')
    model.load_state_dict(checkpoint['model'],strict=True);args.output.mkdir(parents=True)
    report=dict(status='RUNNING',scope='Offline GT-supervised stopped-target diagnostic only, not IDF1/online scoring. No fitting or checkpoint selection. Physical clean/actual histories and current candidates come from frozen dev queries.',model_sha256=sha256(ckpt),query_manifest_sha256=sha256(args.queries/'manifest.json'),protocol_sha256=sha256(args.protocol),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_snapshots=snapshot_sources(args.output,[__file__,'tools/cfp_train_predictor.py','cfp/predictive.py','cfp/provenance.py',args.config,args.query_config,args.protocol]),videos={})
    write_json(args.output/'manifest.json',report)
    for seq in query_meta['videos']:
        selected=[r for r in rows if r['sequence']==seq];values={}
        for history in ['clean','actual']:
            values[history]={'model':helper.diagnose(model,selected,features,history,'cpu')}
            for mode in ['last','ema','fifo']:values[history][mode]=helper.diagnose(None,selected,features,history,'cpu',mode)
        report['videos'][seq]=values;write_json(args.output/'manifest.json',report)
    report['status']='COMPLETE';write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
