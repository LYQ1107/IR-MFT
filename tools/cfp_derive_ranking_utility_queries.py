#!/usr/bin/env python3
"""Derive ranking-only targets from immutable complete-context OOF query arrays."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from cfp.utility_supervision import ranking_utility
from cfp.provenance import sha256,write_json,snapshot_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Derived utility targets are immutable')
    config=json.loads(args.config.read_text());parent=json.loads((args.source/'manifest.json').read_text())
    if config['state']!='FROZEN' or config['utility_target']!='candidate_ranking_only' or parent['status']!='COMPLETE' or parent['config_sha256']!=config['source_query_config_sha256'] or parent['partition']!='train':raise ValueError('Declared complete OOF query source required')
    args.output.mkdir(parents=True);sources=snapshot_sources(args.output,[__file__,'cfp/utility_supervision.py','cfp/provenance.py',args.config])
    report=dict(parent,status='RUNNING',config=config,config_sha256=sha256(args.config),derived_from=dict(root=str(args.source),manifest_sha256=sha256(args.source/'manifest.json')),source_snapshots=sources,source_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),videos={},scope='Same immutable complete candidate/history/evidence/physical targets and OOF teachers. Candidate ranking CE excludes NULL; absent true candidate or singleton target has zero identity-ranking utility. Prior association utility is retained as a separate diagnostic. Not a new modality information source.')
    write_json(args.output/'manifest.json',report)
    for sequence,v in parent['videos'].items():
        source=args.source/(sequence+'.npz')
        if sha256(source)!=v['sha256']:raise ValueError('Source query arrays changed')
        with np.load(source) as a:bank={k:a[k] for k in a.files}
        old=bank['utility'].copy();values=[];counts=dict(null=0,singleton_true=0,multi_true=0)
        for i,target in enumerate(bank['target']):
            lo,hi=bank['offsets'][i:i+2];n=int(hi-lo);target=int(target)
            counts['null' if target==n else 'singleton_true' if n==1 else 'multi_true']+=1
            values.append(ranking_utility(bank['rgb'][lo:hi],bank['ir'][lo:hi],target,config['teacher_temperature'],config['teacher_ir_weight']))
        bank['association_utility']=old;bank['utility']=np.array(values,np.float32);out=args.output/(sequence+'.npz');np.savez_compressed(out,**bank)
        report['videos'][sequence]=dict(queries=len(values),sha256=sha256(out),source_sha256=sha256(source),utility_mean=float(bank['utility'].mean()),utility_std=float(bank['utility'].std()),positive_fraction=float(np.mean(bank['utility']>0)),negative_fraction=float(np.mean(bank['utility']<0)),zero_fraction=float(np.mean(bank['utility']==0)),decision_strata=counts);write_json(args.output/'manifest.json',report)
    report['status']='COMPLETE';write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
