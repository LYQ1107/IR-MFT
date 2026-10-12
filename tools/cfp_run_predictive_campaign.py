#!/usr/bin/env python3
"""Run equal-budget predictive/memory association comparisons on frozen dev."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256,write_json,snapshot_sources


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--derived-gt-root',type=Path,required=True);p.add_argument('--trackeval-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--features-root',type=Path);p.add_argument('--execution-config',type=Path,default=Path('configs/cfp/plim_execution_v1.json'));args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    if plan['state']!='FROZEN' or plan['protocol_sha256']!=sha256(args.protocol) or plan['execution_config_sha256']!=sha256(args.execution_config) or len({j['run_id'] for j in plan['jobs']})!=len(plan['jobs']) or any(Path(j['run_id']).name!=j['run_id'] for j in plan['jobs']):raise ValueError('Declared unique predictive campaign required')
    def run(job):
        out=args.assets_root/job['run_id'];evaluation=out.with_name(out.name+'_evaluation')
        command=[sys.executable,'-u','tools/cfp_replay_predictor.py','--detections',str(args.assets_root.parent/'phase0/causal_v3_dev_detections'),'--features-root',str(args.features_root or args.assets_root.parent/'phase1/causal_dev_detection_features_postneck_seed0'),'--protocol',str(args.protocol),'--config','configs/cfp/plim_trial_v1.json','--execution-config',str(args.execution_config),'--memory',job['memory'],'--appearance-weight',str(job['appearance_weight']),'--output',str(out)]
        if job.get('write_policy_config'):command+=['--write-policy-config',job['write_policy_config']]
        if job.get('save_write_audit'):command+=['--save-write-audit']
        if job.get('checkpoint'):command+=['--checkpoint',str(args.assets_root/job['checkpoint'])]
        if job.get('second_rgb_features_root'):command+=['--second-rgb-features-root',job['second_rgb_features_root']]
        evaluate=[sys.executable,'-u','tools/cfp_evaluate.py','--data-root',str(args.data_root),'--protocol',str(args.protocol),'--partition','dev','--derived-gt-root',str(args.derived_gt_root),'--trackeval-root',str(args.trackeval_root),'--predictions',str(out),'--output',str(evaluation)]
        if not out.exists():
            with out.with_name(out.name+'.log').open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        m=json.loads((out/'manifest.json').read_text())
        if m['seed']!=plan.get('seed',0) or m['status']!='COMPLETE' or m['memory']!=job['memory'] or m['appearance_weight']!=job['appearance_weight'] or m['protocol_sha256']!=sha256(args.protocol):raise ValueError('Existing predictor replay is incomplete or differs')
        if job.get('write_policy_config') and m['write_policy_config_sha256']!=sha256(job['write_policy_config']):raise ValueError('Write policy changed')
        if not job.get('write_policy_config') and m.get('write_policy_config_sha256') is not None:raise ValueError('Unexpected write policy')
        if bool(job.get('save_write_audit'))!=m.get('save_write_audit',False):raise ValueError('Write audit setting changed')
        if job.get('checkpoint') and sha256(args.assets_root/job['checkpoint'])!=m['predictor_checkpoint_sha256']:raise ValueError('Model checkpoint changed')
        if m.get('identity_representation','rgb')!=job.get('identity_representation','rgb'):raise ValueError('Campaign identity representation changed')
        if job.get('second_rgb_features_root') and m['secondary_feature_manifest_sha256']!=sha256(Path(job['second_rgb_features_root'])/'manifest.json'):raise ValueError('Second RGB feature source changed')
        if not evaluation.exists():
            with evaluation.with_name(evaluation.name+'.log').open('x') as stream:subprocess.run(evaluate,stdout=stream,stderr=subprocess.STDOUT,check=True)
        e=json.loads((evaluation/'metrics.json').read_text())
        for seq,v in m['videos'].items():
            if sha256(out/(seq+'.txt'))!=v['prediction_sha256'] or e['artifacts'][seq]['prediction_sha256']!=v['prediction_sha256']:raise ValueError('Replay/evaluation hash mismatch')
            if job.get('save_write_audit') and sha256(out/(seq+'_writes.jsonl'))!=v['write_audit_sha256']:raise ValueError('Write audit changed')
        if job.get('baseline_reference'):
            for seq in m['videos']:
                if (out/(seq+'.txt')).read_bytes()!=(args.assets_root/job['baseline_reference']/(seq+'.txt')).read_bytes():raise ValueError('Disabled write-policy replay differs from original fixed predictor')
        print(out.name,{k:e['metrics']['COMBINED_SEQ'][k] for k in ['HOTA','AssA','IDF1','IDSW']},flush=True)
        return out.name,dict(job=job,commands=[command,evaluate],metrics=e['metrics'],prediction_manifest_sha256=sha256(out/'manifest.json'),evaluation_sha256=sha256(evaluation/'metrics.json'),baseline_reference_verified=True if job.get('baseline_reference') else None)
    summary=dict(status='RUNNING',plan_sha256=sha256(args.plan),source_sha256=sha256(__file__),source_snapshots=snapshot_sources(args.output.with_suffix(''),[__file__,args.plan,args.execution_config]),runs={})
    write_json(args.output,summary)
    with ThreadPoolExecutor(2) as pool:
        for name,entry in pool.map(run,plan['jobs']):summary['runs'][name]=entry;write_json(args.output,summary)
    summary['status']='COMPLETE';write_json(args.output,summary)


if __name__=='__main__':main()
