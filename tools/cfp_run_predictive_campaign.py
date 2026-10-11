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
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--derived-gt-root',type=Path,required=True);p.add_argument('--trackeval-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    if plan['state']!='FROZEN' or plan['protocol_sha256']!=sha256(args.protocol) or len({j['run_id'] for j in plan['jobs']})!=len(plan['jobs']) or any(Path(j['run_id']).name!=j['run_id'] for j in plan['jobs']):raise ValueError('Declared unique predictive campaign required')
    def run(job):
        out=args.assets_root/job['run_id'];evaluation=out.with_name(out.name+'_evaluation')
        command=[sys.executable,'-u','tools/cfp_replay_predictor.py','--detections',str(args.assets_root.parent/'phase0/causal_v3_dev_detections'),'--features-root',str(args.assets_root.parent/'phase1/causal_dev_detection_features_postneck_seed0'),'--protocol',str(args.protocol),'--config','configs/cfp/plim_trial_v1.json','--execution-config','configs/cfp/plim_execution_v1.json','--memory',job['memory'],'--appearance-weight',str(job['appearance_weight']),'--output',str(out)]
        if job.get('checkpoint'):command+=['--checkpoint',str(args.assets_root/job['checkpoint'])]
        evaluate=[sys.executable,'-u','tools/cfp_evaluate.py','--data-root',str(args.data_root),'--protocol',str(args.protocol),'--partition','dev','--derived-gt-root',str(args.derived_gt_root),'--trackeval-root',str(args.trackeval_root),'--predictions',str(out),'--output',str(evaluation)]
        if not out.exists():
            with out.with_name(out.name+'.log').open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        m=json.loads((out/'manifest.json').read_text())
        if m['status']!='COMPLETE' or m['memory']!=job['memory'] or m['appearance_weight']!=job['appearance_weight'] or m['protocol_sha256']!=sha256(args.protocol):raise ValueError('Existing predictor replay is incomplete or differs')
        if job.get('checkpoint') and sha256(args.assets_root/job['checkpoint'])!=m['predictor_checkpoint_sha256']:raise ValueError('Model checkpoint changed')
        if not evaluation.exists():
            with evaluation.with_name(evaluation.name+'.log').open('x') as stream:subprocess.run(evaluate,stdout=stream,stderr=subprocess.STDOUT,check=True)
        e=json.loads((evaluation/'metrics.json').read_text())
        for seq,v in m['videos'].items():
            if sha256(out/(seq+'.txt'))!=v['prediction_sha256'] or e['artifacts'][seq]['prediction_sha256']!=v['prediction_sha256']:raise ValueError('Replay/evaluation hash mismatch')
        print(out.name,{k:e['metrics']['COMBINED_SEQ'][k] for k in ['HOTA','AssA','IDF1','IDSW']},flush=True)
        return out.name,dict(job=job,commands=[command,evaluate],metrics=e['metrics'],prediction_manifest_sha256=sha256(out/'manifest.json'),evaluation_sha256=sha256(evaluation/'metrics.json'))
    summary=dict(status='RUNNING',plan_sha256=sha256(args.plan),source_sha256=sha256(__file__),source_snapshots=snapshot_sources(args.output.with_suffix(''),[__file__,args.plan]),runs={})
    write_json(args.output,summary)
    with ThreadPoolExecutor(2) as pool:
        for name,entry in pool.map(run,plan['jobs']):summary['runs'][name]=entry;write_json(args.output,summary)
    summary['status']='COMPLETE';write_json(args.output,summary)


if __name__=='__main__':main()
