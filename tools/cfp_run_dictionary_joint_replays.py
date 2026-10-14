#!/usr/bin/env python3
"""All modes/seeds, both excluded native videos; actual official scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['root','state']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=['pair','cross_transformer','joint','coded_joint'],required=True);a=ap.parse_args()
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU-only native worker required')
    config=Path('configs/cfp/phase_vii_dictionary_separated_joint_training_v1.json');c=json.loads(config.read_text())
    replay=Path('configs/cfp/phase_vii_native_current_identity_replay_v1.json')
    paths=[__file__,'tools/cfp_replay_dictionary_joint.py',str(config),str(replay),'cfp/dictionary_separated_joint.py',
        'cfp/native_dictionary_joint.py','cfp/joint_identity_decoder.py','cfp/centered_joint_identity.py','cfp/association.py','cfp/native_local_identity.py','cfp/predictive.py','cfp/predictive_association.py']
    hashes={p:sha256(p) for p in paths};tasks=[(s,v) for s in c['seed_plan'] for v in c['inner_videos']+c['outer_videos']]
    state=dict(status='RUNNING',pid=os.getpid(),mode=a.mode,tasks=tasks,completed=[],active=None,source_guards=hashes,sources=snapshot_sources(a.state.with_suffix('.sources'),paths));write_json(a.state,state)
    try:
        for seed,video in tasks:
            if any(sha256(p)!=h for p,h in hashes.items()):raise ValueError('Active native source drift')
            model=a.root/f'dictionary_separated_joint_{a.mode}_s{seed}_v1';m=json.loads((model/'manifest.json').read_text())
            if m['status']!='COMPLETE':raise ValueError('Actual fixed-final model completion required')
            out=a.root/f'native_dictionary_joint_{a.mode}_{video}_s{seed}_v1';state['active']=dict(output=str(out));write_json(a.state,state)
            if out.exists():
                while True:
                    d=json.loads((out/'manifest.json').read_text())
                    if d['status']=='COMPLETE_NATIVE_DICTIONARY_JOINT_REPLAY':break
                    if d['status'].startswith('FAILED'):raise ValueError('Actual native producer failed; do not restart')
                    owned=[line for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines() if line.strip().split(None,1)[-1].startswith('/data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_replay_dictionary_joint.py ') and '--output '+str(out)+' ' in line]
                    if not owned:raise ValueError('Incomplete existing output has no actual live producer')
                    state['status']='WAITING_ACTUAL_NATIVE_PRODUCER';write_json(a.state,state);time.sleep(5)
            else:
                cmd=[sys.executable,'tools/cfp_replay_dictionary_joint.py','--config',str(replay),'--features',str(a.root/f'native_dino_frozen_{video}_k4_v1'),
                    '--detections','/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections','--output',str(out),'--video',video,'--model',str(model)]
                with Path(str(out)+'.log').open('x') as log:
                    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);state['active']['pid']=p.pid;state['status']='RUNNING';write_json(a.state,state);code=p.wait()
                if code:raise RuntimeError('Actual native child failed '+str(code))
                d=json.loads((out/'manifest.json').read_text())
            if d['status']!='COMPLETE_NATIVE_DICTIONARY_JOINT_REPLAY' or d['online_GT_teacher_future_used'] or d['model_checkpoint_sha256']!=m['checkpoint_sha256']:
                raise ValueError('Actual source-matched complete heldout replay required')
            evaluation=Path(str(out)+'.trackeval');evaluation.mkdir(exist_ok=True)
            protocol_path=Path('configs/cfp/phase_vii_joint_evaluation_subset_v1.json');protocol=json.loads(protocol_path.read_text())
            protocol.update(train=[video],dev=[],historical_test=[],role='Offline official scoring, fixed-final excluded native joint control, no threshold/seed selection')
            write_json(evaluation/'protocol.json',protocol)
            with (evaluation/'evaluation.log').open('x') as log:
                code=subprocess.run([sys.executable,'tools/cfp_evaluate.py','--data-root','/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
                    '--derived-gt-root','/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt','--protocol',str(evaluation/'protocol.json'),
                    '--partition','train','--predictions',str(out),'--output',str(evaluation),'--trackeval-root','/data3/liuyeqiang/CFP-FishMOT_references/TrackEval'],stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Actual official evaluator failed')
            metrics=json.loads((evaluation/'metrics.json').read_text())
            if metrics['artifacts'][video]['prediction_sha256']!=d['prediction_sha256']:raise ValueError('Scored output changed')
            state['completed'].append(dict(output=str(out),manifest_sha256=sha256(out/'manifest.json'),metrics_sha256=sha256(evaluation/'metrics.json'),metrics=metrics['metrics'][video]));state['active']=None;write_json(a.state,state)
        state['status']='COMPLETE_THREE_SEED_BOTH_HELDOUT_NATIVE_JOINT_CONTROLS';write_json(a.state,state)
    except Exception as error:
        state.update(status='HALTED_ACTUAL_FAILURE',error=str(error));write_json(a.state,state);raise


if __name__=='__main__':main()
