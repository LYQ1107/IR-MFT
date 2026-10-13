#!/usr/bin/env python3
"""Actual allnine three-seed centered/null controlled fits and native progress."""
import argparse,json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Newimmutable stage report required')
    config=Path('configs/cfp/phase_vii_centered_joint_probe_v2.json');c=json.loads(config.read_text());models={};summary={};digests={}
    for mode in c['architectures']:
        collection=[]
        for seed in c['seed_plan']:
            p=a.root/f'centered_joint_probe_{mode}_s{seed}_v2';m=json.loads((p/'manifest.json').read_text());g=m['guard']
            if m['status']!='COMPLETE' or g['architecture']!=mode or g['seed']!=seed or g['config_sha256']!=sha256(config) or sha256(p/'epoch8.pt')!=m['checkpoint_sha256'] or any(sha256(q)!=v for q,v in g['source_sha256'].items()):raise ValueError('Actualcomplete exactmodel/source/data guards required')
            assert len(m['epochs'])==8 and all(e['optimizer_updates']==400 and e['objective_means']['ranking_null']==0 for e in m['epochs'])
            hashes=[e['sampled_input_sha256'] for e in m['epochs']]
            if seed in digests:assert hashes==digests[seed]
            else:digests[seed]=hashes
            models[p.name]=dict(path=str(p),manifest_sha256=sha256(p/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],evidence=m);collection.append(m)
        summary[mode]={}
        for video in c['calibration_videos']+c['outer_videos']:
            summary[mode][video]={}
            for route in ['full_id','baseline_gru','centered_joint']:
                fields=['accuracy_percent','existing_accuracy_percent','null_recall_percent'];summary[mode][video][route]={}
                for key in fields:
                    values=[m['evaluations'][video][route][key] for m in collection];summary[mode][video][route][key]=dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))
    uniform=a.root/'centered_joint_uniform_native_v2/manifest.json';u=json.loads(uniform.read_text());assert u['status']=='COMPLETE' and all(v['baseline_byte_equal'] and v['joint_counts']['changed_edges']==0 for v in u['videos'].values())
    restart=a.root/'centered_joint_exact_restart_v2.json';rr=json.loads(restart.read_text());assert rr['status']=='EXACT_CENTERED_JOINT_RESTART_VERIFIED' and not rr['differences']
    native_state=a.root/'centered_joint_native_driver_v2.json';n=json.loads(native_state.read_text());complete=[]
    for row in n['completed']:
        p=Path(row['official_metrics']);assert sha256(p)==row['metrics_sha256'];complete.append(row)
    write_json(a.output,dict(status='IN_PROGRESS',completed_stage='COMPLETE_CENTERED_JOINT_NINE_FITS_AND_UNIFORM_PARITY_REAL_RESTART',config=c,config_sha256=sha256(config),models=models,summary=summary,completed_fits=9,successful_optimizer_updates=28800,all_matched_inputs_and_no_null_ranking_loss=True,uniform_native=dict(manifest_sha256=sha256(uniform),evidence=u),actual_restart=dict(sha256=sha256(restart),evidence=rr),native_completed_at_snapshot=complete,native_driver=str(native_state),native_total=18,inner_native_graph=dict(sha256=sha256(a.root/'centered_joint_inner_native_graph_v2.json'),evidence=json.loads((a.root/'centered_joint_inner_native_graph_v2.json').read_text())),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,config]),scope='Original512conditional warmsets only; inheritedencoder/GRU exposure. All9sameinput fits complete; native onlycompleted rowscount. CrossTransformer replaces self-attention by exactlyparameter-matched activeFFNs; independentpair has loweractivecapacity. No new384/allfoldC/nativepositive/rawFPS/novelty/finalgoal claim. Originalv1failures retained.'))
    print(json.dumps(dict(completed_fits=9,native_completed=len(complete),native_total=18)))

if __name__=='__main__':main()
