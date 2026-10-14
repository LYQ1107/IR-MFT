#!/usr/bin/env python3
"""All12 fits/24 native videos, exact coordinates/capacity/input sequences."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256, snapshot_sources, write_json


def stats(values):return dict(values=values,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable all-seed report required')
    config=Path('configs/cfp/phase_vii_dictionary_separated_joint_training_v1.json');c=json.loads(config.read_text())
    queries=a.root/'native_dino_joint_fit_queries_v1';q=json.loads((queries/'manifest.json').read_text())
    coordinate={v['coordinate_sha256'] for v in q['videos'].values()}
    if len(coordinate)!=1 or q['status']!='COMPLETE_FIT_ONLY_NATIVE_DINO_SET_QUERIES':raise ValueError('One exact frozen-source four-fit-video corpus required')
    coordinate=coordinate.pop()
    result=dict(status='COMPLETE_FIRST_FOLD_DICTIONARY_SEPARATED_JOINT_THREE_SEED_CONTROLS',config=c,config_sha256=sha256(config),
        query_manifest=str(queries/'manifest.json'),query_manifest_sha256=sha256(queries/'manifest.json'),coordinate_sha256=coordinate,
        source_teacher=q,models={},native_runs={},summary={},baseline={},
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,config]),
        scope='Twelve fixed-final actualCPUmodels/fourmodes/threeseeds, identicalactual sampledinputs andverifiedsubset RNG, twoencoder/head-excluded whole native videos=24runs. FrozenApacheDINO384samecoordinate, historicaldetector/privatefoundation overlapexplicit. No GT/teacher/augmentation/future inference. NotofficialMOTIP, pixelencoder/allfold/finalmethod/rawFPS/independentnovelty claim. Code/contentconditioning exactactivecapacity; pairactivecapacityexplicit.')
    fits={}
    for mode in c['model_specs']:
        fits[mode]=[]
        for seed in c['seed_plan']:
            path=a.root/f'dictionary_separated_joint_{mode}_s{seed}_v1';m=json.loads((path/'manifest.json').read_text());g=m['guard']
            if (m['status']!='COMPLETE' or g['mode']!=mode or g['seed']!=seed or g['config_sha256']!=sha256(config)
                    or g['query_manifest_sha256']!=sha256(queries/'manifest.json') or m['heldout_data_loaded'] or m['cuda_initialized']
                    or sha256(path/'final.pt')!=m['checkpoint_sha256'] or len(m['epochs'])!=8
                    or any(e['optimizer_updates']!=150 for e in m['epochs']) or any(sha256(p)!=h for p,h in g['source_sha256'].items())):
                raise ValueError('Exact complete actualfit/model/split/input/source required')
            fits[mode].append(m);result['models'][path.name]=dict(path=str(path),manifest_sha256=sha256(path/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],evidence=m)
    for seed in c['seed_plan']:
        expected=[v['sampled_input_sha256'] for v in fits['joint'][seed]['epochs']]
        if any([v['sampled_input_sha256'] for v in fits[mode][seed]['epochs']]!=expected for mode in fits):raise ValueError('Actualsame input/augmentation/codes sequence failed')
        if fits['joint'][seed]['active_parameters']!=fits['coded_joint'][seed]['active_parameters']:raise ValueError('Code/content activecapacity mismatch')
    for video in c['inner_videos']+c['outer_videos']:
        result['summary'][video]={}
        baseline=a.root/f'native_identity_frozen_{video}_fifo_s0_v1.trackeval/metrics.json'
        result['baseline'][video]=dict(path=str(baseline),sha256=sha256(baseline),metrics=json.loads(baseline.read_text())['metrics'][video])
        fm=a.root/f'native_dino_frozen_{video}_k4_v1/manifest.json';f=json.loads(fm.read_text())
        if f['encoder']['coordinate_sha256']!=coordinate or f['encoder']['trained_encoder'] is not None:raise ValueError('Heldoutencoder coordinate differs fromallfitting videos')
        for mode in c['model_specs']:
            values=[]
            for seed in c['seed_plan']:
                name=f'native_dictionary_joint_{mode}_{video}_s{seed}_v1';root=a.root/name;m=json.loads((root/'manifest.json').read_text());metrics=root.with_name(root.name+'.trackeval')/'metrics.json';official=json.loads(metrics.read_text())
                if (m['status']!='COMPLETE_NATIVE_DICTIONARY_JOINT_REPLAY' or m['online_GT_teacher_future_used']
                        or m['model_checkpoint_sha256']!=fits[mode][seed]['checkpoint_sha256'] or m['feature_manifest_sha256']!=sha256(fm)
                        or m['completed_frames']!=f['frames'] or m['frames']!=f['frames']
                        or sha256(root/(video+'.txt'))!=m['prediction_sha256'] or official['artifacts'][video]['prediction_sha256']!=m['prediction_sha256']):
                    raise ValueError('Actual complete source-matched heldout native result required')
                result['native_runs'][name]=dict(path=str(root),manifest_sha256=sha256(root/'manifest.json'),metrics_path=str(metrics),metrics_sha256=sha256(metrics),metrics=official['metrics'][video],source=m['sources'],model_checkpoint_sha256=m['model_checkpoint_sha256'])
                values.append(official['metrics'][video])
            result['summary'][video][mode]={k:stats([v[k] for v in values]) for k,v in values[0].items() if isinstance(v,(int,float))}
    result.update(completed_fits=len(result['models']),completed_native_videos=len(result['native_runs']),successful_optimizer_updates=12*1200,
                  exact_sampled_inputs_and_active_capacity_verified=True)
    write_json(a.output,result);print(json.dumps(dict(status=result['status'],fits=result['completed_fits'],native_videos=result['completed_native_videos'])))


if __name__=='__main__':main()
