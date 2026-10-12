#!/usr/bin/env python3
"""Collect measured JEPA variants and seeded official tracking comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256, write_json


METRICS=['HOTA','AssA','DetA','LocA','AssRe','AssPr','DetRe','DetPr','IDF1','IDP','IDR','MOTA','IDSW','CLR_FP','CLR_FN','Frag','MT','ML']


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    inputs={};models={};runs={};groups={};complete=True
    for path in args.campaign:
        campaign=json.loads(path.read_text());inputs[str(path)]=dict(sha256=sha256(path),status=campaign['status']);complete &= campaign['status']=='COMPLETE'
        if campaign['status']!='COMPLETE' and not args.allow_partial:raise ValueError('Incomplete campaign: '+str(path))
        for name,entry in campaign['runs'].items():
            if name in runs:raise ValueError('Duplicate experiment key')
            spec=entry['spec'];checkpoint=Path(spec['checkpoint']);manifest_path=checkpoint.with_name('manifest.json');model=json.loads(manifest_path.read_text())
            if model['status']!='COMPLETE' or sha256(checkpoint)!=model['inference_checkpoint_sha256']:raise ValueError('Changed model')
            root=checkpoint.parent;replay=root.parent/name;evaluation=replay.with_name(name+'_evaluation')
            if sha256(replay/'manifest.json')!=entry['run_manifest_sha256'] or sha256(evaluation/'metrics.json')!=entry['evaluation_sha256']:raise ValueError('Changed replay/evaluation')
            for seq in ['BT-005','SN-013']:
                official=json.loads((evaluation/'metrics.json').read_text())
                if sha256(replay/(seq+'.txt'))!=official['artifacts'][seq]['prediction_sha256']:raise ValueError('Changed MOT output')
            runs[name]=entry
            models[root.name]={key:model[key] for key in ['guard','spec','source_commit','student_parameters','trainable_parameters','target_parameters','ema_updates','teacher_gradient_count','fit_queries','calibration_queries','epochs','diagnostics','inference_checkpoint_sha256','checkpoint_sha256','teacher_parameter_sha256','inference_teacher_removed']}
            models[root.name]['manifest_path']=str(manifest_path);models[root.name]['manifest_sha256']=sha256(manifest_path)
            key=f"r{model['config']['round']}_{model['guard']['variant']}_w{spec['weight']}"
            group=groups.setdefault(key,dict(variant=model['guard']['variant'],round=model['config']['round'],weight=spec['weight'],config_sha256=model['guard']['config_sha256'],seeds={}))
            if group['config_sha256']!=model['guard']['config_sha256'] or str(spec['seed']) in group['seeds']:raise ValueError('Recipe drift or repeatedseed')
            group['seeds'][str(spec['seed'])]=dict(run=name,metrics=entry['metrics'])
    for group in groups.values():
        count=len(group['seeds']);group['seed_count']=count;group['three_seed_complete']=set(group['seeds'])=={'0','1','2'}
        group['statistics']={seq:{metric:dict(mean=float(np.mean([v['metrics'][seq][metric] for v in group['seeds'].values()])),std_population=float(np.std([v['metrics'][seq][metric] for v in group['seeds'].values()])) if count>1 else None) for metric in METRICS} for seq in ['BT-005','SN-013','COMBINED_SEQ']}
    record=dict(status='EXPERIMENTS_COMPLETE' if complete else 'IN_PROGRESS',source_sha256=sha256(__file__),inputs=inputs,models=models,runs=runs,groups=groups,scope='All reported numbers from completed official two-video evaluations. One-seed groups diagnostic only; three seeds are not across-scene confidence intervals. Independent identity innovation still requires ordinary/noEMA/strongbaseline comparison; report status is not scientific goal status.')
    write_json(args.output,record)
    lines=['# Genuine Fish-JEPA training and controlled tracking results','',f"Status: {record['status']}. This table is measured; research validation and full online stages can still be pending.",'','Five fit train videos:21,443queries. SN015 calibration:10,955queries. Eightepochs,128batch,168steps/epoch, equal query order and separate augmentation RNG streams. Training-only clean current/future teacher outputs are stopped; actual EMA steps follow successful optimizer updates. All inference exports exclude teacher and optimizer. No backbone retraining or new featurebank.','','Frozen RGB encoders used sixtrain videos; V3 detector had development-video exposure. No whole-system blind-test claim. Historical7video tunedtest is not used for selection. Three seeds measure initialization/encoder variation on two development videos, not scene-generalization uncertainty.','','## Training-calibration diagnostics','','|Model|Student params|EMA updates|Clean rank1|Actual rank1|Actual N01−N10|','|---|---:|---:|---:|---:|---:|']
    for name,model in sorted(models.items()):
        c=model['diagnostics']['clean'];a=model['diagnostics']['actual'];lines.append(f"|{name}|{model['student_parameters']}|{model['ema_updates']}|{100*c['rank1']:.3f}|{100*a['rank1']:.3f}|{a['net_corrections']}|")
    lines+=['','Fixed-history retrospective query ranking is not actual online identity correction. Original artificially masked diagnostics used unmaskedmean as comparator; use separate equal-visible stress results for a fair recovery comparison. Allocated student parameter counts include maskedtraininghead; runtime does not execute that head. Samearchitecture noEMA controls preserve the identical architecture/activeauxiliary losses. GRU/Transformer nominalcounts differ<3%; inferenceactive counts differ about5%, so no exact activecapacity equality claim.','','## Fixed-proposal development tracking','','All methods share frozen proposals, seedmatched RGB coordinates, CFP lifecycle and actual current matchedmeasurement output. Original ordinaryGRU currentmeasurement three-seedmean HOTA53.591386 is a stronger output baseline than its previous51.885 Kalmanbox output. No per-video threshold.','','|Variant/weight|Seeds|HOTA|AssA|DetA|IDF1|MOTA|IDSW|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,group in sorted(groups.items()):
        values=group['statistics']['COMBINED_SEQ'];cells=[]
        for metric in ['HOTA','AssA','DetA','IDF1','MOTA','IDSW']:
            v=values[metric];cells.append(f"{v['mean']:.4f}"+(f" ± {v['std_population']:.4f}" if v['std_population'] is not None else ''))
        lines.append('|'+name+f"|{group['seed_count']}|"+'|'.join(cells)+'|')
    lines+=['','Full metric vectors and BT005/SN013 results, individualseed outputs, model/source/input hashes are in the adjacent measured JSON. Means combine official per-seed COMBINED_SEQ results; no averaging of video scores to invent a combined metric.','','Actual own-child epoch3 interruption/resume proof is retained in local phase5/jepa_J3_actual_resume_v1.proof.json: all student/teacher/optimizer/scaler/RNG/EMA/epochloss/queryorder states exactly match continuous training, excluding elapsedseconds. The proof applies to the original audited trainer version; thirdround trainer is separately snapshotted.','','Reproduction: run the committed frozen training plans with tools/cfp_run_jepa_training.py, then corresponding replay plans with tools/cfp_run_recovery_campaign.py. Largeweights and traces remain local under /data3/liuyeqiang/IR-MFT_assets/cfp/phase5. Full onlinetiming uses tools/cfp_run_jepa_online.py and independent manifests; cached association timing is never end-to-end FPS.']
    args.report.write_text('\n'.join(lines)+'\n')
    print(record['status'],len(models),'models',len(runs),'settings',flush=True)


if __name__=='__main__':main()
