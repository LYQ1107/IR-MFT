#!/usr/bin/env python3
"""Fixed selected-method Train factorial removals; never inspect Test metrics."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mft_protected_peripheral_experiment import registered,locations,cache
from tools.mft_protected_peripheral import PeripheralPublisher
from tools.mft_peripheral_frame_bank import SequentialFrameBank
from tools.audit_mft_protected_peripheral import audit_projection
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.fusion_memory_evaluation import validate_mot
from tools.run_pseudo_ir_comparison import sha256
from tools.export_perceptual_metrics47 import official_schema
from tools.run_mft_coverage_recall import window_csv47
from tools.native_pyramid_training_loop import guard_storage
from tools.fit_mft_proposal_quality import utc


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,parent,previous=locations();base=work/'protected_peripheral_Train_components_20261127'
    base.mkdir(parents=True,exist_ok=True);freeze=read(parent/'Test_selection_frozen.json');case=freeze['chosen_case']
    design=work/'protected_peripheral_Train_component_analysis_design_20261127.json';d=read(design)
    assert d['selected_case']==case and d['source_selection_frozen_sha256']==sha256(parent/'Test_selection_frozen.json')
    cases=[dict(case,id='PC_without_spatial',explain=False),dict(case,id='PC_without_handoff',handoff=False),
        dict(case,id='PC_without_both',explain=False,handoff=False)]
    paths=[parent/'registered.json',parent/'Test_selection_frozen.json',parent/'complete_forward_Train_validation.json',design]
    source=dict(r['source_hashes'])
    for n in ['tools/analyse_mft_peripheral_Train_components.py','tools/audit_mft_protected_peripheral.py','tools/audit_mft_coverage_recall.py']:source[n]=sha256(ROOT/n)
    frozen_json(base/'registered.json',dict(status='REGISTERED_SELECTED_TRAIN_FACTORIAL_BEFORE_ITS_NEW_HOTA',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in paths},selected_case=case,cases=cases,
        main_Test_results_existed_at_registration=(parent/'full_Test_results.json').exists(),Test_metrics_read=False,
        no_reselection_or_parameter_changes=True,seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0))
    started=time.monotonic();manifests={}
    for sequence,q in r['qualified']['train'].items():
        meta,a=cache('train',sequence);reader=SequentialFrameBank(sequence,a);pubs={c['id']:PeripheralPublisher(c)for c in cases}
        cut=r['validation_GT_views'][sequence]['window_cut'];files={};digests={key:hashlib.sha256()for key in pubs};counts={key:0 for key in pubs}
        for key in pubs:
            path=base/'streams'/key/'track_results'/(sequence+'.txt');path.parent.mkdir(parents=True,exist_ok=True);files[key]=(path,path.open('x'))
        try:
            for frame in range(1,q['frames']+1):
                value=reader.next(frame)
                for key,pub in pubs.items():
                    rows=pub.update(value)
                    if frame<=cut:continue
                    text=''.join(mot_line((f-cut,box,identity,score))for f,box,identity,score in rows)
                    files[key][1].write(text);digests[key].update(text.encode());counts[key]+=len(rows)
        finally:
            for path,stream in files.values():stream.close()
        manifests[sequence]={}
        for key,pub in pubs.items():
            path=files[key][0];assert sha256(path)==digests[key].hexdigest();validate_mot(path,q['frames']-cut,counts[key])
            assert pub.main_projection.hexdigest()==meta['original_main_MOT_sha256']
            main_rows,output_rows=audit_projection(path,a,cut,pub.main_alias,meta['scored_reference_MOT_sha256'])
            manifests[sequence][key]=dict(path=str(path),sha256=sha256(path),rows=counts[key],frames=q['frames']-cut,
                statistics=pub.statistics,main_aliases=pub.main_alias,independent_main_rows_checked=main_rows,independent_output_rows_checked=output_rows,
                source_bank_sha256=meta['sha256'])
        guard_storage(work);print(json.dumps(dict(sequence=sequence,status='FACTORIAL_TRAIN_STREAMS_COMPLETE')),flush=True)
    from tools.eval_mft26ir import run_trackeval
    schema,families=official_schema(plan['evaluation']['TrackEval_root']);results=[]
    gt_root=Path(next(iter(r['validation_GT_views'].values()))['validation_GT_path']).parents[3]
    for case_variant in cases:
        key=case_variant['id'];directory=base/'streams'/key/'track_results';output=base/'evaluation'/key;assert not output.exists();output.mkdir(parents=True)
        headline=run_trackeval(str(directory),str(gt_root),str(output),list(r['specification']['validation_lengths']),plan['evaluation']['TrackEval_root'],data_split='train')
        parsed=window_csv47(output/'trackeval_output/pedestrian_detailed.csv',schema,r['specification']['validation_lengths'],headline)
        result=dict(case=case_variant,frames=6814,metrics39=headline,official_schema=schema,official_family_field_counts=families,**parsed)
        frozen_json(output/'metrics.json',result);results.append(result)
        cleaned=[]
        for sequence in r['qualified']['train']:
            duplicate=output/'trackeval_format'/(sequence+'.txt');canonical=directory/(sequence+'.txt')
            assert sha256(duplicate)==sha256(canonical);cleaned.append(dict(path=str(duplicate),bytes=duplicate.stat().st_size,sha256=sha256(duplicate),retained=str(canonical)));duplicate.unlink()
        (output/'trackeval_format').rmdir();frozen_json(output/'derived_duplicate_cleanup.json',dict(rows=cleaned,canonical_retained=True))
    full=read(parent/'complete_forward_Train_validation.json');selected=next(v for v in full['rows']if v['case']==freeze['chosen_case'])
    unconditional=next(v for v in full['rows']if v['case']['kind']=='unconditional');reference=next(v for v in full['rows']if v['case']['kind']=='reference')
    values={v['case']['id']:v['combined47']['HOTA']for v in results};F=selected['combined47']['HOTA'];no_spatial=values['PC_without_spatial'];no_handoff=values['PC_without_handoff'];neither=values['PC_without_both']
    for n,h in source.items():assert sha256(ROOT/n)==h,n
    registered();frozen_json(base/'complete_Train_components.json',dict(status='COMPLETE_SELECTED_METHOD_STANDALONE_TRAIN_FACTORIAL47',UTC=utc(),
        selected_case=freeze['chosen_case'],selected_original_validation=selected,unconditional_original_validation=unconditional,reference_original_validation=reference,
        removal_rows=results,source_manifests=manifests,
        HOTA_differences_pp=dict(full_over_reference=F-reference['combined47']['HOTA'],spatial_given_handoff=F-no_spatial,
            handoff_given_spatial=F-no_handoff,spatial_without_handoff=no_handoff-neither,handoff_without_spatial=no_spatial-neither,
            interaction=F-no_handoff-no_spatial+neither,need_gate_given_both=F-unconditional['combined47']['HOTA']),
        Test_metrics_read=False,no_selection_or_threshold_change=True,source_cache_no_new_NN=True,seconds=time.monotonic()-started,
        scope='Train development-window mechanism removals only; old validation data/encoder identity exposure, no independent IR novelty or new Test result.',
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_TRAIN_COMPONENTS',HOTA_full=F,removal_HOTA=values,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
