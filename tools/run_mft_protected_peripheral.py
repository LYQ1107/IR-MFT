#!/usr/bin/env python3
"""Bounded causal-source, forward-Train, full-Dev and frozen-Test study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mft_protected_peripheral_experiment import (locations,register,registered,engineer,produce,diagnostic,publish,evaluate,
    policies,cases_for,cache_paths,controls)
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47
from tools.fit_mft_proposal_quality import utc

PREFIX='PROTECTED_PERIPHERAL'


def source_phase(queue,splits):
    plan,protocol,work,base,previous=locations();test=splits==['test'];label='TEST'if test else'TRAIN_DEV'
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    tasks=[dict(id=split+'_'+s,split=split,sequence=s,command=cmd+['produce','--split',split,'--sequence',s])for split in splits for s in LENGTHS[split]]
    def check(job):
        path,meta_path=cache_paths(job['split'],job['sequence']);m=read(meta_path)
        assert m['frames']==LENGTHS[job['split']][job['sequence']]and not m['GT_labels_parsed']
        assert m['whole_main_roundtrip_MOT_exact']and m['prior_main_reference_exact']
        assert sha256(path)==m['sha256'];job.update(metadata_sha256=sha256(meta_path),cache_sha256=m['sha256'])
        return True
    phase=PREFIX+'_SOURCE_'+label;queue.jobs(phase,tasks,False,check)
    rows={name:dict(actual_exit_code=j['exit_code'],identity={k:j[k]for k in ['pid','start_ticks','command_sha256']},
        metadata_sha256=j['metadata_sha256'],cache_sha256=j['cache_sha256'])for name,j in queue.state['phases'][phase].items()}
    frozen_json(base/('source_Test_ready.json'if test else'source_Train_Dev_ready.json'),dict(status='ALL_RELATED_SOURCE_PRODUCERS_ACTUAL_EXIT_ZERO',UTC=utc(),rows=rows,
        no_GT_labels_in_producer=True,images_descriptors_weights_saved=False))


def phase(queue,split,stage='normal'):
    plan,protocol,work,base,previous=locations();r=registered();cases=cases_for(split,stage)
    lengths=r['specification']['validation_lengths']if split=='trainval'else LENGTHS[split]
    label=split if stage=='normal'else split+'_intervention';cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    args=['--split',split,'--stage',stage]
    queue.jobs(PREFIX+'_'+label.upper()+'_PUBLISH',[dict(id=s,command=cmd+['publish',*args,'--sequence',s])for s in lengths],False,
        lambda j:read(base/'streams'/label/'sequence_manifests'/(j['id']+'.json'))['all_trusted_main_projection_hashes_exact'])
    queue.jobs(PREFIX+'_'+label.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate',*args,'--case',c['id']])for c in cases],False,
        lambda j:read(base/'evaluation'/label/j['id']/'metrics.json')['frames']==sum(lengths.values()))
    return [read(base/'evaluation'/label/c['id']/'metrics.json')for c in cases]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,base,previous=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_PROTECTED_PERIPHERAL_STUDY',controller_pid=os.getpid(),phases={},seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            source_phase(queue,['train','dev'])
            queue.jobs(PREFIX+'_SOURCE_DIAGNOSTIC',[dict(id='all8',command=cmd+['diagnostic'])],False,
                lambda j:read(base/'complete_source_diagnostic.json')['frames']==32085)
            val=phase(queue,'trainval');reference=next(v for v in val if v['case']['kind']=='reference')
            gains={v['case']['id']:v['combined47']['HOTA']-reference['combined47']['HOTA']for v in val}
            old=read(previous/'complete_forward_Train_validation.json');oldref=next(v for v in old['rows']if v['case']['kind']=='reference')
            assert reference['combined47']==oldref['combined47']and reference['per_sequence47']==oldref['per_sequence47']
            frozen_json(base/'complete_forward_Train_validation.json',dict(status='COMPLETE_ALL18_PROTECTED_FORWARD_TRAIN_VALIDATION',UTC=utc(),
                rows=val,HOTA_gains_pp=gains,reference_case=reference['case'],scored_frames=6814,all_past_frames=27251,
                trusted_reference_all47_exact=True,late_calibrated_parent_thresholds_not_used=True,
                validation_previously_inspected_in_development=True,legacy_encoder_Train_identity_exposure=True,
                Test_read_for_selection=False,goal_met=False,supported_innovations=0))
            dev=phase(queue,'dev');ref=next(v for v in dev if v['case']['kind']=='reference')
            assert ref['combined47']==r['full_parent_Dev47']
            spec=r['specification'];eligible=[v for v in dev if v['case']['kind']in spec['primary_selection_kinds']and
                gains[v['case']['id']]>=spec['minimum_validation_HOTA_gain_pp']and
                v['combined47']['HOTA']>=spec['full_Dev_noninferiority_HOTA']-spec['full_Dev_tolerance_pp']]
            chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL18_PROTECTED_FULL_DEV',UTC=utc(),rows=dev,
                selected_case=None if chosen is None else chosen['case'],eligible_ids=[v['case']['id']for v in eligible],
                validation_HOTA_gains_pp=gains,selection_rule=spec['selection_rule'],Test_read_for_selection=False,
                original_trusted_reference_Dev47_exact=True,goal_met=False,supported_innovations=0))
            if chosen is None:
                state.update(status='COMPLETE_PROTECTED_PERIPHERAL_NO_ELIGIBLE_TRAIN_DEV_CASE_NO_TEST');queue.save();return
            case=chosen['case'];test=[c for c in policies()if c['kind']in ['reference','unconditional']or
                c['need_mode']==case['need_mode']and c['threshold']==case['threshold']]
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_PROTECTED_PERIPHERAL_BEFORE_TEST',UTC=utc(),chosen_case=case,
                Test_cases=test,interventions=controls(case),complete_Dev_sha256=sha256(base/'complete_Dev.json'),
                validation_sha256=sha256(base/'complete_forward_Train_validation.json'),Test_read_for_selection=False,
                controls_not_promoted_from_Test=True,historical_Test_exposed=True))
            intervention=phase(queue,'dev','intervention')
            frozen_json(base/'complete_Dev_interventions.json',dict(status='COMPLETE_PROTECTED_GATE_AND_PUBLISHER_DEV_REMOVALS',UTC=utc(),rows=intervention,
                scope='New gate IR and publisher components only; frozen IR detector and old quality/archive retained',no_reselection=True))
            source_phase(queue,['test']);rows=phase(queue,'test');selected=next(v for v in rows if v['case']==case)
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root']);baseline=official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            gain=selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_FROZEN_PROTECTED_PERIPHERAL_TEST7',UTC=utc(),rows=rows,selected_case=case,
                original_RGB_baseline=baseline,HOTA_gain_over_original_RGB_pp=gain,target_plus5_met=gain>=5,target_plus10_met=gain>=10,
                Test_used_for_selection=False,controls_not_promoted_from_Test=True,historical_Test_exposed=True,goal_met=False,supported_innovations=0))
            if gain>=5:
                removal=phase(queue,'test','intervention')
                frozen_json(base/'full_Test_interventions.json',dict(status='COMPLETE_FROZEN_PROTECTED_TEST7_GATE_AND_COMPONENT_REMOVALS',UTC=utc(),rows=removal,no_reselection=True))
            state.update(status='COMPLETE_PROTECTED_PERIPHERAL_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',HOTA_gain_over_original_RGB_pp=gain);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_PROTECTED_PERIPHERAL_ATTEMPT',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','run','engineer','produce','diagnostic','publish','evaluate'])
    p.add_argument('--split',choices=['train','trainval','dev','test']);p.add_argument('--stage',choices=['normal','intervention'],default='normal')
    p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode in ['register','run','engineer','diagnostic']:{'register':register,'run':run,'engineer':engineer,'diagnostic':diagnostic}[a.mode]()
    elif a.mode=='produce':produce(a.split,a.sequence)
    elif a.mode=='publish':publish(a.split,a.stage,a.sequence)
    else:evaluate(a.split,a.stage,a.case)
