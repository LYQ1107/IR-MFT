#!/usr/bin/env python3
"""Independent geometry, trusted-row retention, official47 and selection audit."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_protected_peripheral_experiment import locations,registered,cache,qualified_input
from tools.audit_mft_coverage_recall import parse_official,audit_validation_views
from tools.mft_scene_coverage import KINDS
from tools.mft_peripheral_frame_bank import MODES
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.native_pyramid_training_loop import guard_storage
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS


def grid_union(candidate,regions):
    """Independent clipped-cell union, rather than producer's sweep/merge."""
    box=np.asarray(candidate,np.float64);r=np.asarray(regions,np.float64).reshape(-1,4)
    lo=np.maximum(r[:,:2],box[:2]);hi=np.minimum(r[:,2:],box[2:]);valid=(hi>lo).all(axis=1)
    lo,hi=lo[valid],hi[valid]
    if not len(lo):return 0.
    xs=np.unique(np.r_[lo[:,0],hi[:,0]]);ys=np.unique(np.r_[lo[:,1],hi[:,1]])
    cx=(xs[:-1]+xs[1:])/2;cy=(ys[:-1]+ys[1:])/2
    inside=((cx[:,None,None]>=lo[None,None,:,0])&(cx[:,None,None]<hi[None,None,:,0])&
        (cy[None,:,None]>=lo[None,None,:,1])&(cy[None,:,None]<hi[None,None,:,1])).any(axis=2)
    return float((np.diff(xs)[:,None]*np.diff(ys)[None,:]*inside).sum()/np.prod(box[2:]-box[:2]))


def _main_line(frame,identity,box,score):
    return f'{frame},{identity},{float(box[0]):.3f},{float(box[1]):.3f},{float(box[2]):.3f},{float(box[3]):.3f},{float(score):.6f},-1,-1,-1\n'


def audit_projection(path,arrays,cut,aliases,expected_sha):
    aliases={int(k):int(v)for k,v in aliases.items()};assert len(aliases)==len(set(aliases.values()))
    digest=hashlib.sha256();main_rows=0;total=0;previous=0;seen=set()
    with path.open()as stream:
        groups=itertools.groupby(stream,key=lambda line:int(line.split(',',1)[0]));pending=next(groups,None)
        for display in range(1,len(arrays['frame_ids'])-cut+1):
            frame=display+cut
            if pending is not None:assert pending[0]>=display
            cells=[]
            if pending is not None and pending[0]==display:
                cells=[line.rstrip('\r\n').split(',')for line in pending[1]];pending=next(groups,None)
            lo,hi=map(int,arrays['main_offsets'][frame-1:frame+1]);base=arrays['main'][lo:hi]
            assert len(cells)>=len(base);ids=set()
            for row in cells:
                assert len(row)==10 and int(row[0])==display and int(row[1])>0 and int(row[1])not in ids
                ids.add(int(row[1]));values=list(map(float,row[2:7]))
                assert all(math.isfinite(x)for x in values)and values[2]>0 and values[3]>0 and 0<=values[4]<=1
            for parent,observed in zip(base,cells):
                original=int(parent['identity']);assert int(observed[1])==aliases[original]
                expected=_main_line(display,aliases[original],parent['tlwh'],parent['score']).rstrip('\n').split(',')
                assert observed==expected
                digest.update(_main_line(display,original,parent['tlwh'],parent['score']).encode());main_rows+=1
            total+=len(cells)
        assert pending is None
    assert digest.hexdigest()==expected_sha
    return main_rows,total


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,base,previous=locations();execution=read(base/'run_r1.execution.json');state=read(base/'state.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']and identity(execution['identity']['PID'])is None
    jobs=[j for rows in state['phases'].values()for j in rows.values()]
    assert all(j['status']=='COMPLETE'and type(j['exit_code'])is int and j['exit_code']==0 and j['argv_identity_observed']and identity(j['pid'])is None for j in jobs)
    for n,h in execution['source_hashes'].items():assert sha256(ROOT/n)==h,n
    validation_views=audit_validation_views(r)
    checked_cache_frames=0;geometry=0;maximum_error=0.;source_banks={}
    logical_splits=['train','dev']+(['test']if(base/'source_Test_ready.json').exists()else[])
    for split in logical_splits:
        for sequence in LENGTHS[split]:
            meta,a=cache(split,sequence);q,raw=qualified_input(split,sequence,r,all_features=False)
            assert q['bank_sha256']==meta['source_NN_bank_sha256']
            np.testing.assert_array_equal(a['needs'][:,:,KINDS.index('RGB')],a['needs'][:,:,KINDS.index('IR_missing')])
            assert (a['needs'][:,2,:]+1e-12>=a['needs'][:,1,:]).all()
            whole_main=hashlib.sha256()
            for i in range(q['frames']):
                dl,dh=map(int,raw['offsets'][i:i+2]);d=raw['detections'][dl:dh]
                al,ah=map(int,a['assigned_offsets'][i:i+2]);assigned=a['assigned'][al:ah]
                assert not len(assigned)or assigned.max()<len(d)
                sl,sh=map(int,a['shadow_offsets'][i:i+2]);shadow=a['shadow'][sl:sh]
                for row in shadow:
                    index=int(row['proposal']);assert 0<=index<len(d)
                    np.testing.assert_array_equal(row['xyxy'],d[index,:4]);assert row['score']==float(d[index,4])
                    fraction=grid_union(row['xyxy'],d[assigned,:4]);error=abs(fraction-float(row['explained']))
                    assert error<1e-10;maximum_error=max(maximum_error,error);geometry+=1
                ml,mh=map(int,a['main_offsets'][i:i+2])
                for row in a['main'][ml:mh]:
                    assert int(row['proposal'])in assigned
                    whole_main.update(_main_line(i+1,int(row['identity']),row['tlwh'],row['score']).encode())
            assert whole_main.hexdigest()==meta['original_main_MOT_sha256']
            source_banks[(split,sequence)]=(meta,a);checked_cache_frames+=q['frames']
    files=0;projected_rows=0;all_output_rows=0;gate_checks=0
    for label_dir in sorted((base/'streams').iterdir()):
        label=label_dir.name;logical='train'if label=='trainval'else label.split('_')[0]
        for path in sorted((label_dir/'sequence_manifests').glob('*.json')):
            m=read(path);sequence=m['sequence'];meta,a=source_banks[(logical,sequence)];cut=m['validation_window_cut']
            assert not m['GT_labels_read']and not m['future_frame_input']and not m['past_outputs_rewritten']
            assert m['source_versions']==r['source_hashes']and m['source_frame_bank_sha256']==meta['sha256']
            assert m['all_past_frames_consumed']==LENGTHS[logical][sequence]and m['frames']==LENGTHS[logical][sequence]-cut
            if logical=='train':assert cut==r['validation_GT_views'][sequence]['window_cut']
            else:assert cut==0
            for c in m['cases']:
                key=c['id'];stream=m['streams'][key];file=Path(stream['track_path']);assert sha256(file)==stream['track_sha256']
                main_count,row_count=audit_projection(file,a,cut,m['main_aliases'][key],meta['scored_reference_MOT_sha256'])
                assert row_count==stream['track_rows'];projected_rows+=main_count;all_output_rows+=row_count;files+=1
                stat=m['statistics'][key]
                # Publishers store plain counters. The independently checked
                # final MOT projection above proves retained scored rows;
                # full warm-up projection is asserted by the frozen worker.
                assert m['all_trusted_main_projection_hashes_exact']and stat['main_rows']==len(a['main'])
                if c['kind']=='reference':expected=0;assert stream['track_sha256']==meta['scored_reference_MOT_sha256']
                elif c['kind']=='unconditional':expected=LENGTHS[logical][sequence]
                else:
                    mode='expanded_without_explanation'if c['need_mode']=='expanded'and not c.get('explain',True)else c['need_mode']
                    intervention=c.get('intervention');kind='IR_missing'if intervention=='missing'else'IR_roll'if intervention=='roll'else c['kind']
                    expected=int(np.count_nonzero(a['needs'][:,MODES.index(mode),KINDS.index(kind)]>=c['threshold']))
                assert stat['coverage_gated_frames']==expected;gate_checks+=1
            if label.endswith('_intervention'):
                freeze=read(base/'Test_selection_frozen.json');chosen=freeze['chosen_case'];normal=read(base/'streams'/logical/'sequence_manifests'/(sequence+'.json'))
                rgb=next(v['id']for v in r['normal_cases']if v['kind']=='RGB'and v['need_mode']==chosen['need_mode']and v['threshold']==chosen['threshold'])
                assert m['streams']['PP_gate_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    fields=0;tables=0;cleaned_bytes=0
    for label_dir in sorted((base/'evaluation').iterdir()):
        label=label_dir.name;logical='trainval'if label=='trainval'else label.split('_')[0]
        lengths=r['specification']['validation_lengths']if logical=='trainval'else LENGTHS[logical]
        for path in sorted(label_dir.glob('*/metrics.json')):
            row=read(path);csv=path.parent/'trackeval_output/pedestrian_detailed.csv';values=parse_official(csv,row['official_schema'],lengths)
            assert sha256(csv)==row['official_CSV_sha256']
            assert {k:values[-1][k]for k in row['combined47']}==row['combined47']
            for exported,original in zip(row['per_sequence47'],values):assert all(exported[k]==v for k,v in original.items());fields+=47
            summary=(path.parent/'trackeval_output/pedestrian_summary.txt').read_text().splitlines();headers=summary[0].split();rounded=list(map(float,summary[1].split()))
            assert len(headers)==len(rounded)==39 and all(abs(row['combined47'][k]-v)<=.000501 for k,v in zip(headers,rounded))
            cleanup=read(path.parent/'derived_duplicate_cleanup.json')
            for v in cleanup['rows']:
                assert not Path(v['path']).exists()and sha256(v['retained_canonical'])==v['sha256'];cleaned_bytes+=v['bytes']
            tables+=1
    val=read(base/'complete_forward_Train_validation.json');dev=read(base/'complete_Dev.json');spec=r['specification']
    reference=next(v for v in val['rows']if v['case']['kind']=='reference')
    gains={v['case']['id']:v['combined47']['HOTA']-reference['combined47']['HOTA']for v in val['rows']}
    assert gains==val['HOTA_gains_pp']==dev['validation_HOTA_gains_pp']
    eligible=[v for v in dev['rows']if v['case']['kind']in spec['primary_selection_kinds']and gains[v['case']['id']]>=spec['minimum_validation_HOTA_gain_pp']and v['combined47']['HOTA']>=spec['full_Dev_noninferiority_HOTA']-spec['full_Dev_tolerance_pp']]
    assert [v['case']['id']for v in eligible]==dev['eligible_ids']
    chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
    assert dev['selected_case']==(None if chosen is None else chosen['case'])
    assert next(v for v in dev['rows']if v['case']['kind']=='reference')['combined47']==r['full_parent_Dev47']
    test=None
    if chosen is None:
        assert not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
        assert state['status']=='COMPLETE_PROTECTED_PERIPHERAL_NO_ELIGIBLE_TRAIN_DEV_CASE_NO_TEST'
    else:
        freeze=read(base/'Test_selection_frozen.json');test=read(base/'full_Test_results.json')
        assert freeze['chosen_case']==chosen['case']==test['selected_case']and not test['Test_used_for_selection']
        assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')and freeze['validation_sha256']==sha256(base/'complete_forward_Train_validation.json')
        selected=next(v for v in test['rows']if v['case']==chosen['case'])
        assert test['original_RGB_baseline']['official_CSV_sha256']=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
        assert test['HOTA_gain_over_original_RGB_pp']==selected['combined47']['HOTA']-test['original_RGB_baseline']['combined47']['HOTA']
        assert test['target_plus5_met']==(test['HOTA_gain_over_original_RGB_pp']>=5)
    result=dict(status='COMPLETE_INDEPENDENT_PROTECTED_PERIPHERAL_AUDIT',UTC=utc(),controller_actual_exit=0,
        controller_execution_sha256=sha256(base/'run_r1.execution.json'),related_jobs_actual_exit0=len(jobs),source_files=len(r['source_hashes']),
        source_cache_frames_checked=checked_cache_frames,independent_union_geometry_rows=geometry,maximum_union_fraction_error=maximum_error,
        original_main_MOT_projections_checked=files,original_main_rows_preserved=projected_rows,all_output_rows_checked=all_output_rows,
        gate_activation_counts_rebuilt=gate_checks,official_tables_reparsed=tables,official47_fields_checked=fields,
        validation_GT_views_reconstructed=validation_views,new_duplicate_MOT_copy_cleanup_verified_bytes=cleaned_bytes,
        prospective_selection_rebuilt=True,selected_case=None if chosen is None else chosen['case'],
        Test_gain_pp=None if test is None else test['HOTA_gain_over_original_RGB_pp'],
        own_source_sha256=sha256(Path(__file__)),helper_parser_sha256=sha256(ROOT/'tools/audit_mft_coverage_recall.py'),
        scope='Independent clipped-grid spatial union, source coordinates, all trusted-row geometry/score retention and immutable injective aliases from final MOT, original references, gate counts, official47/39, cleanup and selection. Not a second pixel/encoder producer or complete independent Kalman/LAP replay.',
        budget=guard_storage(work),goal_met=False,supported_innovations=0)
    frozen_json(base/'independent_results_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
