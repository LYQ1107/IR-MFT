"""Registered data/worker components for protected peripheral tracking."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from tools.mft_protected_peripheral import (ProtectedStreams,PeripheralPublisher,ProtectedPeripheralTracker,policies)
from tools.mft_peripheral_frame_bank import FrameBankBuilder,SequentialFrameBank,validate as validate_frame_bank,MODES
from tools.mft_scene_coverage import KINDS,Dev_analysis_labels,uncovered_target
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.run_mft_coverage_recall import (models,locations as previous_locations,registered as previous_registered,window_csv47)
from tools.mft_proposal_quality_evaluation import ShapeTeacher
from tools.mft_birth_only import BirthOnlyQuality
from tools.diagnose_mft_birth_only import state_equal
from tools.native_pyramid_experiment import ROOT,read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered

SPEC=ROOT/'configs/perceptual_protected_peripheral_20261127.json'
SOURCES=['tools/mft_protected_peripheral.py','tools/mft_peripheral_frame_bank.py',
    'tools/mft_protected_peripheral_experiment.py','tools/run_mft_protected_peripheral.py',
    'tests/test_mft_protected_peripheral.py',str(SPEC.relative_to(ROOT))]


def locations():
    plan,protocol,work,previous=previous_locations()
    return plan,protocol,work,work/'protected_peripheral_20261127',previous


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    old=previous_registered();plan,protocol,work,base,previous=locations();spec=read(SPEC)
    assert spec['normal_cases']==policies()and len(policies())==18
    for name in ['run_r1','audit_r1']:
        e=read(previous/(name+'.execution.json'))
        assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert read(previous/'complete_Dev.json')['selected_case']is None
    assert not(previous/'Test_selection_frozen.json').exists()
    proof=read(base/'unit_regressions.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==6
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    source={**old['source_hashes'],**{n:sha256(ROOT/n)for n in SOURCES}}
    paths=[SPEC,base/'unit_regressions.json',previous/'registered.json',previous/'complete_forward_Train_validation.json',
        previous/'complete_Dev.json',previous/'independent_results_audit.json',previous/'validation_GT_view_audit.json',
        previous/'failure_analysis_Train_false_need.json',previous/'false_need_probe_actual_tool.json']
    m=models();fit,archive=m[-2:]
    frozen_json(base/'registered.json',dict(status='REGISTERED_PROTECTED_PERIPHERAL_BEFORE_NEW_SOURCE_OR_HOTA',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in paths},specification=spec,
        qualified=old['qualified'],validation_GT_views=old['validation_GT_views'],
        quality_model_sha256=fit['model_sha256'],archive_model_sha256=archive['model_sha256'],
        full_parent_Dev47=old['full_parent_Dev47'],normal_cases=policies(),
        Test_read_for_selection=False,historical_Test_exposed=True,validation_previously_inspected_in_development=True,
        legacy_ReID_Train_identity_exposure=True,new_neural_training=False,new_GPU_work=False,seed=0,
        goal_met=False,supported_innovations=0))
    guard_storage(work);print(json.dumps(dict(status='REGISTERED_PROTECTED_PERIPHERAL',sources=len(source),normal_cases=18)),flush=True)


def registered():
    plan,protocol,work,base,previous=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    for sequence,v in r['validation_GT_views'].items():
        p=Path(v['validation_GT_path']);assert sha256(p)==v['validation_GT_sha256']
        assert sha256(p.parent.parent/'seqinfo.ini')==v['validation_seqinfo_sha256']
    return r


def qualified_input(split,sequence,r,*,all_features):
    plan,protocol,work,base,previous=locations();assert sequence in LENGTHS[split]
    if split=='train':q=r['qualified']['train'][sequence]
    elif split=='dev':q=r['qualified']['dev'][sequence]
    else:
        assert (base/'Test_selection_frozen.json').exists()
        peer=work/'tracklet_permanence_20261127_r2';audit=bank_audit(peer,'test',sequence,peer_registered())
        meta=read(peer/'cache/test'/(sequence+'.json'))
        q=dict(bank_path=meta['path'],bank_sha256=meta['sha256'],frames=LENGTHS['test'][sequence],bank_audit=audit)
    assert sha256(q['bank_path'])==q['bank_sha256']
    if all_features and split=='train':a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
    else:
        with np.load(q['bank_path'],allow_pickle=False)as z:
            keys=z.files if all_features else ['detections','offsets','frame_ids'];a={k:z[k]for k in keys}
        assert a['detections'].dtype==np.float32 and a['detections'].shape[1:]==(5,)
        if all_features:
            for k in ['rgb','ir']:assert a[k].dtype==np.float32 and a[k].shape==(len(a['detections']),512)
    np.testing.assert_array_equal(a['frame_ids'],np.arange(1,q['frames']+1))
    assert a['offsets'].shape==(q['frames']+1,)and a['offsets'][0]==0 and a['offsets'][-1]==len(a['detections'])
    return q,a


def cache_paths(split,sequence):
    _,_,_,base,_=locations();folder=base/'inputs'/split
    return folder/(sequence+'.npz'),folder/(sequence+'.json')


def cache(split,sequence):
    path,meta_path=cache_paths(split,sequence);meta=read(meta_path)
    _,_,_,base,_=locations();registration=read(base/'registered.json')
    ready=read(base/('source_Test_ready.json'if split=='test'else'source_Train_Dev_ready.json'))
    receipt=ready['rows'][split+'_'+sequence]
    assert receipt['actual_exit_code']==0 and receipt['metadata_sha256']==sha256(meta_path)and receipt['cache_sha256']==meta['sha256']
    assert meta['source_versions']==registration['source_hashes']
    assert meta['pipeline_registration_sha256']==sha256(base/'registered.json')
    assert meta['quality_model_sha256']==registration['quality_model_sha256']and meta['archive_model_sha256']==registration['archive_model_sha256']
    assert meta['status']=='COMPLETE_GT_FREE_PROTECTED_FRAME_BANK'and not meta['GT_labels_parsed']
    assert meta['frames']==LENGTHS[split][sequence]and sha256(path)==meta['sha256']and path.stat().st_size==meta['bytes']
    with np.load(path,allow_pickle=False)as z:a={k:z[k]for k in z.files}
    validate_frame_bank(a,meta['frames']);return meta,a


def controls(case):
    return [dict(case,id='PP_gate_'+mode,intervention=mode)for mode in ['roll','missing']]+[
        dict(case,id='PP_without_explanation',explain=False),dict(case,id='PP_without_handoff',handoff=False)]


def cases_for(split,stage):
    _,_,_,base,_=locations();assert stage in ['normal','intervention']
    if split=='test'or stage=='intervention':
        freeze=read(base/'Test_selection_frozen.json');assert not freeze['Test_read_for_selection']
        assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
        return freeze['interventions']if stage=='intervention'else freeze['Test_cases']
    return policies()


def _prefix(sequence,a,count,m,standalone,cases):
    source=ProtectedStreams(*m[:6],standalone=standalone);pubs={c['id']:PeripheralPublisher(c)for c in cases}
    baseline=ShapeTeacher(m[3])if standalone else BirthOnlyQuality(dict(id='BO_paired_q999_c3',kind='paired',quantile='q999',confirmation=3),m[0],m[1],m[2],m[3],m[4],m[5])
    outputs={k:[]for k in pubs};builder=FrameBankBuilder();main_digest=hashlib.sha256()
    for i in range(count):
        lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
        value=source.update(sequence,i+1,d,rgb,ir);builder.append(value)
        expected=baseline.update(sequence,i+1,d,rgb,ir);text=''.join(map(mot_line,expected))
        assert text==''.join(map(mot_line,value['main_rows']));main_digest.update(text.encode())
        for key,pub in pubs.items():outputs[key].extend(pub.update(value))
        if i in [31,count-1]:state_equal(source.main.teacher,baseline.teacher)
    for pub in pubs.values():assert pub.main_projection.hexdigest()==main_digest.hexdigest()
    reader=SequentialFrameBank(sequence,builder.arrays());restored={c['id']:PeripheralPublisher(c)for c in cases};texts={k:[]for k in restored}
    for frame in range(1,count+1):
        value=reader.next(frame)
        for key,pub in restored.items():texts[key].extend(pub.update(value))
    original={k:''.join(map(mot_line,v))for k,v in outputs.items()}
    assert original=={k:''.join(map(mot_line,v))for k,v in texts.items()}
    assert hashlib.sha256(original['PP_reference'].encode()).hexdigest()==main_digest.hexdigest()
    return original,main_digest.hexdigest()


def engineer():
    r=registered();m=models();plan,protocol,work,base,previous=locations();cases=policies()
    missing=[dict(c,id='MISSING_'+c['id'],intervention='missing')for c in cases if c['kind']=='paired']
    allcases=cases+missing;rows={}
    for sequence,q in r['qualified']['train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);summary={}
        for standalone in [True,False]:
            full,digest=_prefix(sequence,a,64,m,standalone,allcases)
            short,_=_prefix(sequence,a,32,m,standalone,allcases)
            assert all(full[k].startswith(v)for k,v in short.items())
            assert hashlib.sha256(full['PP_reference'].encode()).hexdigest()==digest
            for c in cases:
                if c['kind']=='paired':
                    rgb=next(v['id']for v in cases if v['kind']=='RGB'and v['need_mode']==c['need_mode']and v['threshold']==c['threshold'])
                    assert full['MISSING_'+c['id']]==full[rgb]
            summary['standalone'if standalone else'full_parent']=dict(frames=64,main_rows_state_exact=True,
                all22_direct_source_vs_framebank_MOT_exact=True,all22_prefix32_exact=True,all4_missing_gate_RGB_exact=True)
        rows[sequence]=summary;guard_storage(work);print(json.dumps(dict(sequence=sequence,status='ENGINEERING_COMPLETE')),flush=True)
    # A longer GT-free Train prefix exercises the previously observed contained-box failure.
    sequence='PF-001';q=r['qualified']['train'][sequence];a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
    _prefix(sequence,a,1330,m,True,cases)
    registered();frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_PROTECTED_PERIPHERAL_ALL6_ENGINEERING',UTC=utc(),
        sequences=6,rows=rows,PF_Train_failure_prefix_frames=1330,PF_selection_for_engineering_not_inference_rule=True,
        GT_labels_read=False,new_accuracy=False))


def produce(split,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=registered();plan,protocol,work,base,previous=locations();started=time.monotonic();m=models()
    q,a=qualified_input(split,sequence,r,all_features=True);n=q['frames'];source=ProtectedStreams(*m[:6],standalone=split=='train')
    builder=FrameBankBuilder();original=hashlib.sha256();window=hashlib.sha256();packet=hashlib.sha256();cut=int(.75*n)if split=='train'else 0
    for i in range(n):
        lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi];before=d.copy()
        value=source.update(sequence,i+1,d,a['rgb'][lo:hi],a['ir'][lo:hi]);builder.append(value)
        text=''.join(map(mot_line,value['main_rows']));original.update(text.encode())
        if i+1>cut:window.update(''.join(mot_line((f-cut,box,identity,s))for f,box,identity,s in value['main_rows']).encode())
        packet.update(np.asarray([i+1,len(d)],dtype='<i8').tobytes()+np.asarray(d,dtype='<f4').tobytes())
        np.testing.assert_array_equal(d,before)
        if (i+1)%1000==0 or i+1==n:
            guard_storage(work);atomic_json(base/'progress'/'source'/split/(sequence+'.json'),dict(sequence=sequence,frame=i+1,frames=n,seconds=time.monotonic()-started))
    arrays=builder.arrays();reader=SequentialFrameBank(sequence,arrays);restored=hashlib.sha256()
    for frame in range(1,n+1):restored.update(''.join(map(mot_line,reader.next(frame)['main_rows'])).encode())
    assert restored.hexdigest()==original.hexdigest()
    label='trainval'if split=='train'else split
    if split=='test':prior=read(work/'birth_only_evaluation_20261127/streams/test/sequence_manifests'/(sequence+'.json'));key='BO_paired_q999_c3'
    else:prior=read(previous/'streams'/label/'sequence_manifests'/(sequence+'.json'));key='CR_reference'
    assert window.hexdigest()==prior['streams'][key]['track_sha256']
    path,meta_path=cache_paths(split,sequence);path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()and not meta_path.exists()
    np.savez_compressed(path,**arrays)
    assert sum(p.stat().st_size for p in (base/'inputs').rglob('*.npz'))<=r['specification']['private_source_cache_total_cap_bytes']
    assert sha256(m[-2]['model_file'])==r['quality_model_sha256']and sha256(m[-1]['model_file'])==r['archive_model_sha256']
    registered();guard_storage(work)
    frozen_json(meta_path,dict(status='COMPLETE_GT_FREE_PROTECTED_FRAME_BANK',UTC=utc(),split=split,sequence=sequence,frames=n,
        path=str(path),sha256=sha256(path),bytes=path.stat().st_size,payload_bytes=sum(v.nbytes for v in arrays.values()),
        main_rows=len(arrays['main']),shadow_rows=len(arrays['shadow']),source_NN_bank_path=q['bank_path'],source_NN_bank_sha256=q['bank_sha256'],
        original_detector_packet_sha256=packet.hexdigest(),original_main_MOT_sha256=original.hexdigest(),scored_reference_MOT_sha256=window.hexdigest(),
        whole_main_roundtrip_MOT_exact=True,prior_main_reference_exact=True,GT_labels_parsed=False,images_descriptors_weights_saved=False,
        future_frame_input=False,past_outputs_rewritten=False,source_versions=r['source_hashes'],producer_pid=os.getpid(),
        pipeline_registration_sha256=sha256(base/'registered.json'),
        quality_model_sha256=r['quality_model_sha256'],archive_model_sha256=r['archive_model_sha256'],
        source_processing_seconds=time.monotonic()-started,neural_inference_image_IO_IR_generation_excluded=True))
    print(json.dumps(dict(status='SOURCE_COMPLETE',split=split,sequence=sequence,frames=n,bytes=path.stat().st_size)),flush=True)


def diagnostic():
    r=registered();plan,protocol,work,base,previous=locations();old=read(work/'scene_coverage_diagnostic_20261127/complete_Train_Dev.json')
    private=old['private_scalar_table'];assert sha256(private['path'])==private['sha256']
    with np.load(private['path'],allow_pickle=False)as z:old_features={k:z[k]for k in z.files if k.endswith('_features')}
    rows={};total_frames=0
    for split in ['train','dev']:
        rows[split]={}
        for sequence,q in r['qualified'][split].items():
            meta,arrays=cache(split,sequence);old_x=old_features[split+'_'+sequence+'_features']
            np.testing.assert_array_equal(arrays['needs'][:,0,:],old_x)
            _,a=qualified_input(split,sequence,r,all_features=False)
            # All8 source producers have completed GT-free snapshots before this analysis worker starts.
            gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);reader=SequentialFrameBank(sequence,arrays)
            counts={phase:{mode:dict(opportunities=0,positive=0,background=0,unknown=0,uncovered_positive=0)for mode in ['broad','spatially_unexplained']}for phase in ['early75','last25']}
            for i in range(q['frames']):
                value=reader.next(i+1);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi];current=gt[left:right]
                if split=='train':labels=current_detection_labels(d,current,frame=i+1,split='train')
                else:labels=Dev_analysis_labels(d,current,frame=i+1,split='dev',purpose='read_only_coverage_diagnostic')
                candidates=np.asarray([v['proposal_index']for v in value['shadow_records']if v['streak']>=3 and .3<v['score']<.7 and v['proposal_index']not in value['main_assigned']],np.int64)
                selected=candidates[value['explained'][candidates]<.5]
                out=np.asarray([np.r_[v[1][:2],v[1][:2]+v[1][2:]]for v in value['main_rows']],np.float64).reshape(-1,4)
                phase='early75'if i<int(.75*q['frames'])else'last25'
                for name,indices in [('broad',candidates),('spatially_unexplained',selected)]:
                    target=uncovered_target(d,labels,current,out,indices);c=counts[phase][name]
                    for key,source in [('opportunities','confirmed_opportunities'),('positive','known_positive'),('background','known_background'),('unknown','unknown'),('uncovered_positive','uncovered_known_positive')]:c[key]+=target[source]
            gates={mode:{kind:{str(t):int(np.count_nonzero(arrays['needs'][:,j,h]>=t))for t in [.1,.25]}for h,kind in enumerate(KINDS)}for j,mode in enumerate(MODES)}
            rows[split][sequence]=dict(frames=q['frames'],source_sha256=meta['sha256'],GT_sha256=q['GT_sha256'],
                original_need_frame_values_exact_prior=True,all3_missing_RGB_need_exact=True,candidate_analysis=counts,gate_frame_counts=gates)
            total_frames+=q['frames'];print(json.dumps(dict(split=split,sequence=sequence,status='INPUT_DIAGNOSTIC_COMPLETE')),flush=True)
    assert total_frames==32085;registered()
    frozen_json(base/'complete_source_diagnostic.json',dict(status='COMPLETE_ALL_TRAIN_DEV_SOURCE_AND_SPATIAL_DIAGNOSTIC',UTC=utc(),rows=rows,
        frames=total_frames,source_GT_labels_not_parsed=True,diagnostic_GT_after_completed_all8_sources=True,
        unknown_not_background=True,new_fits=False,thresholds_unchanged=True,Test_read=False,proxy_not_HOTA=True,goal_met=False,supported_innovations=0))


def publish(split,stage,sequence):
    r=registered();plan,protocol,work,base,previous=locations();logical='train'if split=='trainval'else split
    meta,a=cache(logical,sequence);q,dets=qualified_input(logical,sequence,r,all_features=False)
    cases=cases_for(split,stage);pubs={c['id']:PeripheralPublisher(c)for c in cases};reader=SequentialFrameBank(sequence,a)
    label=split if stage=='normal'else split+'_intervention';cut=int(.75*q['frames'])if split=='trainval'else 0
    writer=FusionMemoryStreamWriter(base/'streams'/label,sequence,[c['id']for c in cases]);started=time.monotonic()
    try:
        for frame in range(1,q['frames']+1):
            value=reader.next(frame);rows={k:p.update(value)for k,p in pubs.items()}
            if frame>cut:
                lo,hi=map(int,dets['offsets'][frame-1:frame+1]);d=dets['detections'][lo:hi]
                shifted={k:[(f-cut,box,identity,score)for f,box,identity,score in v]for k,v in rows.items()}
                writer.append(frame-cut,shifted,{k:d for k in pubs})
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress'/label/(sequence+'.json'),dict(sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
        streams=writer.close(q['frames']-cut)
    finally:writer.abort()
    for p in pubs.values():assert p.main_projection.hexdigest()==meta['original_main_MOT_sha256']
    if stage=='normal':assert streams['PP_reference']['track_sha256']==meta['scored_reference_MOT_sha256']
    else:
        chosen=read(base/'Test_selection_frozen.json')['chosen_case'];rgb=next(c['id']for c in policies()if c['kind']=='RGB'and c['need_mode']==chosen['need_mode']and c['threshold']==chosen['threshold'])
        prior=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert streams['PP_gate_missing']['track_sha256']==prior['streams'][rgb]['track_sha256']
    if split!='trainval':assert all(v['detection_packet_sha256']==meta['original_detector_packet_sha256']for v in streams.values())
    registered();frozen_json(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_CAUSAL_PROTECTED_PUBLISHER_SEQUENCE',UTC=utc(),
        split=split,stage=stage,sequence=sequence,frames=q['frames']-cut,all_past_frames_consumed=q['frames'],validation_window_cut=cut,
        cases=cases,streams=streams,source_frame_bank_sha256=meta['sha256'],source_versions=r['source_hashes'],
        statistics={k:p.statistics for k,p in pubs.items()},main_aliases={k:p.main_alias for k,p in pubs.items()},
        all_trusted_main_projection_hashes_exact=True,GT_labels_read=False,future_frame_input=False,past_outputs_rewritten=False,
        cached_publisher_seconds=time.monotonic()-started,cost_excludes_source_feature_tracking_neural_and_images=True))


def evaluate(split,stage,case_id):
    r=registered();plan,protocol,work,base,previous=locations();cases=cases_for(split,stage);case=next(c for c in cases if c['id']==case_id)
    lengths=r['specification']['validation_lengths']if split=='trainval'else LENGTHS[split]
    label=split if stage=='normal'else split+'_intervention';coverage={}
    for sequence,n in lengths.items():
        v=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'));assert v['frames']==n and v['cases']==cases and v['all_trusted_main_projection_hashes_exact']
        stream=v['streams'][case_id];assert sha256(stream['track_path'])==stream['track_sha256'];validate_mot(stream['track_path'],n,stream['track_rows']);coverage[sequence]=stream
    directory=base/'streams'/label/case_id/'track_results';output=base/'evaluation'/label/case_id;assert not output.exists();output.mkdir(parents=True)
    if split=='trainval':
        from tools.eval_mft26ir import run_trackeval
        headline=run_trackeval(str(directory),str(Path(next(iter(r['validation_GT_views'].values()))['validation_GT_path']).parents[3]),str(output),list(lengths),plan['evaluation']['TrackEval_root'],data_split='train')
        schema,counts=official_schema(plan['evaluation']['TrackEval_root'])
        result=dict(metrics39=headline,official_schema=schema,official_family_field_counts=counts,
            **window_csv47(output/'trackeval_output/pedestrian_detailed.csv',schema,lengths,headline))
    else:result=run_official_tracks(directory,output,plan,split)
    registered();frozen_json(output/'metrics.json',dict(status='COMPLETE_PROTECTED_PERIPHERAL_OFFICIAL47',case=case,
        frames=sum(lengths.values()),coverage=coverage,source_versions=r['source_hashes'],**result,goal_met=False,supported_innovations=0))
    # These are newly derived byte-identical TrackEval inputs, not canonical evidence.
    copied=output/'trackeval_format';receipts=[]
    for sequence in lengths:
        path=copied/(sequence+'.txt');source=directory/(sequence+'.txt')
        assert path.is_file()and sha256(path)==sha256(source)
        receipts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),retained_canonical=str(source)))
    for v in receipts:Path(v['path']).unlink()
    if not list(copied.iterdir()):copied.rmdir()
    frozen_json(output/'derived_duplicate_cleanup.json',dict(status='REMOVED_ONLY_VERIFIED_NEW_TRACK_EVAL_DUPLICATE_INPUTS',rows=receipts,
        original_canonical_streams_retained=True,official_CSV_summary_metrics_retained=True))
