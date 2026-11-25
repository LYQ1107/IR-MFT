#!/usr/bin/env python3
"""Task-utility complete B1 production, canonical replay and separate official CPU evaluation."""
import argparse
import os
from pathlib import Path
import resource
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot
from tools.fusion_memory_observation import FixedSequenceImages,LENGTHS,write_json
from tools.native_pyramid_experiment import gpu_settings,load_teacher,read
from tools.neck72_runtime import require_host_memory,require_storage
from tools.observe_spatial_native import source_anchor
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import CANONICAL,FRESH,require_test_freeze,validate_trial
from tools.task_utility_model_loading import load_auxiliary_cpu
from tools.task_utility_observation import observe_sequence,replay_canonical_sequence
from tools.task_utility_observer import TaskUtilityObserver
from tools.task_utility_study import STUDY,REGISTRY,context,predecessor_ready,registered


def load_modules(models,study,base,registration,device,observer=None):
    modules={};information={}
    for row in models:
        if row.get('baseline'):
            if row!=FRESH:raise ValueError('Only fresh current-image RGB is an online source')
            modules[row['id']]=None;information[row['id']]=dict(row);continue
        module,details=load_auxiliary_cpu(row,study,base,registration)
        module=module.to(device).float().eval().requires_grad_(False)
        modules[row['id']]=module;information[row['id']]=details
        if observer is not None:observer.attach(row['id'],module)
    if len(modules)!=len(models):raise ValueError('Repeated model identities')
    return modules,information


def checked_spec(path,split,base,registration):
    value=validate_trial(read(path),base,registration)
    if value['split']!=split:raise ValueError('Original full observation split differs')
    if split=='test':require_test_freeze(base,path)
    return value


def produce(args):
    import torch
    from tools.native_pyramid_online_inference import OnlineNativeDetector
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    if not predecessor_ready(work):raise RuntimeError('Complete prior queue exports and actual exits required before CUDA')
    path=Path(args.specification);spec=checked_spec(path,args.split,base,registration)
    if args.sequence not in LENGTHS[args.split]:raise ValueError('Original complete sequence required')
    reference,anchor=source_anchor(work,args.split,args.sequence)
    if reference['frames']!=LENGTHS[args.split][args.sequence]:raise ValueError('Incomplete original source anchor')
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work,growth_bytes=int(.5*2**30))
    physical,uuid=gpu_settings();teacher=load_teacher(checkpoint,training=False);observer=TaskUtilityObserver()
    modules,information=load_modules(spec['models'],study,base,registration,'cuda',observer)
    engine=OnlineNativeDetector(teacher,spec['models'],modules,study['inference_input_hw'],calibration)
    images=FixedSequenceImages(plan,args.split,args.sequence)
    def resources():
        require_storage(work)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own producer exceeded8GiB')
    def matched(streams):
        parent=reference['streams']['NP_source_fresh_RGB']
        for key in ['track_sha256','detection_packet_sha256']:
            if streams[FRESH['id']][key]!=parent[key]:raise ValueError('Fresh source packets/tracks differ from original B1 anchor')
            for row in spec['models']:
                if row.get('intervention')=='unavailable' and streams[row['id']][key]!=streams[FRESH['id']][key]:
                    raise ValueError('Actual missing cue must reproduce the entire fresh RGB stream')
    provenance=dict(study_sha256=sha256(STUDY),specification_sha256=sha256(path),
        pipeline_registration_sha256=sha256(base/REGISTRY),source_versions=registration['source_hashes'],
        physical_gpu=physical,GPU_uuid=uuid,model_information=information,parent_native_source_anchor=anchor,
        parent_fresh_packets_and_tracks_exact=True,actual_device='cuda')
    try:
        observe_sequence(engine,images,spec['models'],spec['stream_directory'],
            reference['original_image_bytes_delivery_sha256'],provenance,resource_check=resources,verify_streams=matched,
            diagnostics=lambda:{name:observer.summary(name) for name in observer.rows})
        registered(work)
    finally:observer.close()


def canonical(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Canonical packet replay is CPU only')
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    spec=checked_spec(args.specification,args.split,base,registration)
    if not spec['include_sources']:raise ValueError('This original trial has no canonical-source evaluation')
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    provenance=dict(specification_sha256=sha256(args.specification),pipeline_registration_sha256=sha256(base/REGISTRY),
        source_versions=registration['source_hashes'])
    for sequence in LENGTHS[args.split]:
        replay_canonical_sequence(plan,args.split,sequence,spec['stream_directory'],provenance,
            resource_check=lambda:require_storage(work))
    registered(work)


def verify_observations(specification_path,config_path,base,registration,plan,protocol):
    """Only original full files; called before GT and again after evaluation."""
    case=read(config_path);split=case['split'];spec=checked_spec(specification_path,split,base,registration)
    if (case['id']!=case['model']['id'] or case['study_sha256']!=sha256(STUDY) or
            case['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
            Path(case['specification'])!=Path(specification_path) or case['specification_sha256']!=sha256(specification_path) or
            case['stream_directory']!=spec['stream_directory']):
        raise ValueError('Original frozen case/specification binding required')
    canonical_case=case.get('canonical') is True
    if canonical_case:
        if case['model']!=CANONICAL or not spec['include_sources']:raise ValueError('Original declared canonical source required')
    elif case['model'] not in spec['models']:raise ValueError('Case is absent from original observations')
    directory=Path(spec['stream_directory']);coverage={};sources={}
    for sequence,count in LENGTHS[split].items():
        path=directory/'sequence_manifests'/(sequence+('.canonical.json' if canonical_case else '.json'))
        value=read(path)
        status='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT' if canonical_case else 'COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT'
        if (value['status']!=status or value['split']!=split or value['sequence']!=sequence or value['frames']!=count or
                value['GT_read'] or value['future_frame_input'] or value['previous_outputs_rewritten']):
            raise ValueError('Complete original causal observations required before GT')
        if canonical_case:
            source=Path(plan['frozen_detection_sources']['rgb_'+split])/'detections'/(sequence+'.npz')
            if (value['specification_sha256']!=sha256(specification_path) or
                    value['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
                    value['source_packet']!=str(source) or value['source_packet_sha256']!=sha256(source) or
                    value['source_versions']!=registration['source_hashes']):
                raise ValueError('Original canonical packet/source provenance differs')
        else:
            provenance=value['provenance'];reference,anchor=source_anchor(Path(base).parent,split,sequence)
            if (not value['formal_Dev_Test_sequence_coverage'] or value['complete_source_sequence_frames']!=count or
                    value['models']!=spec['models'] or value['independent_input_batch_size']!=1 or
                    value['original_image_bytes_delivery_sha256']!=reference['original_image_bytes_delivery_sha256'] or
                    value['actual_IR_byte_frames_hashed']!=count or provenance['parent_native_source_anchor']!=anchor or
                    provenance['study_sha256']!=sha256(STUDY) or provenance['specification_sha256']!=sha256(specification_path) or
                    provenance['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
                    provenance['source_versions']!=registration['source_hashes'] or provenance['physical_gpu'] not in ['4','5']):
                raise ValueError('Original complete current-image execution and source anchors required')
            expected_decode=count if any(m['cue']=='ir' and m.get('intervention')!='unavailable' and not m.get('baseline') for m in spec['models']) else 0
            if value['actual_IR_frames_decoded']!=expected_decode:raise ValueError('IR decode/use scope differs')
            info=provenance['model_information'][case['id']]
            if {k:info[k] for k in case['model']}!=case['model']:raise ValueError('Actual loaded model differs from frozen case')
            for field in ['track_sha256','detection_packet_sha256']:
                if value['streams'][FRESH['id']][field]!=reference['streams']['NP_source_fresh_RGB'][field]:
                    raise ValueError('Fresh complete B1 source differs')
                if case['model'].get('intervention')=='unavailable' and value['streams'][case['id']][field]!=value['streams'][FRESH['id']][field]:
                    raise ValueError('Missing input differs from complete RGB reference')
        for name in ([case['id']] if canonical_case else list(dict.fromkeys([case['id'],FRESH['id']]))):
            stream=value['streams'][name];track=directory/name/'track_results'/(sequence+'.txt')
            if (Path(stream['track_path'])!=track or sha256(track)!=stream['track_sha256'] or
                    stream['frames']!=count or stream['sequence']!=sequence or len(stream['detection_packet_sha256'])!=64):
                raise ValueError('Original complete MOT file path/bytes/coverage differ')
            validate_mot(track,count,stream['track_rows'])
        coverage[sequence]=value['streams'][case['id']];sources[sequence]=dict(path=str(path),sha256=sha256(path))
    sequences=protocol['dev' if split=='dev' else 'historical_test']
    if sequences!=list(LENGTHS[split]):raise ValueError('Original protocol sequence order differs')
    return dict(config=case,config_sha256=sha256(config_path),study_sha256=sha256(STUDY),split=split,
        sequences=sequences,frames=sum(LENGTHS[split].values()),coverage=coverage,sources=sources,
        specification_sha256=sha256(specification_path),source_versions=registration['source_hashes'],
        pipeline_registration_sha256=sha256(Path(base)/REGISTRY),GT_inference_input=False,
        future_frame_input=False,previous_outputs_rewritten=False)


def evaluate_case(specification_path,config_path,base,registration,plan,protocol):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Official metrics run separately on CPU')
    manifest=verify_observations(specification_path,config_path,base,registration,plan,protocol)
    case=manifest['config'];output=Path(base)/'evaluation'/case['split']/case['id']
    if output.exists():raise ValueError('Preserve previous complete/failed evaluation output')
    output.mkdir(parents=True);write_json(output/'inference_manifest.json',
        dict(status='TASK_UTILITY_COMPLETE_INFERENCE_BEFORE_GT_EVALUATION',**manifest),exclusive=True)
    values=run_official_tracks(Path(case['stream_directory'])/case['id']/'track_results',output,plan,case['split'])
    if verify_observations(specification_path,config_path,base,registration,plan,protocol)!=manifest:
        raise ValueError('Original inference files changed during evaluation')
    if case.get('canonical'):
        previous=Path(plan['frozen_detection_sources']['rgb_'+case['split']])
        if values['metrics39']!=read(previous/'metrics.json')['trackeval'] or values['official_CSV_sha256']!=sha256(previous/'trackeval_output/pedestrian_detailed.csv'):
            raise ValueError('Canonical source official metrics/CSV differ')
    result=dict(status='COMPLETE_TASK_UTILITY_OFFICIAL_METRICS',manifest=manifest,**values,
        formal_launch_ready=False,goal_met=False,supported_innovations=0)
    write_json(output/'metrics.json',result,exclusive=True);return result


def evaluate(args):
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    case=read(args.config)
    if case['split']!=args.split:raise ValueError('Original evaluation split required')
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    evaluate_case(case['specification'],args.config,base,registration,plan,protocol);registered(work)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['produce','canonical','evaluate'])
    parser.add_argument('--split',choices=['dev','test'],required=True);parser.add_argument('--sequence')
    parser.add_argument('--specification');parser.add_argument('--config');args=parser.parse_args()
    if args.mode=='produce':produce(args)
    elif args.mode=='canonical':canonical(args)
    else:evaluate(args)


if __name__=='__main__':main()
