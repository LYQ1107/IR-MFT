#!/usr/bin/env python3
"""Independent current-image streams and official CPU factorial evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from tools.evaluate_native_pyramid import lengths,read_image
from tools.fusion_factorial_workflow import (
    CELLS,REGISTRY,SCIENCE,authorized,context,load_modules,read,registered,require_gpu,
)
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.neck72_runtime import require_storage
from tools.run_pseudo_ir_comparison import dump,sha256

FRESH=dict(id='FQ_source_RGB',cue='rgb',seed=0,baseline='fresh_RGB_B1')


def validate_observation_contract(specification,split,authorization,freeze=None):
    """Pure gate: frozen Dev choices must precede every new Test observation."""
    if split not in ['dev','test'] or specification['split']!=split:
        raise ValueError('Only the registered complete Dev or Test split')
    if specification['status']!='FROZEN_FACTORIAL_OBSERVATIONS':
        raise ValueError('Freeze the observation specification before producing frames')
    rows=specification['models'];identifiers=[r['id'] for r in rows]
    if len(identifiers)!=len(set(identifiers)) or [r for r in rows if r.get('baseline')]!=[FRESH]:
        raise ValueError('One unchanged fresh RGB source and unique model IDs are required')
    trained=[r for r in rows if not r.get('baseline')]
    if not trained:raise ValueError('An empty extra trial cannot supply a formal ablation')
    for row in trained:
        if (row['cue']!='ir' or row['seed']!=0 or row['cell'] not in CELLS or
                row['mode']!=authorization['mode'] or row.get('geometry_fixed',False) or
                row.get('intervention','normal')!='normal' or
                not 1<=row['epoch']<=authorization['stage']):
            raise ValueError('Only preregistered normal seed0 cells in the fixed mode/budget')
    if split=='dev':
        if any(r['cell']!='00' for r in trained) or {r['epoch'] for r in trained}!=set(range(1,authorization['stage']+1)) or len(trained)!=authorization['stage']:
            raise ValueError('The new cell requires every complete Dev snapshot; reuse the three parent curves')
    else:
        if (freeze is None or freeze['status']!='FROZEN_FOUR_FACTORIAL_CELLS_FROM_COMPLETE_DEV' or
                not freeze['complete_all_four_Dev'] or freeze['Test_accuracy_used_for_selection']):
            raise ValueError('Full four-cell Dev and an unchanged Test freeze are mandatory')
        choices={r['id']:r for r in freeze['selected_unique_models']}
        if any(choices.get(r['id'])!=r for r in trained):
            raise ValueError('A Test stream cannot introduce a new checkpoint or condition')
    return trained


def trial(path,split):
    plan,study,protocol,calibration,work,checkpoint,base=context()
    authorization,arm,registration=authorized(work)
    value=read(path)
    assert value['science_sha256']==sha256(SCIENCE)
    assert value['registry_sha256']==sha256(work/REGISTRY)
    assert value['authorization_sha256']==sha256(base/'authorization.json')
    freeze=None
    if split=='test':
        freeze=read(base/'Test_selection_frozen.json')
        assert value['Test_selection_sha256']==sha256(base/'Test_selection_frozen.json')
    validate_observation_contract(value,split,authorization,freeze)
    expected=base/'streams'/split
    assert Path(value['stream_directory'])==expected
    return plan,study,protocol,calibration,work,checkpoint,base,value


def produce(args):
    import torch
    plan,study,protocol,calibration,work,checkpoint,base,specification=trial(args.specification,args.split)
    _,_,registration,physical,uuid=require_gpu(work)
    from tools.native_pyramid_experiment import load_teacher
    from tools.native_pyramid_online_inference import OnlineNativeDetector
    from tools.native_pyramid_gate_observer import NativeGateObserver
    expected_lengths=lengths(plan,protocol,args.split);assert args.sequence in expected_lengths
    anchor=read(specification['fresh_anchor_metrics'])
    assert sha256(specification['fresh_anchor_metrics'])==specification['fresh_anchor_metrics_sha256']
    assert anchor['status']=='COMPLETE' and anchor['manifest']['frames']==sum(expected_lengths.values())
    assert anchor['manifest']['split']==args.split
    source=anchor['manifest']['sources'][args.sequence]
    assert sha256(source['manifest_path'])==source['manifest_sha256']
    original=read(source['manifest_path']);assert not original['GT_read']
    original_source=original['streams']['NP_source_fresh_RGB']
    folder=Path(specification['stream_directory']);complete=folder/'sequence_manifests'/(args.sequence+'.json')
    assert not complete.exists()
    model=load_teacher(checkpoint,training=False);observer=NativeGateObserver()
    modules,information=load_modules(specification['models'],study,'cuda',observer)
    engine=OnlineNativeDetector(model,specification['models'],modules,study['inference_input_hw'],calibration)
    assert engine.needs_actual_IR_image
    require_storage(work)
    writer=StreamTrackWriter(folder,args.sequence,modules)
    digest,prefix=hashlib.sha256(),[];started=time.monotonic()
    try:
        for frame in range(1,expected_lengths[args.sequence]+1):
            rgb_path=Path(plan['data_root'])/('train' if args.split=='dev' else 'test')/args.sequence/'visible'/f'{frame:06d}.jpg'
            ir_path=Path(plan['generated_ir_root'])/args.split/args.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rgb,rgb_sha=read_image(rgb_path);ir,ir_sha=read_image(ir_path,scalar=True)
            digest.update(json.dumps([args.sequence,frame,rgb_sha,ir_sha],separators=(',',':')).encode()+b'\n')
            if frame<=32:prefix.append([frame,rgb_sha,ir_sha])
            observations,_=engine.step(rgb,ir,args.sequence,frame);writer.append(frame,observations)
            if frame%100==0 or frame==expected_lengths[args.sequence]:
                writer.flush();require_storage(work)
                dump(folder/'sequence_manifests'/(args.sequence+'.progress.json'),dict(
                    frame=frame,frames=expected_lengths[args.sequence],physical_gpu=physical,
                    seconds=time.monotonic()-started))
        streams=writer.close(expected_lengths[args.sequence]);fresh=streams[FRESH['id']]
        assert fresh['track_sha256']==original_source['track_sha256']
        assert fresh['detection_packet_sha256']==original_source['detection_packet_sha256']
        assert digest.hexdigest()==original['original_image_bytes_delivery_sha256']
        gates={name:observer.summary(name) for name in observer.rows}
        assert all(v['frames']==expected_lengths[args.sequence] for v in gates.values())
        registered(work)
        dump(complete,dict(status='COMPLETE_FACTORIAL_CURRENT_IMAGE_STREAM_WITHOUT_GT',
            split=args.split,sequence=args.sequence,frames=expected_lengths[args.sequence],
            streams=streams,models=information,gate_statistics=gates,
            specification_sha256=sha256(args.specification),science_sha256=sha256(SCIENCE),
            original_image_bytes_delivery_sha256=digest.hexdigest(),first32_image_content=prefix,
            fresh_B1_packets_tracks_exact_parent=True,actual_RGB_IR_bytes_digest_exact_parent=True,
            parent_fresh_source_manifest=source,physical_gpu=physical,GPU_uuid=uuid,
            independent_input_batch_size=1,GT_read=False,future_frame_input=False,
            previous_outputs_rewritten=False,seconds=time.monotonic()-started,
            peak_own_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            runtime_scope='Shared stream production with read-only gate instrumentation; not individual end-to-end FPS.'))
    except Exception:
        writer.abort();raise
    finally:observer.close()


def evaluate(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Official evaluation uses CPU only'
    from tools.eval_mft26ir import run_trackeval
    case=read(args.config)
    assert case['split']==args.split and sha256(case['specification'])==case['specification_sha256']
    plan,study,protocol,calibration,work,checkpoint,base,specification=trial(case['specification'],args.split)
    assert case['model'] in specification['models'] and not case['model'].get('baseline')
    assert case['id']==case['model']['id']
    expected=lengths(plan,protocol,args.split)
    sequences=protocol['dev' if args.split=='dev' else 'historical_test'];coverage,sources={},{}
    folder=Path(specification['stream_directory'])
    for sequence in sequences:
        path=folder/'sequence_manifests'/(sequence+'.json');value=read(path)
        assert value['status']=='COMPLETE_FACTORIAL_CURRENT_IMAGE_STREAM_WITHOUT_GT'
        assert value['split']==args.split and value['sequence']==sequence and value['frames']==expected[sequence]
        assert not value['GT_read'] and value['specification_sha256']==sha256(case['specification'])
        assert value['fresh_B1_packets_tracks_exact_parent'] and value['actual_RGB_IR_bytes_digest_exact_parent']
        stream=value['streams'][case['id']]
        assert sha256(stream['track_path'])==stream['track_sha256']
        coverage[sequence]=dict(frames=value['frames'],output_rows=stream['track_rows'],
            output_sha256=stream['track_sha256'],detections=stream['detections'],
            detection_packet_sha256=stream['detection_packet_sha256'])
        sources[sequence]=dict(manifest_path=str(path),manifest_sha256=sha256(path))
    output=base/'evaluation'/args.split/case['id'];assert not (output/'metrics.json').exists()
    output.mkdir(parents=True,exist_ok=True)
    registration=registered(work)
    manifest=dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION',split=args.split,
        config=case,config_sha256=sha256(args.config),science_sha256=sha256(SCIENCE),
        sequences=sequences,frames=sum(expected.values()),coverage=coverage,sources=sources,
        source_versions=registration['source_hashes'],GT_inference_input=False,
        future_frame_input=False,previous_outputs_rewritten=False)
    dump(output/'inference_manifest.json',manifest)
    metrics=run_trackeval(str(folder/case['id']/'track_results'),plan['evaluation']['GT_root'],
        str(output),sequences,plan['evaluation']['TrackEval_root'],data_split='train' if args.split=='dev' else 'test')
    assert len(metrics)==39
    registered(work)
    dump(output/'metrics.json',dict(status='COMPLETE',trackeval=metrics,manifest=manifest))
    print(json.dumps(dict(status='COMPLETE_FACTORIAL_OFFICIAL_EVALUATION',id=case['id'],split=args.split)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['produce','evaluate'])
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--specification');parser.add_argument('--sequence');parser.add_argument('--config')
    args=parser.parse_args()
    if args.action=='produce':
        assert args.specification and args.sequence;produce(args)
    else:
        assert args.config;evaluate(args)
