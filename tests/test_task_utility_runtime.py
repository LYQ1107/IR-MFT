"""Artificial clocks/models/files; no actual GPU eligibility, method FPS or accuracy."""
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import CANONICAL,FRESH
from tools.task_utility_runtime import (CanonicalRGBBatch,RuntimeImages,aggregate_runtime,
    measure_prefix,packet_digest)
import tools.task_utility_runtime_sources as sources


def observation(frame):return np.asarray([[10+frame,20,40+frame,60,.95]],dtype=np.float32)


def reference(case,frames=8):
    packets=hashlib.sha256();tracks=hashlib.sha256();tracker=IndependentByteTrack()
    for frame in range(1,frames+1):
        values=observation(frame);packet_digest(packets,frame,values)
        tracks.update(''.join(mot_line(r) for r in tracker.update(values,frame)).encode())
    return dict(id=case['id'],sequence='BT-003',prefix_frames=frames,
        source_rows=[['BT-003',f,'a'*64,'b'*64] for f in range(1,frames+1)],
        packet_sha256=packets.hexdigest(),literal_MOT_sha256=tracks.hexdigest())


class Images:
    def __init__(self,needs_ir=False):
        self.sequence='BT-003';self.frames=8;self.last_frame=0;self.hashed_RGB_frames=0
        self.hashed_IR_frames=self.decoded_IR_frames=0
    def read_frame(self,frame,needs_ir):
        assert frame==self.last_frame+1;self.last_frame=frame;self.hashed_RGB_frames+=1
        self.hashed_IR_frames+=int(needs_ir);self.decoded_IR_frames+=int(needs_ir)
        rgb=np.full((32,32,3),frame,np.uint8);ir=np.full((32,32),frame,np.uint8) if needs_ir else None
        return rgb,ir,[self.sequence,frame,'a'*64,'b'*64 if needs_ir else None]


class Engine:
    def __init__(self,case,batch=1,needs_ir=False):
        self.rows=[case];self.batch_size=batch;self.needs_actual_IR_image=needs_ir
        self.sequence=None;self.last_frame=0;self.calls=[];self.closed=False
    def step(self,rgb,ir,sequence,frame):
        assert frame==self.last_frame+1 and int(rgb[0,0,0])==frame
        self.sequence=sequence;self.last_frame=frame;self.calls.append([frame])
        return {self.rows[0]['id']:observation(frame)},dict(GT_read=False,previous_outputs_rewritten=False,future_frame_input=False)
    def step_batch(self,images,sequence,first):
        assert len(images)==4 and [int(rgb[0,0,0]) for rgb in images]==list(range(first,first+4))
        self.sequence=sequence;self.last_frame=first+3;self.calls.append(list(range(first,first+4)))
        return [{self.rows[0]['id']:observation(f)} for f in range(first,first+4)],dict(GT_read=False,previous_outputs_rewritten=False)
    def close(self):self.closed=True


@pytest.mark.parametrize('batch,needs_ir',[(1,False),(1,True),(4,False)])
def test_independent_passes_preserve_warmup_state_real_batch_counts_and_component_accounting(batch,needs_ir):
    case=dict(CANONICAL if batch==4 else FRESH);made=[];ticks=[]
    def factory():
        engine=Engine(case,batch,needs_ir);made.append(engine);return engine
    def clock():
        ticks.append(1);return len(ticks)*.01
    value=measure_prefix(factory,Images,case,'BT-003',reference(case),frames=8,warmup=4,clock=clock)
    assert len(made)==2 and all(e.closed for e in made)
    assert all(sum(map(len,e.calls))==8 for e in made)
    assert value['verification']['measured_frames']==0 and value['timing']['measured_frames']==4
    assert value['timing']['warmup_to_timing_tracker_and_detector_state_preserved']
    assert value['timing']['IR_frames_decoded']==(8 if needs_ir else 0)
    assert len(ticks)==4*(4//batch)
    costs=aggregate_runtime([value]);assert costs['measured_batches']==4//batch
    assert costs['seconds']==pytest.approx(.03*(4//batch))
    assert sum(costs['component_seconds'].values())==pytest.approx(costs['seconds'])
    assert ('mean_frame_latency_ms' in costs)==(batch==1)
    assert ('mean_batch_latency_ms' in costs)==(batch==4)
    if batch==4:assert costs['mean_batch_latency_ms']==pytest.approx(30) and costs['amortized_wall_ms_per_frame']==pytest.approx(7.5)


@pytest.mark.parametrize('field',['packet_sha256','literal_MOT_sha256'])
def test_wrong_original_output_stops_before_any_timed_pass(field):
    case=dict(FRESH);ref=reference(case);ref[field]='0'*64;made=[]
    def factory():
        e=Engine(case);made.append(e);return e
    with pytest.raises(ValueError,match='prefix differs'):
        measure_prefix(factory,Images,case,'BT-003',ref,frames=8,warmup=4,
            clock=lambda:pytest.fail('A failed source-equivalence pass cannot start timing'))
    assert len(made)==1 and made[0].closed


def test_bundled_method_is_rejected_before_image_read_and_engine_is_closed():
    case=dict(FRESH);engine=Engine(case);engine.rows.append(dict(id='other'))
    with pytest.raises(ValueError,match='Exactly one'):
        measure_prefix(lambda:engine,lambda:pytest.fail('No image read for a bundled method'),case,'BT-003',reference(case),frames=8,warmup=4)
    assert engine.closed


def test_changed_required_source_in_second_pass_fails_and_closes_the_route():
    case=dict(FRESH);engines=[]
    def factory():
        e=Engine(case);engines.append(e);return e
    def images():
        values=Images()
        if len(engines)==2:
            original=values.read_frame
            def changed(*args):
                rgb,ir,row=original(*args);row[2]='c'*64;return rgb,ir,row
            values.read_frame=changed
        return values
    with pytest.raises(ValueError,match='bytes changed'):
        measure_prefix(factory,images,case,'BT-003',reference(case),frames=8,warmup=4,
            clock=lambda:pytest.fail('Changed first warmup image must stop before timing'))
    assert len(engines)==2 and all(e.closed for e in engines)


def test_invalid_clock_and_incomplete_B4_warmup_are_rejected():
    case=dict(FRESH)
    with pytest.raises(ValueError,match='wall times'):
        measure_prefix(lambda:Engine(case),Images,case,'BT-003',reference(case),frames=8,warmup=4,clock=lambda:1.)
    case=dict(CANONICAL)
    with pytest.raises(ValueError,match='complete real batches'):
        measure_prefix(lambda:Engine(case,4),Images,case,'BT-003',reference(case),frames=8,warmup=2)


@pytest.mark.parametrize('field',['GT_read','previous_outputs_rewritten','future_frame_input'])
def test_B1_route_cannot_read_GT_future_or_rewrite_past_outputs(field):
    case=dict(FRESH);engine=Engine(case);original=engine.step
    def changed(*args):
        packets,info=original(*args);info[field]=True;return packets,info
    engine.step=changed
    with pytest.raises(ValueError):
        measure_prefix(lambda:engine,Images,case,'BT-003',reference(case),frames=8,warmup=4,
            clock=lambda:pytest.fail('An invalid observation cannot start a timed pass'))
    assert engine.closed


def test_runtime_RGB_does_not_read_unused_IR_and_real_IR_requires_original_aligned_bytes(tmp_path):
    rgb=tmp_path/'rgb/train/BT-003/visible/000001.jpg';rgb.parent.mkdir(parents=True)
    assert cv2.imwrite(str(rgb),np.zeros((24,32,3),np.uint8))
    plan=dict(data_root=str(tmp_path/'rgb'),generated_ir_root=str(tmp_path/'missing_IR'))
    reader=RuntimeImages(plan,'train','BT-003',frames=1);_,ir,row=reader.read_frame(1,False)
    assert ir is None and row==['BT-003',1,sha256(rgb),None] and reader.hashed_IR_frames==0
    with pytest.raises(FileNotFoundError):RuntimeImages(plan,'train','BT-003',1).read_frame(1,True)
    with pytest.raises(ValueError):reader.read_frame(1,False)
    with pytest.raises(ValueError):RuntimeImages(plan,'test','BT-003')


class FakeYOLOX(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.batches=[]
    def forward(self,images):
        self.batches.append(images.detach().clone())
        # The original NMS expects multiple decoded anchors (real YOLOX:23625).
        result=torch.zeros((len(images),2,6),dtype=images.dtype,device=images.device)
        result[:,:,0]=16+images[:,0,0,0,None];result[:,:,1:4]=16;result[:,0,4:]=.95
        return result


def test_canonical_route_forwards_four_distinct_RGB_samples_through_actual_NN_API():
    model=FakeYOLOX().eval();case=dict(CANONICAL);engine=CanonicalRGBBatch(model,case,(32,32))
    images=[np.full((32,32,3),v,np.uint8) for v in [20,40,60,80]]
    packets,info=engine.step_batch(images,'BT-003',1)
    assert len(model.batches)==1 and model.batches[0].shape==(4,3,32,32)
    assert all(not torch.equal(model.batches[0][i],model.batches[0][i+1]) for i in range(3))
    assert len({float(p[case['id']][0,0]) for p in packets})==4 and not info['online_B1']
    assert engine.last_frame==4 and info['independent_NN_sample_batch']==4
    with pytest.raises(ValueError):engine.step_batch(images,'BT-003',6)
    with pytest.raises(ValueError):engine.step_batch(images[:3],'BT-003',5)
    with pytest.raises(ValueError):CanonicalRGBBatch(model,dict(FRESH),(32,32))
    with pytest.raises(ValueError):CanonicalRGBBatch(model.train(),case,(32,32))


def test_aggregate_cannot_mix_batch_policies_or_duplicate_videos():
    case=dict(FRESH);a=measure_prefix(lambda:Engine(case),Images,case,'BT-003',reference(case),frames=8,warmup=4)
    with pytest.raises(ValueError,match='Unique'):aggregate_runtime([a,a])
    b=copy.deepcopy(a);b['timing']['sequence']='other';b['timing']['NN_batch_size']=4
    with pytest.raises(ValueError,match='batch policy'):aggregate_runtime([a,b])


def test_original_all26_case_bindings_keep_canonical_B4_separate_and_reject_case_tampering(tmp_path,monkeypatch):
    spec=tmp_path/'spec.json';spec.write_text('{"artificial":true}');cases=[]
    for i in range(26):
        model=dict(CANONICAL) if i==0 else dict(FRESH) if i==1 else dict(id='fake_'+str(i),cue='ir')
        path=tmp_path/(model['id']+'.json');path.write_text(json.dumps(dict(id=model['id'],model=model,
            canonical=i==0,split='test',specification_sha256=sha256(spec))))
        cases.append(dict(id=model['id'],config=str(path),config_sha256=sha256(path)))
    monkeypatch.setattr(sources,'require_test_freeze',lambda *args:dict(cases=cases))
    rows=sources.runtime_routes(tmp_path,spec)
    assert len(rows)==26 and rows[0]['NN_batch_size']==4 and all(r['NN_batch_size']==1 for r in rows[1:])
    assert sources.runtime_policy()['accuracy_or_training_budget_changed'] is False
    path=Path(cases[0]['config']);value=json.loads(path.read_text());value['canonical']=False;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='file changed'):sources.runtime_routes(tmp_path,spec)
    cases[0]['config_sha256']=sha256(path)
    with pytest.raises(ValueError,match='route must be explicit'):sources.runtime_routes(tmp_path,spec)


@pytest.mark.parametrize('canonical',[False,True])
def test_original_seven_video_references_bind_actual_MOT_packet_files_and_detect_mutation(tmp_path,monkeypatch,canonical):
    from tools.fusion_memory_observation import LENGTHS
    case=dict(CANONICAL if canonical else FRESH);base=tmp_path/'component';directory=base/'streams/test_frozen'
    base.mkdir();spec=base/'spec.json';spec.write_text('{"artificial":true}')
    (base/'pipeline_registered.json').write_text('{"artificial":true}')
    (base/'Test_selection_frozen.json').write_text('{"artificial":true}')
    config=base/'case.json';config.write_text(json.dumps(dict(id=case['id'],model=case,split='test',
        stream_directory=str(directory),canonical=canonical)))
    manifest=dict(sources={});fresh_manifest=dict(sources={},coverage={});references={}
    for sequence,frames in LENGTHS['test'].items():
        track=directory/case['id']/'track_results'/(sequence+'.txt');track.parent.mkdir(parents=True,exist_ok=True);track.write_text('')
        stream=dict(track_path=str(track),track_sha256=sha256(track))
        digest=hashlib.sha256()
        for frame in range(1,33):packet_digest(digest,frame,np.empty((0,5),np.float32))
        current=dict(status='COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT',frames=frames,prefix_frames=32,
            GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,
            provenance=dict(pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),specification_sha256=sha256(spec)),
            prefix32_source_rows=[[sequence,f,'a'*64,'b'*64] for f in range(1,33)],
            streams={case['id']:stream},prefix32_packet_sha256={case['id']:digest.hexdigest()},
            prefix32_literal_MOT_sha256={case['id']:sha256(track)})
        original=directory/'sequence_manifests'/(sequence+'.json');original.parent.mkdir(parents=True,exist_ok=True);original.write_text(json.dumps(current))
        fresh_manifest['sources'][sequence]=dict(path=str(original),sha256=sha256(original))
        fresh_manifest['coverage'][sequence]=stream
        if canonical:
            packets=base/'old_packets'/(sequence+'.npz');packets.parent.mkdir(exist_ok=True)
            np.savez(packets,frame_ids=np.arange(1,frames+1),offsets=np.zeros(frames+1,np.int64),boxes=np.empty((0,4),np.float32),scores=np.empty(0,np.float32))
            path=directory/'sequence_manifests'/(sequence+'.canonical.json')
            path.write_text(json.dumps(dict(source_packet=str(packets),source_packet_sha256=sha256(packets),streams={case['id']:stream})))
        else:path=original
        manifest['sources'][sequence]=dict(path=str(path),sha256=sha256(path));references[sequence]=original
    routes=[dict(id=case['id'],config=str(config),config_sha256=sha256(config))]
    fresh_config=base/'fresh.json'
    if canonical:
        fresh_config.write_text(json.dumps(dict(model=dict(FRESH))))
        routes.append(dict(id=FRESH['id'],config=str(fresh_config),config_sha256=sha256(fresh_config)))
    monkeypatch.setattr(sources,'verify_observations',lambda spec,path,*args:fresh_manifest if Path(path)==fresh_config else manifest)
    monkeypatch.setattr(sources,'runtime_routes',lambda *args:routes)
    result=sources.runtime_references(spec,config,base,{}, {},{})
    assert len(result['references'])==7 and all(r['prefix_frames']==32 for r in result['references'].values())
    sources.unchanged_runtime_sources(result)
    changed=next(iter(result['file_hashes']));assert changed==str(sources.RUNTIME)
    first=next(iter(LENGTHS['test']));track=directory/case['id']/'track_results'/(first+'.txt');track.write_text('changed')
    with pytest.raises(ValueError,match='references changed'):sources.unchanged_runtime_sources(result)
    track.write_text('');original=references[first];data=json.loads(original.read_text());data['frames']-=1;original.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='complete current-image'):
        sources.runtime_references(spec,config,base,{}, {},{})
