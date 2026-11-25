"""Artificial recipes/streams only; these do not qualify a GPU or a fitted model."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.observe_task_utility as command
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import (CANONICAL,FRESH,additional_Dev_models,frozen_test_models,
    register_trial,require_test_freeze,snapshot,stage_models,validate_trial)
from tools.task_utility_observation import observe_sequence
from tools.task_utility_study import STUDY,REGISTRY,arm_path,make_arm
from tools.task_utility_study_rules import NORMAL,select_models


@pytest.fixture
def recipes(tmp_path):
    study=read(STUDY);arms={}
    for fit in study['normal_fits']:
        path=arm_path(tmp_path,fit['id']);path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(make_arm(study,fit['id'],fit['variant'],fit['cue'])))
        arms[fit['id']]=sha256(path)
        directory=tmp_path/'fits'/fit['id'];directory.mkdir(parents=True)
        for epoch in range(1,13):
            (directory/f'epoch_{epoch:02d}.pt').write_bytes(f'FAKE_RECIPE_BYTES_NOT_A_TORCH_MODEL:{fit["id"]}:{epoch}'.encode())
    registration=dict(normal_arms=arms,study_sha256=sha256(STUDY),source_hashes={'artificial':'not_execution_evidence'})
    (tmp_path/REGISTRY).write_text(json.dumps(registration))
    return study,tmp_path,registration


def artificial_selection():
    rows=[]
    for variant,cue in NORMAL:
        for epoch in range(1,7):
            for intervention in (['normal','unavailable','IR_image_spatial_roll'] if cue=='ir' else ['normal']):
                # Deliberately retain conditions with worse synthetic Dev than RGB.
                rows.append(dict(fit_id=f'TU_{variant}_{cue}',variant=variant,cue=cue,epoch=epoch,seed=0,
                    split='dev',sequences=['BT-005','SN-013'],frames=4834,intervention=intervention,
                    selection_metrics=dict(HOTA=30+epoch/100-(cue=='ir'),AssA=40.,IDF1=50.)))
    return select_models(rows,6)


def test_stage_contains_every_normal_epoch_and_both_IR_interventions(recipes):
    study,base,registration=recipes
    for first,last,count in [(1,6,121),(7,9,61),(10,12,61)]:
        models=stage_models(study,base,first,last)
        assert len(models)==count and models[0]==FRESH
        for variant,cue in NORMAL:
            expected=(last-first+1)*(3 if cue=='ir' else 1)
            assert sum(m.get('fit_id')==f'TU_{variant}_{cue}' for m in models)==expected
    with pytest.raises(ValueError):stage_models(study,base,1,3)
    bad=copy.deepcopy(study);bad['normal_fits'].pop()
    with pytest.raises(ValueError):stage_models(bad,base,1,6)


def test_test_recipes_keep_all_twelve_including_negative_IR_and_separate_interventions(recipes):
    study,base,registration=recipes;selection=artificial_selection()
    models,roles=frozen_test_models(base,selection)
    assert len(models)==25 and len({m['id'] for m in models})==25
    assert sum(m.get('kind')=='normal' for m in models)==12
    assert sum(m.get('kind')=='input_intervention' for m in models)==8
    assert sum(m.get('kind')=='inference_gate_removal' for m in models)==3
    assert sum(m.get('kind')=='inference_only_fixed_geometry' for m in models)==1
    assert roles['matched_RGB']==selection['matched_RGB']['fit_id']
    assert len(additional_Dev_models(base,selection))==5
    assert selection['primary']['selection_metrics']['HOTA']<selection['matched_RGB']['selection_metrics']['HOTA']


def test_trial_binds_original_checkpoint_arm_flags_and_unique_case_identity(recipes):
    study,base,registration=recipes;fit='TU_foreground_utility_labels_ir'
    models=[dict(FRESH),snapshot(base,fit,1)]
    path,cases=register_trial(base,'artificial_dev','dev',models,True);value=read(path)
    assert len(cases)==3 and cases[-1]['id']==CANONICAL['id']
    assert validate_trial(value,base,registration)==value
    for change in [dict(seed=True),dict(cue='gray'),dict(kind='input_intervention'),
            dict(checkpoint_sha256='wrong'),dict(geometry_fixed=1),
            dict(gate_removal='utility',intervention='unavailable')]:
        bad=copy.deepcopy(value);bad['models'][1].update(change)
        with pytest.raises(ValueError):validate_trial(bad,base,registration)
    bad=copy.deepcopy(value);bad['models'].append(copy.deepcopy(bad['models'][1]))
    with pytest.raises(ValueError):validate_trial(bad,base,registration)
    Path(models[1]['checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError):validate_trial(value,base,registration)


def test_test_freeze_rejects_absence_or_changed_original_Dev_and_test_recipe(recipes):
    study,base,registration=recipes
    spec,cases=register_trial(base,'artificial_test','test',[dict(FRESH)],True)
    with pytest.raises(FileNotFoundError):require_test_freeze(base,spec)
    dev=base/'complete_Dev.json';dev.write_text('{"artificial":true}')
    original=base/'fake_Dev_metric.json';original.write_text('{"not_real_accuracy":true}')
    frozen=dict(status='FROZEN_TASK_UTILITY_ALL_COMPLETE_DEV_BEFORE_TEST',study_sha256=sha256(STUDY),
        complete_Dev_before_Test=True,Test_accuracy_used_for_selection=False,specification_sha256=sha256(spec),
        pipeline_registration_sha256=sha256(base/REGISTRY),complete_Dev_sha256=sha256(dev),
        Dev_file_hashes={str(original):sha256(original)})
    path=base/'Test_selection_frozen.json';path.write_text(json.dumps(frozen))
    assert require_test_freeze(base,spec)==frozen
    original.write_text('changed')
    with pytest.raises(ValueError):require_test_freeze(base,spec)
    original.write_text('{"not_real_accuracy":true}');bad=dict(frozen,Test_accuracy_used_for_selection=True)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):require_test_freeze(base,spec)


class Images:
    def __init__(self,split='train',frames=4,complete=243):
        self.sequence='BT-003';self.split=split;self.frames=frames;self.complete_source_frames=complete
        self.last_frame=0;self.hashed_IR_frames=0;self.decoded_IR_frames=0
        self.rows=[['BT-003',f,'fake_RGB_bytes','fake_IR_bytes'] for f in range(1,frames+1)]
    def read_frame(self,frame,needs_ir):
        assert frame==self.last_frame+1;self.last_frame=frame;self.hashed_IR_frames+=1
        self.decoded_IR_frames+=int(needs_ir)
        return np.zeros((32,32,3),np.uint8),None,self.rows[frame-1]
    def digest(self):
        return hashlib.sha256(b''.join(json.dumps(r,separators=(',',':')).encode()+b'\n' for r in self.rows)).hexdigest()


class Engine:
    def __init__(self,models):
        self.rows=models;self.sequence=None;self.needs_actual_IR_image=False;self.frame=0
    def step(self,rgb,ir,sequence,frame):
        assert frame==self.frame+1;self.frame=frame;self.sequence=sequence
        packet=np.asarray([[1,1,22,22,.95]],dtype=np.float32)
        return {m['id']:packet.copy() for m in self.rows},dict(GT_read=False,future_frame_input=False,previous_outputs_rewritten=False)


def test_train_prefix_writer_preserves_isolated_source_and_missing_MOT_and_no_overwrite(tmp_path):
    models=[dict(FRESH),dict(id='fake_missing',cue='ir',intervention='unavailable')];images=Images()
    value=observe_sequence(Engine(models),images,models,tmp_path,images.digest(),{'artificial':True})
    assert value['status']=='COMPLETE_TASK_UTILITY_TRAIN_PREFIX_ENGINEERING_OBSERVATIONS'
    assert not value['formal_Dev_Test_sequence_coverage'] and value['complete_source_sequence_frames']==243
    assert value['actual_IR_byte_frames_hashed']==4 and value['actual_IR_frames_decoded']==0
    first,second=[value['streams'][m['id']] for m in models]
    assert first['track_sha256']==second['track_sha256'] and first['detection_packet_sha256']==second['detection_packet_sha256']
    assert Path(first['track_path']).read_text().splitlines()[0].startswith('1,1,')
    with pytest.raises(ValueError):observe_sequence(Engine(models),Images(),models,tmp_path,images.digest(),{})


@pytest.mark.parametrize('failure',['byte_digest','postcondition','NN_future','frame_error'])
def test_observation_failure_keeps_emitted_tracks_without_complete_receipt(tmp_path,failure):
    models=[dict(FRESH)];images=Images();engine=Engine(models);original=engine.step
    def step(*args):
        packets,information=original(*args)
        if failure=='NN_future' and engine.frame==2:information['future_frame_input']=True
        if failure=='frame_error' and engine.frame==3:raise RuntimeError('artificial encoder failure')
        return packets,information
    engine.step=step
    with pytest.raises((ValueError,RuntimeError)):
        observe_sequence(engine,images,models,tmp_path,'wrong' if failure=='byte_digest' else images.digest(),{},
            verify_streams=lambda streams:False if failure=='postcondition' else None)
    assert not (tmp_path/'sequence_manifests/BT-003.json').exists()
    assert len(list((tmp_path/'sequence_manifests').glob('BT-003.failure_*.json')))==1
    assert (tmp_path/FRESH['id']/'track_results/BT-003.txt').read_text()


def test_truncated_formal_range_is_rejected_before_tracks(tmp_path):
    models=[dict(FRESH)]
    for split in ['dev','test']:
        images=Images(split=split,frames=4,complete=100)
        with pytest.raises(ValueError,match='truncated'):
            observe_sequence(Engine(models),images,models,tmp_path/split,images.digest(),{})
        assert not (tmp_path/split).exists()


def test_canonical_official_input_verifier_binds_all_sequences_files_and_packet_bytes(tmp_path,monkeypatch):
    base=tmp_path/'component';base.mkdir();registration=dict(source_hashes={'fake':'source'})
    (base/REGISTRY).write_text(json.dumps(registration));source=tmp_path/'original_source';(source/'detections').mkdir(parents=True)
    directory=base/'streams/art_dev';specpath=base/'spec.json';configpath=base/'case.json'
    spec=dict(split='dev',include_sources=True,stream_directory=str(directory),models=[dict(FRESH)])
    specpath.write_text(json.dumps(spec));case=dict(id=CANONICAL['id'],model=dict(CANONICAL),canonical=True,
        split='dev',study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(base/REGISTRY),
        specification=str(specpath),specification_sha256=sha256(specpath),stream_directory=str(directory))
    configpath.write_text(json.dumps(case))
    monkeypatch.setattr(command,'checked_spec',lambda *args:spec)
    for sequence,frames in LENGTHS['dev'].items():
        packetpath=source/'detections'/(sequence+'.npz');packetpath.write_bytes(b'ARTIFICIAL_NOT_ACTUAL_PACKETS')
        track=directory/CANONICAL['id']/'track_results'/(sequence+'.txt');track.parent.mkdir(parents=True,exist_ok=True);track.write_text('')
        stream=dict(sequence=sequence,frames=frames,track_path=str(track),track_sha256=sha256(track),
            detection_packet_sha256='a'*64,track_rows=0)
        value=dict(status='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT',split='dev',sequence=sequence,
            frames=frames,streams={CANONICAL['id']:stream},GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,
            specification_sha256=sha256(specpath),pipeline_registration_sha256=sha256(base/REGISTRY),source_packet=str(packetpath),
            source_packet_sha256=sha256(packetpath),source_versions=registration['source_hashes'])
        path=directory/'sequence_manifests'/(sequence+'.canonical.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    plan=dict(frozen_detection_sources=dict(rgb_dev=str(source)));protocol=dict(dev=list(LENGTHS['dev']))
    value=command.verify_observations(specpath,configpath,base,registration,plan,protocol)
    assert value['frames']==4834 and value['sequences']==['BT-005','SN-013']
    manifest=directory/'sequence_manifests/BT-005.canonical.json';original=read(manifest)
    for change in [dict(frames=1833),dict(GT_read=True),dict(status='COMPLETE_TASK_UTILITY_TRAIN_PREFIX_ENGINEERING_OBSERVATIONS')]:
        manifest.write_text(json.dumps(dict(original,**change)))
        with pytest.raises(ValueError):command.verify_observations(specpath,configpath,base,registration,plan,protocol)
    manifest.write_text(json.dumps(original));(source/'detections/BT-005.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):command.verify_observations(specpath,configpath,base,registration,plan,protocol)
