"""Artificial selection/file controls; not actual complete Dev or Test freezing."""
import copy
import json
from pathlib import Path

import pytest

import tools.task_utility_selection as selection
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_study import STUDY,REGISTRY
from tools.task_utility_study_rules import NORMAL


def curves(through=6,improving=False):
    rows=[]
    for variant,cue in NORMAL:
        for epoch in range(1,through+1):
            for intervention in (['normal','unavailable','IR_image_spatial_roll'] if cue=='ir' else ['normal']):
                rows.append(dict(id=f'fake_{variant}_{cue}_{epoch}_{intervention}',fit_id=f'TU_{variant}_{cue}',
                    variant=variant,cue=cue,epoch=epoch,seed=0,split='dev',frames=4834,sequences=['BT-005','SN-013'],
                    kind='normal' if intervention=='normal' else 'input_intervention',intervention=intervention,
                    selection_metrics=dict(HOTA=40+(epoch if improving else 0)-(cue=='ir'),AssA=50.,IDF1=55.)))
    return rows


def test_exact_curves_keep_negative_IR_and_extend_all_when_RGB_improves():
    rows=curves();normal,decision,chosen=selection.validate_curves(rows,6)
    assert len(normal)==120 and not decision['extend_all']
    assert chosen['primary']['selection_metrics']['HOTA']<chosen['matched_RGB']['selection_metrics']['HOTA']
    for row in rows:
        if row['cue']=='rgb' and row['epoch']==6:row['selection_metrics']['HOTA']=40.200001
    _,decision,_=selection.validate_curves(rows,6)
    assert decision['extend_all'] and decision['through_epoch']==9
    assert all(r['cue']=='rgb' for r in decision['improved_conditions'])


@pytest.mark.parametrize('corruption',['missing_roll','duplicate','extra_epoch','gate_case','Test'])
def test_selection_rejects_incomplete_duplicated_extra_and_nonselection_observations(corruption):
    rows=curves()
    if corruption=='missing_roll':rows.pop()
    elif corruption=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif corruption=='extra_epoch':rows[0]['epoch']=7
    elif corruption=='gate_case':rows[0]['kind']='inference_gate_removal'
    else:rows[0]['split']='test'
    with pytest.raises(ValueError):selection.validate_curves(rows,6)


def test_file_digest_rejects_stale_sources_and_conflicting_prior_receipts(tmp_path):
    path=tmp_path/'original.csv';path.write_text('artificial,not_actual_metrics\n');files={}
    selection.put_hash(files,path);original=sha256(path)
    path.write_text('changed\n')
    with pytest.raises(ValueError):selection.put_hash(files,path)
    with pytest.raises(ValueError):selection.put_hash({},path,original)


def test_common_budget_cannot_extend_after_complete_Dev_stop(tmp_path,monkeypatch):
    (tmp_path/REGISTRY).write_text('{"artificial":true}');cases=[dict(id='fake',config='not_a_real_case')]
    rows=curves();monkeypatch.setattr(selection,'original_Dev_rows',lambda *args:(rows,{}))
    first=selection.complete_stage({}, {},tmp_path,cases,6)
    assert not first['budget_decision']['extend_all']
    rows=curves(9,improving=True)
    monkeypatch.setattr(selection,'load_stage',lambda *args:first)
    with pytest.raises(ValueError,match='extra Train epochs'):selection.complete_stage({}, {},tmp_path,cases,9)
    assert not selection.stage_path(tmp_path,9).exists()


def test_original_stage_receipt_rechecks_original_files_before_selection(tmp_path,monkeypatch):
    registry=tmp_path/REGISTRY;registry.write_text('{"artificial":true}')
    source=tmp_path/'fake_completed_Dev_metric.json';source.write_text('{"fake":true}')
    rows=curves();monkeypatch.setattr(selection,'original_Dev_rows',lambda *args:(rows,{str(source):sha256(source)}))
    cases=[dict(id='fake')];selection.complete_stage({}, {},tmp_path,cases,6)
    assert selection.load_stage({}, {},tmp_path,6)['through_epoch']==6
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='stage changed'):selection.load_stage({}, {},tmp_path,6)


def test_Test_freeze_cannot_skip_budget_or_the_primary_intervention_Dev(tmp_path,monkeypatch):
    normal,decision,chosen=selection.validate_curves(curves(improving=True),6)
    stage=dict(selection=chosen,budget_decision=decision)
    monkeypatch.setattr(selection,'load_stage',lambda *args:stage)
    with pytest.raises(ValueError,match='budget must finish'):selection.freeze_Test({}, {},tmp_path,6,[])
    stage['budget_decision']['extend_all']=False
    monkeypatch.setattr(selection,'frozen_test_models',lambda *args:([],{}))
    monkeypatch.setattr(selection,'additional_Dev_models',lambda *args:[{},dict(id='must_run_actual_primary_Dev')])
    with pytest.raises(ValueError,match='geometry Dev'):selection.freeze_Test({}, {},tmp_path,6,[])
    assert not (tmp_path/'Test_selection_frozen.json').exists()
