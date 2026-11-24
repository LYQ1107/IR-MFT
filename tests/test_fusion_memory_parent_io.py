"""Temporary synthetic filesystem contracts, never experimental accuracy.

Checkpoint files below are inert text. They test byte provenance only; strict
tensor loading, real data coverage and GPU qualification are not established.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.fusion_memory_parent_io import DevParentFiles, LENGTHS, ParentEvidencePending


ROOT = Path(__file__).resolve().parents[1]
TRAIN = {'BT-001': 3000, 'BT-003': 243, 'MSK-002': 1754,
         'PF-001': 15000, 'SN-001': 684, 'SN-015': 6570}
COLUMNS = ['seq', 'HOTA___AUC', 'AssA___AUC', 'IDF1', 'CLR_Frames', 'GT_Dets', 'GT_IDs']
COLUMNS += ['synthetic_column_'+str(i) for i in range(276-len(COLUMNS))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')
    return path


def table(path, value=.2, *, frames=4834, gt=10, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for seq, count in [*LENGTHS.items(), ('COMBINED', frames)]:
            row = {key: 0 for key in columns}
            row.update(seq=seq, HOTA___AUC=value, AssA___AUC=value, IDF1=value,
                       CLR_Frames=count, GT_Dets=gt, GT_IDs=2)
            writer.writerow(row)
    return path


@pytest.fixture(scope='module')
def schedules():
    # Use the real deterministic address schedule; no image/GT/checkpoint reads.
    from tools.native_pyramid_train_data import schedule_proof
    return [schedule_proof(TRAIN, epoch) for epoch in range(1, 7)]


def fixture_tree(root, schedules, *, corrupt=None, removals=False):
    """Independent producer-shaped native/spatial documents for six epochs."""
    work, configs = root/'SYNTHETIC_NOT_RESEARCH', root/'configs'
    root.mkdir(parents=True, exist_ok=True)
    source = write(root/'source.json', {'synthetic': True})
    teacher = write(root/'INERT_RGB_CHECKPOINT.json', {'no_tensors': True})
    protocol_path = write(configs/'protocol.json', {'dev': list(LENGTHS),
        'derived_gt_sha256': {'train/'+seq: 'a'*64 for seq in TRAIN}})
    canonical = table(work/'canonical/trackeval_output/pedestrian_detailed.csv')
    plan_path = write(configs/'perceptual_mft_20261122.json', dict(work_root=str(work),
        protocol='configs/protocol.json', frozen_detection_sources={'rgb_dev': str(canonical.parents[1])}))
    rules = root/'tools/fusion_memory_design.py'
    rules.parent.mkdir(parents=True)
    rules.write_bytes((ROOT/'tools/fusion_memory_design.py').read_bytes())
    spec = dict(main_plan_sha256=digest(plan_path), F4_design_rules_sha256=digest(rules), parents={})
    design = dict(main_plan_sha256=digest(plan_path), protocol_sha256=digest(protocol_path), parent_studies={})
    all_info, anchors = {}, {}
    for family, architectures in [('native', ['quality', 'temporal']),
            ('spatial', ['spatial', 'quality_spatial', 'quality_spatial_redundancy'])]:
        base = work if family == 'native' else work/'native_spatial'
        arms = configs/'perceptual_native_pyramid_fit_cases_20261124' if family == 'native' else base/'arms'
        fits = base/('native_pyramid_fits' if family == 'native' else 'fits')
        specs = base/('native_pyramid_specs' if family == 'native' else 'specifications')
        phase = 'dev_normal_06'
        streams = base/('native_pyramid_streams' if family == 'native' else 'streams')/phase
        cases = configs/'perceptual_native_pyramid_dev_cases_20261124' if family == 'native' else base/'cases/dev'
        evaluation = base/('native_pyramid_evaluation/dev' if family == 'native' else 'evaluation/dev')
        prefix = 'NP' if family == 'native' else 'NS'
        normal = [dict(id=arch+'_'+cue, cue=cue, **{('mode' if family == 'native' else 'variant'): arch})
                  for arch in architectures for cue in ['rgb', 'gray', 'ir']]
        study = dict(main_plan_sha256=digest(plan_path), protocol_sha256=digest(protocol_path),
            source_RGB_sha256=digest(teacher), calibration_sha256='c'*64, fixed_seed=0,
            Train_delivery={'lengths': TRAIN}, normal_fits=normal,
            removals={'all_modes': ['no_spectral_contrast'], 'temporal_only': []},
            variants=[dict(name=a, use_spatial=True, use_quality=i > 0, use_redundancy=i > 1)
                      for i, a in enumerate(architectures)])
        study_path = write(configs/(family+'_study.json'), study)
        design['parent_studies'][family] = dict(path=str(study_path.relative_to(root)), sha256=digest(study_path))
        configs_by_id = {}
        removal_fit = dict(id='remove_no_spectral_contrast', cue='ir', mode='quality', removal='no_spectral_contrast')
        fit_records = normal + ([removal_fit] if removals and family == 'native' else [])
        for arm in fit_records:
            mode = arm.get('mode', 'quality')
            value = dict(arm, fit_id=arm['id'], mode=mode, removal=arm.get('removal'),
                geometry_fixed=False, module_kwargs={'SYNTHETIC': True}, Train_delivery={'lengths': TRAIN})
            configs_by_id[arm['id']] = write(arms/(arm['id']+'.json'), value)
        registry_path = base/('native_pyramid_pipeline_registered.json' if family == 'native' else 'pipeline_registered.json')
        registry = dict(study_sha256=digest(study_path), source_hashes={'source.json': digest(source)},
            artifact_hashes={str(teacher): digest(teacher)}, **{
                ('fit_configs' if family == 'native' else 'normal_arms'):
                    {arm['id']: digest(configs_by_id[arm['id']]) for arm in normal}})
        write(registry_path, registry)
        spec['parents'][family] = dict(registry=str(registry_path), registry_sha256=digest(registry_path), normal_fits=len(normal))
        models, metadata = [], {}
        for arm in fit_records:
            key = arm['id']
            initial = hashlib.sha256((family+arm.get('mode', arm.get('variant'))).encode()).hexdigest()
            meta = dict(fit_id=key, cue=arm['cue'], seed=0, fit_config_sha256=digest(configs_by_id[key]),
                study_sha256=digest(study_path), registry_sha256=digest(registry_path),
                source_RGB_sha256=digest(teacher), source_hashes=registry['source_hashes'],
                protocol_sha256=digest(protocol_path), calibration_sha256='c'*64,
                Train_GT_sha256={seq: 'a'*64 for seq in TRAIN}, initial_native_sha256=initial,
                physical_gpu='4', Dev_fit_rows=0, Test_fit_rows=0, **{
                    ('original_RGB_encoder_and_head_frozen' if family == 'native' else 'original_RGB_frozen'): True})
            metadata[key] = meta
            history = [dict(epoch=i, delivery=dict(s, actual_address_sha256=s['address_sha256'],
                delivered_clips=6814, complete_all_Train_frames=True), frozen_RGB_exact_unchanged=True)
                       for i, s in enumerate(schedules, 1)]
            if corrupt == 'Train_address' and family == 'native' and key == normal[0]['id']:
                history[0]['delivery']['actual_address_sha256'] = 'f'*64
            for epoch in range(1, 7):
                checkpoint = write(fits/key/f'epoch_{epoch:02d}.pt', {'SYNTHETIC_NO_TENSORS': epoch})
                model = dict(id=f'{prefix}_{key}_e{epoch:02d}', fit_id=key, epoch=epoch, mode=arm.get('mode', 'quality'),
                    cue=arm['cue'], removal=arm.get('removal'), seed=0, intervention='normal', geometry_fixed=False,
                    fit_config_sha256=digest(configs_by_id[key]), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
                if family == 'spatial':
                    model['variant'] = arm['variant']
                models.append(model)
                if arm['cue'] == 'ir' and not arm.get('removal'):
                    for kind, suffix in [('unavailable', '_off'), ('IR_image_spatial_roll', '_roll')]:
                        models.append(dict(model, id=model['id']+suffix, intervention=kind))
            write(fits/key/'progress.json', dict(status='COMPLETE_REGISTERED_TRAIN_STAGE', completed_full_Train_epochs=6,
                registered_through_epoch=6, delivered_clips_total=6*6814, actual_optimizer_updates=6*6814,
                history=history, metadata=meta, frozen_RGB_exact_unchanged=True, checkpoint_sha256=digest(checkpoint)))
        fresh = dict(id=prefix+'_source_fresh_RGB', cue='rgb', seed=0, baseline='fresh_RGB_B1')
        canonical_model = dict(id=prefix+'_source_canonical_RGB', cue='rgb', seed=0,
                               baseline='historical_canonical_RGB_B4', **({'canonical': True} if family == 'native' else {}))
        models.insert(0, fresh)
        recipe = write(specs/(phase+'.json'), dict(study_sha256=digest(study_path), split='dev', models=models, stream_directory=str(streams)))
        all_models = models+[canonical_model]
        producers = {}
        for seq, frames in LENGTHS.items():
            for is_canonical in [False, True]:
                chosen_models = [canonical_model] if is_canonical else models
                output = {}
                for model in chosen_models:
                    path = streams/model['id']/'track_results'/(seq+'.txt')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('1,1,1,1,12,12,0.9,-1,-1,-1\n')
                    output[model['id']] = dict(sequence=seq, frames=frames, track_path=str(path),
                        track_sha256=digest(path), detection_packet_sha256='b'*64, detections=1, track_rows=1)
                status = ('COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT' if is_canonical else
                    'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT') if family == 'native' else (
                    'COMPLETE_SPATIAL_CANONICAL_REPLAY_WITHOUT_GT' if is_canonical else
                    'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT')
                producer = dict(status=status, sequence=seq, frames=frames, split='dev', streams=output,
                    study_sha256=digest(study_path), specification_sha256=digest(recipe), GT_read=False)
                if not is_canonical:
                    producer.update(physical_gpu='4', future_frame_input=False, previous_outputs_rewritten=False,
                        original_image_bytes_delivery_sha256=hashlib.sha256(seq.encode()).hexdigest(), models={}, gate_statistics={})
                    producer['independent_input_batch_size' if family == 'native' else 'independent_input_batch'] = 1
                    if family == 'native':
                        producer['source_RGB_sha256'] = digest(teacher)
                    else:
                        producer.update(parent_native_source_anchor=anchors[seq], parent_fresh_packets_and_tracks_exact=True)
                    for model in models:
                        producer['models'][model['id']] = dict(model)
                        if not model.get('baseline'):
                            producer['models'][model['id']]['fit_metadata'] = metadata[model['fit_id']]
                            missing = model['intervention'] == 'unavailable'
                            producer['gate_statistics'][model['id']] = dict(frames=frames,
                                available_frames=0 if missing else frames, missing_frames=frames if missing else 0,
                                levels=[{}, {}, {}])
                path = write(streams/'sequence_manifests'/(seq+('.canonical' if is_canonical else '')+'.json'), producer)
                producers[seq, is_canonical] = path, producer
                if family == 'native' and not is_canonical:
                    anchors[seq] = dict(path=str(path), sha256=digest(path))
        rows = []
        for model in all_models:
            is_canonical = model == canonical_model
            cfg = dict(model, split='dev', study_sha256=digest(study_path), specification=str(recipe),
                       specification_sha256=digest(recipe), stream_directory=str(streams)) if family == 'native' else dict(
                       id=model['id'], model=model, split='dev', study_sha256=digest(study_path), canonical=is_canonical,
                       specification=str(recipe), specification_sha256=digest(recipe), stream_directory=str(streams))
            config = write(cases/(model['id']+'.json'), cfg)
            metric = evaluation/model['id']/'metrics.json'
            # Identical artificial values deliberately produce no eligible parent.
            csv_path = table(metric.parent/'trackeval_output/pedestrian_detailed.csv')
            summary = {key: 20. for key in ['HOTA', 'AssA', 'IDF1', *['fixture_'+str(i) for i in range(36)]]}
            coverage, sources = {}, {}
            for seq in LENGTHS:
                producer_path, producer = producers[seq, is_canonical]
                data = producer['streams'][model['id']]
                coverage[seq] = dict(data) if family == 'spatial' else dict(frames=data['frames'],
                    output_rows=data['track_rows'], output_sha256=data['track_sha256'], detections=data['detections'],
                    detection_packet_sha256=data['detection_packet_sha256'])
                sources[seq] = dict(manifest_path=str(producer_path), manifest_sha256=digest(producer_path))
            result = dict(status='COMPLETE', trackeval=summary, manifest=dict(config=cfg, config_sha256=digest(config),
                split='dev', study_sha256=digest(study_path), frames=4834, sequences=list(LENGTHS), coverage=coverage,
                sources=sources, source_versions=registry['source_hashes'], GT_inference_input=False,
                future_frame_input=False, previous_outputs_rewritten=False))
            if family == 'spatial':
                result.update(selection_metrics={k: 20. for k in ['HOTA', 'AssA', 'IDF1']}, official_CSV_sha256=digest(csv_path))
            write(metric, result)
            row = dict(cfg if family == 'native' else model, source_metrics=str(metric), source_metrics_sha256=digest(metric))
            row['metrics' if family == 'native' else 'metrics39'] = summary
            if family == 'spatial':
                row.update(selection_metrics=result['selection_metrics'], official_CSV_sha256=digest(csv_path))
            rows.append(row)
        own = [next(r for r in rows if r.get('fit_id') == arm['id'] and r.get('epoch') == 1 and r['intervention'] == 'normal') for arm in normal]
        primary = next(r for r in own if r['cue'] == 'ir')
        selection = dict(primary=primary, **{('all_six_own_Dev_best' if family == 'native' else 'all_nine_own_Dev_best'): own,
            ('fit_registered_removals' if family == 'native' else 'fit_enabled_removals'): removals and family == 'native'})
        if corrupt == 'chosen_metadata' and family == 'native':
            selection = copy.deepcopy(selection)
            selection['primary']['checkpoint_sha256'] = 'f'*64
        if corrupt == 'missing_curve' and family == 'native':
            rows = [r for r in rows if not (r.get('fit_id') == normal[0]['id'] and r.get('epoch') == 6)]
        normal_rows = [r for r in rows if not r.get('removal')]
        stage_path = write(base/('native_pyramid_Dev_stage_06.json' if family == 'native' else 'complete_Dev_stage_06.json'),
            dict(study_sha256=digest(study_path), rows=normal_rows, **{
                ('Test_read_for_selection' if family == 'native' else 'Test_accuracy_used'): False,
                ('all_required_off_full_metrics_CSV_tracks_exact' if family == 'native' else 'complete_all9_and_interventions'): True}))
        choice_path = write(base/('native_pyramid_normal_selection_frozen.json' if family == 'native' else 'normal_selection_frozen.json'),
            dict(study_sha256=digest(study_path), selection=selection, **{
                ('complete_epochs_each' if family == 'native' else 'stage'): 6,
                ('complete_Dev_stage_sha256' if family == 'native' else 'complete_stage_sha256'): digest(stage_path),
                ('Test_read_for_selection' if family == 'native' else 'Test_accuracy_used'): False}))
        choices = {}
        if family == 'native' and removals:
            write(base/'native_pyramid_removals_registered.json', dict(study_sha256=digest(study_path),
                selection_sha256=digest(choice_path), same_complete_Train_epochs=6, Test_read_for_selection=False,
                fits=[removal_fit], fit_configs={removal_fit['id']: digest(configs_by_id[removal_fit['id']])}))
            values = [r for r in rows if r.get('fit_id') == removal_fit['id']]
            choices['no_spectral_contrast'] = dict(Dev_best=values[0], paired_primary_epoch=values[0])
            if corrupt == 'missing_removal':
                rows = [r for r in rows if not (r.get('fit_id') == removal_fit['id'] and r.get('epoch') == 6)]
        removal_plan = dict(new_fits=[], conditions={}, Test_accuracy_used=False)
        if family == 'spatial':
            write(base/'removals_registered.json', dict(study_sha256=digest(study_path),
                normal_selection_sha256=digest(choice_path), new_arms={}, plan=removal_plan))
        complete_path = write(base/('native_pyramid_complete_Dev.json' if family == 'native' else 'complete_Dev.json'),
            dict(study_sha256=digest(study_path), rows=rows, selection=selection, removals=choices, **{
                ('executed_complete_epochs_each' if family == 'native' else 'complete_epochs_each'): 6,
                ('complete_normal_and_required_removal_Dev' if family == 'native' else 'complete_all9_normal_and_required_removal_Dev'): True,
                ('Test_read_for_selection' if family == 'native' else 'Test_accuracy_used'): False},
                **({'removal_provenance': removal_plan} if family == 'spatial' else {})))
        roles = dict(primary_IR=primary['id'], primary_IR_unavailable=primary['id']+'_off',
            primary_IR_image_roll=primary['id']+'_roll', source_RGB_canonical=canonical_model['id'], source_RGB_fresh=fresh['id'])
        test_spec = write(specs/'test_frozen.json', dict(split='test', models=models))
        if corrupt == 'frozen_model' and family == 'native':
            models = copy.deepcopy(models)
            next(r for r in models if r['id'] == primary['id'])['checkpoint_sha256'] = 'f'*64
            write(test_spec, dict(split='test', models=models))
        frozen = write(base/('native_pyramid_Test_selection_frozen.json' if family == 'native' else 'Test_selection_frozen.json'),
            dict(status='FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST' if family == 'native' else 'FROZEN_SPATIAL_ALL_COMPLETE_DEV_BEFORE_TEST',
                study_sha256=digest(study_path), complete_Dev_before_Test=True, complete_Dev_sha256=digest(complete_path),
                specification_sha256=digest(test_spec), models=models, roles=roles, selection=selection, removals=choices,
                **{('Test_read_for_selection' if family == 'native' else 'Test_accuracy_used_for_selection'): False}))
        all_info[family] = dict(complete=complete_path, frozen=frozen, registry=registry_path, study=study_path)
    design_path = write(configs/'perceptual_fusion_memory_study_20261125.json', design)
    spec['F4_scientific_design_sha256'] = digest(design_path)
    write(configs/'perceptual_f4_parent_io_20261125.json', spec)
    return all_info


def test_complete_two_family_files_are_not_formal_launch_permission(tmp_path, schedules):
    fixture_tree(tmp_path, schedules)
    value = DevParentFiles(tmp_path).load()
    assert value['status'] == 'COMPLETE_ORIGINAL_PARENT_DEV_FILE_EVIDENCE'
    assert value['selection']['chosen_family'] is None
    assert not value['formal_launch_ready'] and not value['checkpoint_tensor_load_performed']
    assert not value['goal_met'] and value['supported_innovations'] == 0
    assert len(value['records'][0]['normal_best']) == 6
    assert len(value['records'][1]['normal_best']) == 9


@pytest.mark.parametrize('corrupt,match', [('missing_curve', 'Incomplete full normal'),
    ('chosen_metadata', 'identical original complete Dev row'), ('Train_address', 'Actual Train addresses'),
    ('missing_removal', 'Incomplete required removal'), ('frozen_model', 'substituted a Dev-selected model')])
def test_incomplete_claims_cannot_qualify(tmp_path, schedules, corrupt, match):
    fixture_tree(tmp_path, schedules, corrupt=corrupt, removals=corrupt == 'missing_removal')
    with pytest.raises(ValueError, match=match):
        DevParentFiles(tmp_path).load()


def test_missing_future_markers_return_pending_before_accuracy_or_checkpoint_reads(tmp_path, schedules):
    fixture_tree(tmp_path, schedules)
    reader = DevParentFiles(tmp_path)
    for family in ['native', 'spatial']:
        for key in ['complete', 'choice', 'freeze']:
            reader.layout(family)[key].unlink()
    with pytest.raises(ParentEvidencePending) as error:
        reader.load()
    assert len(error.value.paths) == 6
    assert not any(p.suffix == '.pt' or p.name == 'metrics.json' or p.suffix == '.csv' for p in reader.digests)


def test_modified_registered_source_is_rejected(tmp_path, schedules):
    fixture_tree(tmp_path, schedules)
    (tmp_path/'source.json').write_text('CHANGED')
    with pytest.raises(ValueError, match='Frozen parent source changed'):
        DevParentFiles(tmp_path).load()


def test_verified_file_mutation_is_rejected(tmp_path):
    reader = DevParentFiles.__new__(DevParentFiles)
    reader.digests = {}
    path = write(tmp_path/'evidence.json', {'original': True})
    reader.hash(path)
    path.write_text('CHANGED')
    with pytest.raises(ValueError, match='changed during the read'):
        reader.verify_unchanged()


@pytest.mark.parametrize('path', ['evaluation/test/a/metrics.json', 'native_pyramid_evaluation/test/x/metrics.json',
                                'gt.txt', 'test/track_results/x.txt', 'evaluation/Test_frozen/x/metrics.json'])
def test_test_accuracy_and_gt_are_rejected_before_open(tmp_path, path):
    reader = DevParentFiles.__new__(DevParentFiles)
    reader.digests = {}
    with pytest.raises(ValueError, match='Test accuracy cannot enter'):
        reader.hash(tmp_path/path)


@pytest.mark.parametrize('kind', ['nonfinite', 'GT_count', 'frames', 'schema'])
def test_official_csv_contract_rejects_incompatible_evaluation(tmp_path, kind):
    reader = DevParentFiles.__new__(DevParentFiles)
    reader.digests = {}
    reader.csv(table(tmp_path/'reference.csv'), reference=True)
    path = table(tmp_path/'candidate.csv', float('nan') if kind == 'nonfinite' else .2,
        frames=4833 if kind == 'frames' else 4834, gt=11 if kind == 'GT_count' else 10,
        columns=COLUMNS[:-1]+['unexpected'] if kind == 'schema' else COLUMNS)
    with pytest.raises(ValueError):
        reader.csv(path)


@pytest.mark.parametrize('line', ['1,1,0,0,-1,2,0.9,-1,-1,-1\n', '1835,1,0,0,2,2,0.9,-1,-1,-1\n',
    '1,1,0,0,2,2,nan,-1,-1,-1\n', '1,1,0,0,2,2,0.9,-1,-1,-1\n1,1,0,0,2,2,0.9,-1,-1,-1\n'])
def test_invalid_mot_is_not_accepted_as_delivery(tmp_path, line):
    reader = DevParentFiles.__new__(DevParentFiles)
    reader.digests = {}
    path = tmp_path/'tracks.txt'
    path.write_text(line)
    with pytest.raises(ValueError):
        reader.track_file(path, 1834, len(line.splitlines()))
