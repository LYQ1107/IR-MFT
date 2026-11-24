"""Read original complete Dev evidence; never read Test accuracy or load tensors.

This verifies metadata, all curves, original CSVs/MOT, checkpoint bytes and Train
delivery. Strict tensor loading and real GPU engineering belong to the later
executor. A complete receipt here does not authorize CUDA or Test execution.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from tools.fusion_memory_design import ARCHITECTURES, select_visual_parent

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = {'BT-005': 1834, 'SN-013': 3000}
INTERVENTION = {'normal': 'normal', 'unavailable': 'IR_unavailable',
                'IR_image_spatial_roll': 'IR_image_spatial_roll'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ParentEvidencePending(Exception):
    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        super().__init__('Original complete parent Dev evidence is pending: '+', '.join(self.paths))


class DevParentFiles:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.digests, self.json_cache, self.fit_cache, self.schedule_cache = {}, {}, {}, {}
        self.source_image_digests = {seq: set() for seq in LENGTHS}
        self.spec = self.read(self.root/'configs/perceptual_f4_parent_io_20261125.json')
        require(self.hash(self.root/'tools/fusion_memory_design.py') == self.spec['F4_design_rules_sha256'],
                'Registered F4 rules changed')
        self.design_path = self.root/'configs/perceptual_fusion_memory_study_20261125.json'
        require(self.hash(self.design_path) == self.spec['F4_scientific_design_sha256'], 'F4 scientific design changed')
        self.design = self.read(self.design_path)
        plan_path = self.root/'configs/perceptual_mft_20261122.json'
        require(self.hash(plan_path) == self.spec['main_plan_sha256'] == self.design['main_plan_sha256'], 'Main plan changed')
        self.plan = self.read(plan_path)
        self.work = Path(self.plan['work_root']).resolve()
        self.protocol = self.read(self.root/self.plan['protocol'])
        require(self.hash(self.root/self.plan['protocol']) == self.design['protocol_sha256'], 'Protocol changed')
        require(self.protocol['dev'] == list(LENGTHS), 'Only the original complete Dev may choose an F4 parent')

    def hash(self, path):
        path = Path(path).resolve()
        require(not self.is_test_accuracy(path), 'Test accuracy cannot enter F4 parent selection')
        if path not in self.digests:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            self.digests[path] = digest.hexdigest()
        return self.digests[path]

    @staticmethod
    def is_test_accuracy(path):
        parts = tuple(p.lower() for p in path.parts)
        return (any(p in ['test', 'test_frozen'] for p in parts) and any(p in parts for p in ['evaluation', 'native_pyramid_evaluation',
                    'detections', 'track_results', 'gt'])) or path.name == 'gt.txt'

    def read(self, path):
        path = Path(path).resolve()
        self.hash(path)
        if path not in self.json_cache:
            self.json_cache[path] = json.loads(path.read_text())
        return self.json_cache[path]

    def layout(self, family):
        base = self.work if family == 'native' else self.work/'native_spatial'
        return dict(base=base,
            complete=base/('native_pyramid_complete_Dev.json' if family == 'native' else 'complete_Dev.json'),
            choice=base/('native_pyramid_normal_selection_frozen.json' if family == 'native' else 'normal_selection_frozen.json'),
            freeze=base/('native_pyramid_Test_selection_frozen.json' if family == 'native' else 'Test_selection_frozen.json'),
            fits=base/('native_pyramid_fits' if family == 'native' else 'fits'),
            specs=base/('native_pyramid_specs' if family == 'native' else 'specifications'),
            streams=base/('native_pyramid_streams' if family == 'native' else 'streams'),
            cases=self.root/'configs/perceptual_native_pyramid_dev_cases_20261124' if family == 'native' else base/'cases/dev',
            evaluation=base/'native_pyramid_evaluation/dev' if family == 'native' else base/'evaluation/dev',
            arms=self.root/'configs/perceptual_native_pyramid_fit_cases_20261124' if family == 'native' else base/'arms')

    def preflight(self):
        missing = [paths[key] for family in ARCHITECTURES for paths in [self.layout(family)]
                   for key in ['complete', 'choice', 'freeze'] if not paths[key].exists()]
        if missing:
            raise ParentEvidencePending(missing)

    def csv(self, path, reference=False):
        self.hash(path)
        with Path(path).open(newline='') as stream:
            reader = csv.DictReader(stream)
            rows, columns = list(reader), reader.fieldnames
        require(columns is not None and len(columns) == len(set(columns)) == 276, 'Original official 276-column CSV required')
        require([r['seq'] for r in rows] == [*LENGTHS, 'COMBINED'], 'Complete two-sequence official CSV required')
        for row in rows:
            require(set(row) == set(columns) and all(math.isfinite(float(row[k])) for k in columns if k != 'seq'),
                    'Official CSV has missing or nonfinite fields')
            expected = sum(LENGTHS.values()) if row['seq'] == 'COMBINED' else LENGTHS[row['seq']]
            require(float(row['CLR_Frames']) == expected, 'Official CSV frame coverage differs')
        if reference:
            self.columns, self.canonical_rows = columns, rows
        else:
            require(columns == self.columns, 'Official evaluator column schema differs from canonical Dev')
            for row, canonical in zip(rows, self.canonical_rows):
                require(all(row[k] == canonical[k] for k in ['GT_Dets', 'GT_IDs', 'CLR_Frames']), 'Different Dev GT/coverage')
        combined = rows[-1]
        return {key: 100*float(combined[key+'___AUC' if key != 'IDF1' else key]) for key in ['HOTA', 'AssA', 'IDF1']}

    def track_file(self, path, frames, expected_rows):
        self.hash(path)
        previous, identities, count = 0, set(), 0
        with Path(path).open() as stream:
            for line in stream:
                cells = line.rstrip('\n').split(',')
                require(len(cells) == 10, 'Original MOT record shape differs')
                frame, identity = int(cells[0]), int(cells[1])
                values = [float(v) for v in cells[2:7]]
                require(1 <= frame <= frames and identity > 0 and frame >= previous, 'MOT frame/identity order differs')
                require(all(math.isfinite(v) for v in values) and values[2] > 0 and values[3] > 0 and
                        0 <= values[4] <= 1, 'Nonfinite or invalid MOT geometry/score')
                if frame != previous:
                    identities.clear()
                require(identity not in identities, 'Duplicate identity in one emitted frame')
                previous = frame
                identities.add(identity)
                count += 1
        require(count == expected_rows, 'MOT row count disagrees with the producer')

    def family_context(self, family):
        paths = self.layout(family)
        study_path = self.root/self.design['parent_studies'][family]['path']
        require(self.hash(study_path) == self.design['parent_studies'][family]['sha256'], 'Original parent study changed')
        study = self.read(study_path)
        require(study['main_plan_sha256'] == self.spec['main_plan_sha256'] and
                study['protocol_sha256'] == self.design['protocol_sha256'] and
                study['fixed_seed'] == 0, 'Parent data/protocol/seed mismatch')
        registry_path = Path(self.spec['parents'][family]['registry'])
        require(self.hash(registry_path) == self.spec['parents'][family]['registry_sha256'], 'Parent registration changed')
        registry = self.read(registry_path)
        require(registry['study_sha256'] == self.hash(study_path), 'Registry binds another study')
        for name, digest in registry['source_hashes'].items():
            require(self.hash(self.root/name) == digest, 'Frozen parent source changed: '+name)
        for name, digest in registry['artifact_hashes'].items():
            require(self.hash(name) == digest, 'Frozen parent artifact changed: '+name)
        complete, choice, freeze = [self.read(paths[key]) for key in ['complete', 'choice', 'freeze']]
        require(all(r['study_sha256'] == self.hash(study_path) for r in [complete, choice, freeze]), 'Parent evidence uses another study')
        require(complete['selection'] == choice['selection'] == freeze['selection'], 'Original selection was changed after freezing')
        require(freeze['complete_Dev_before_Test'] is True and freeze['complete_Dev_sha256'] == self.hash(paths['complete']),
                'Original Test choice lacks the exact complete Dev')
        require(freeze['status'] == ('FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST' if family == 'native' else
                                    'FROZEN_SPATIAL_ALL_COMPLETE_DEV_BEFORE_TEST'), 'Unrecognized original freeze')
        require(complete['removals'] == freeze['removals'], 'Removal choices changed after freezing')
        flag = 'Test_read_for_selection' if family == 'native' else 'Test_accuracy_used'
        require(complete[flag] is False and choice[flag] is False, 'Test accuracy entered parent selection')
        require(freeze['Test_read_for_selection' if family == 'native' else 'Test_accuracy_used_for_selection'] is False,
                'Test accuracy entered the frozen choice')
        require(complete['complete_normal_and_required_removal_Dev' if family == 'native' else
                         'complete_all9_normal_and_required_removal_Dev'] is True, 'Parent Dev is incomplete')
        stage = complete['executed_complete_epochs_each' if family == 'native' else 'complete_epochs_each']
        require(stage in [6, 9, 12] and choice['complete_epochs_each' if family == 'native' else 'stage'] == stage,
                'Parent full-Train stage differs')
        stage_path = paths['base']/(f'native_pyramid_Dev_stage_{stage:02d}.json' if family == 'native' else
                                   f'complete_Dev_stage_{stage:02d}.json')
        require(choice['complete_Dev_stage_sha256' if family == 'native' else 'complete_stage_sha256'] == self.hash(stage_path),
                'Normal selection does not bind its complete stage')
        stage_record = self.read(stage_path)
        require(stage_record['study_sha256'] == self.hash(study_path) and stage_record[flag] is False and
                stage_record['all_required_off_full_metrics_CSV_tracks_exact' if family == 'native' else
                             'complete_all9_and_interventions'] is True, 'Incomplete original normal Dev stage')
        frozen_spec = paths['specs']/'test_frozen.json'
        require(self.hash(frozen_spec) == freeze['specification_sha256'], 'Original frozen Test design changed')
        frozen_metadata = self.read(frozen_spec)
        require(frozen_metadata['split'] == 'test' and frozen_metadata['models'] == freeze['models'],
                'Original Test metadata is inconsistent; no Test accuracy is read here')
        return dict(family=family, paths=paths, study=study, study_path=study_path, registry=registry,
                    registry_path=registry_path, complete=complete, freeze=freeze, stage=stage,
                    stage_record=stage_record)

    def fit(self, context, model):
        family, paths, stage = context['family'], context['paths'], context['stage']
        key = family, model['fit_id']
        arm_path = paths['arms']/(model['fit_id']+'.json')
        require(self.hash(arm_path) == model['fit_config_sha256'], 'Selected fit configuration changed')
        arm = self.read(arm_path)
        if key not in self.fit_cache:
            registry = context['registry']
            owners = dict(registry['fit_configs' if family == 'native' else 'normal_arms'])
            if arm.get('removal'):
                removal_path = paths['base']/('native_pyramid_removals_registered.json' if family == 'native' else 'removals_registered.json')
                removal = self.read(removal_path)
                require(removal['study_sha256'] == self.hash(context['study_path']), 'Removal owner differs')
                owners.update(removal['fit_configs' if family == 'native' else 'new_arms'])
            require(owners.get(model['fit_id']) == self.hash(arm_path), 'Fit has no exact original registration')
            progress_path = paths['fits']/model['fit_id']/'progress.json'
            progress = self.read(progress_path)
            require(progress['status'] == 'COMPLETE_REGISTERED_TRAIN_STAGE' and
                    progress['completed_full_Train_epochs'] == progress['registered_through_epoch'] == stage,
                    'Fit did not complete the common full-Train stage')
            meta = progress['metadata']
            expected = dict(fit_id=model['fit_id'], cue=model['cue'], seed=0,
                fit_config_sha256=self.hash(arm_path), study_sha256=self.hash(context['study_path']),
                registry_sha256=self.hash(context['registry_path']), source_RGB_sha256=context['study']['source_RGB_sha256'],
                protocol_sha256=context['study']['protocol_sha256'], calibration_sha256=context['study']['calibration_sha256'])
            require(all(meta[k] == value for k, value in expected.items()), 'Fit metadata lineage disagrees')
            require(meta['source_hashes'] == registry['source_hashes'] and meta['Dev_fit_rows'] == meta['Test_fit_rows'] == 0,
                    'Fit source or split differs')
            require(meta['physical_gpu'] in ['4', '5'] and
                    re.fullmatch(r'[0-9a-f]{64}', meta['initial_native_sha256']) is not None,
                    'Fit GPU or initialization provenance missing')
            require(meta['original_RGB_encoder_and_head_frozen' if family == 'native' else 'original_RGB_frozen'] is True,
                    'Original RGB was not frozen')
            lengths = arm['Train_delivery']['lengths']
            require(sum(lengths.values()) == 27251 and lengths == context['study']['Train_delivery']['lengths'], 'Different Train delivery')
            require(meta['Train_GT_sha256'] == {seq: self.protocol['derived_gt_sha256']['train/'+seq] for seq in lengths},
                    'Train GT provenance differs')
            history = progress['history']
            require(len(history) == stage and progress['frozen_RGB_exact_unchanged'] is True, 'Incomplete or unfrozen fit')
            from tools.native_pyramid_train_data import schedule_proof
            for epoch, row in enumerate(history, 1):
                schedule_key = tuple(sorted(lengths.items())), epoch
                if schedule_key not in self.schedule_cache:
                    self.schedule_cache[schedule_key] = schedule_proof(lengths, epoch)
                expected_delivery = self.schedule_cache[schedule_key]
                delivery = row['delivery']
                require(row['epoch'] == epoch and row['frozen_RGB_exact_unchanged'] is True and
                    delivery['complete_all_Train_frames'] is True and delivery['frames'] == 27251 and
                    delivery['clips'] == delivery['delivered_clips'] == 6814 and delivery['repeated_frames'] == 0 and
                    delivery['per_video_frames'] == lengths and
                    delivery['address_sha256'] == delivery['actual_address_sha256'] == expected_delivery['address_sha256'],
                    'Actual Train addresses, coverage or frozen state differ')
            require(progress['delivered_clips_total'] == stage*6814 and
                    0 < progress['actual_optimizer_updates'] <= stage*6814, 'Fit optimizer/delivery counts differ')
            last = paths['fits']/model['fit_id']/f'epoch_{stage:02d}.pt'
            require(self.hash(last) == progress['checkpoint_sha256'], 'Final checkpoint bytes differ')
            self.fit_cache[key] = dict(metadata=meta, arm=arm, progress_path=str(progress_path),
                                      progress_sha256=self.hash(progress_path))
        stored = self.fit_cache[key]
        expected_checkpoint = paths['fits']/model['fit_id']/f"epoch_{model['epoch']:02d}.pt"
        require(Path(model['checkpoint']).resolve() == expected_checkpoint.resolve() and
                self.hash(expected_checkpoint) == model['checkpoint_sha256'], 'Selected checkpoint bytes/path differ')
        require(1 <= model['epoch'] <= stage and model['seed'] == 0 and
                all(model[k] == stored['arm'][k] for k in ['fit_id', 'cue', 'removal', 'mode', 'geometry_fixed']) and
                (family != 'spatial' or model['variant'] == stored['arm']['variant']),
                'Selected visual architecture differs from the actual fit')
        return stored

    def case(self, context, raw):
        family, paths = context['family'], context['paths']
        identifier = raw['id']
        require(re.fullmatch(r'[A-Za-z0-9_-]+', identifier) is not None, 'Unsafe parent case identity')
        path = paths['evaluation']/identifier/'metrics.json'
        require(Path(raw['source_metrics']).resolve() == path.resolve() and self.hash(path) == raw['source_metrics_sha256'],
                'Original Dev metric reference changed')
        value = self.read(path)
        cfg_path = paths['cases']/(identifier+'.json')
        cfg = self.read(cfg_path)
        model = cfg if family == 'native' else cfg['model']
        manifest = value['manifest']
        require(value['status'] == 'COMPLETE' and len(value['trackeval']) == 39 and
                all(math.isfinite(float(v)) for v in value['trackeval'].values()), 'Incomplete official Dev metrics')
        require(manifest['config'] == cfg and manifest['config_sha256'] == self.hash(cfg_path), 'Dev case configuration differs')
        require(manifest['split'] == cfg['split'] == 'dev' and manifest['frames'] == 4834 and
                manifest['sequences'] == list(LENGTHS) and manifest['study_sha256'] == self.hash(context['study_path']),
                'Incomplete or wrong-split Dev result')
        require(manifest['source_versions'] == context['registry']['source_hashes'] and
                manifest['GT_inference_input'] is False and manifest['future_frame_input'] is False and
                manifest['previous_outputs_rewritten'] is False, 'Dev execution source or causality differs')
        require(all(raw.get(k) == v for k, v in model.items() if k not in ['study_sha256', 'split',
            'specification', 'specification_sha256', 'stream_directory']), 'Complete Dev row changed its model identity')
        spec_path = Path(cfg['specification'])
        require(spec_path.parent.resolve() == paths['specs'].resolve() and spec_path.stem.startswith('dev_') and
                self.hash(spec_path) == cfg['specification_sha256'], 'Dev observation specification differs')
        spec = self.read(spec_path)
        require(spec['split'] == 'dev' and spec['study_sha256'] == self.hash(context['study_path']), 'Non-Dev observation specification')
        directory = paths['streams']/spec_path.stem
        require(Path(spec['stream_directory']).resolve() == directory.resolve() == Path(cfg['stream_directory']).resolve(),
                'Observation directory differs')
        canonical = bool(cfg.get('canonical'))
        if not canonical:
            require(any(m['id'] == identifier and all(model.get(k) == v for k, v in m.items())
                        for m in spec['models']), 'Visual model absent from its original observation specification')
        csv_path = path.parent/'trackeval_output/pedestrian_detailed.csv'
        selected = self.csv(csv_path)
        summary = raw['metrics' if family == 'native' else 'metrics39']
        require(summary == value['trackeval'], 'Original rounded metric summary changed')
        require(all(abs(selected[k]-summary[k]) <= .000501 for k in selected), 'Official CSV and summary differ')
        if family == 'spatial':
            require(selected == raw['selection_metrics'] == value['selection_metrics'] and
                    self.hash(csv_path) == value['official_CSV_sha256'], 'Full precision spatial selection differs')
        if canonical:
            require(self.hash(csv_path) == self.canonical_csv_sha, 'Canonical Dev table did not reproduce the fixed source')
        fit = None if model.get('baseline') else self.fit(context, model)
        receipts = {}
        require(set(manifest['sources']) == set(manifest['coverage']) == set(LENGTHS), 'Missing Dev sequence evidence')
        for sequence, frames in LENGTHS.items():
            reference = manifest['sources'][sequence]
            producer_path = directory/'sequence_manifests'/(sequence+('.canonical' if canonical else '')+'.json')
            require(Path(reference['manifest_path']).resolve() == producer_path.resolve() and
                    self.hash(producer_path) == reference['manifest_sha256'], 'Observation source manifest changed')
            producer = self.read(producer_path)
            require(producer['split'] == 'dev' and producer['sequence'] == sequence and producer['frames'] == frames and
                    producer['GT_read'] is False and producer['specification_sha256'] == self.hash(spec_path), 'Incomplete observation delivery')
            expected_status = ('COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT' if canonical else
                               'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT') if family == 'native' else (
                               'COMPLETE_SPATIAL_CANONICAL_REPLAY_WITHOUT_GT' if canonical else
                               'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT')
            require(producer['status'] == expected_status, 'Unexpected parent observation status')
            if not canonical:
                require(producer['physical_gpu'] in ['4', '5'] and producer['future_frame_input'] is False and
                        producer['previous_outputs_rewritten'] is False and
                        producer['study_sha256'] == self.hash(context['study_path']) and
                        producer['independent_input_batch_size' if family == 'native' else 'independent_input_batch'] == 1,
                        'Parent observations are not the registered causal B1 path')
                if family == 'native':
                    require(producer['source_RGB_sha256'] == context['study']['source_RGB_sha256'], 'Different original RGB source')
                else:
                    anchor = producer['parent_native_source_anchor']
                    require(self.hash(anchor['path']) == anchor['sha256'], 'Native source anchor changed')
                    original = self.read(anchor['path'])
                    require(original['sequence'] == sequence and original['split'] == 'dev' and
                            original['frames'] == frames and original['GT_read'] is False and
                            original['source_RGB_sha256'] == context['study']['source_RGB_sha256'] and
                            producer['parent_fresh_packets_and_tracks_exact'] is True,
                            'Spatial observations lack their fixed native source anchor')
                image_digest = producer['original_image_bytes_delivery_sha256']
                require(re.fullmatch(r'[0-9a-f]{64}', image_digest) is not None, 'Image delivery digest missing')
                self.source_image_digests[sequence].add(image_digest)
                info = producer['models'][identifier]
                require(all(info[k] == model[k] for k in model if k not in ['study_sha256', 'split',
                    'specification', 'specification_sha256', 'stream_directory']), 'Producer loaded another model')
                if fit is not None:
                    static = ['fit_id', 'cue', 'seed', 'fit_config_sha256', 'study_sha256',
                              'registry_sha256', 'source_RGB_sha256', 'source_hashes', 'initial_native_sha256',
                              'protocol_sha256', 'calibration_sha256', 'Train_GT_sha256', 'Dev_fit_rows', 'Test_fit_rows']
                    require(all(info['fit_metadata'][k] == fit['metadata'][k] for k in static), 'Producer/checkpoint fit lineage differs')
                    gates = producer['gate_statistics'][identifier]
                    require(gates['frames'] == gates['available_frames']+gates['missing_frames'] == frames and len(gates['levels']) == 3,
                            'Incomplete visual gate observations')
                    if model.get('intervention') == 'unavailable':
                        require(gates['missing_frames'] == frames, 'Missing-IR control still received a cue')
            stream, coverage = producer['streams'][identifier], manifest['coverage'][sequence]
            track = directory/identifier/'track_results'/(sequence+'.txt')
            require(Path(stream['track_path']).resolve() == track.resolve() and self.hash(track) == stream['track_sha256'] ==
                    coverage.get('output_sha256', coverage.get('track_sha256')) and
                    stream['frames'] == coverage['frames'] == frames, 'Track path/bytes/coverage differ')
            require(stream['detection_packet_sha256'] == coverage['detection_packet_sha256'] and
                    stream['detections'] == coverage['detections'] and stream['track_rows'] ==
                    coverage.get('output_rows', coverage.get('track_rows')), 'Packet or track row evidence differs')
            self.track_file(track, frames, stream['track_rows'])
            receipts[sequence] = dict(track_sha256=stream['track_sha256'], detection_packet_sha256=stream['detection_packet_sha256'],
                source_manifest=str(producer_path), source_manifest_sha256=self.hash(producer_path))
        architecture = 'direct_rgb' if model.get('baseline') else model['mode' if family == 'native' else 'variant']
        normalized_model = dict(model) if family == 'spatial' else {k: v for k, v in model.items() if k not in
            ['study_sha256', 'split', 'specification', 'specification_sha256', 'stream_directory']}
        return dict(id=identifier, family='direct' if model.get('baseline') else family,
            architecture=architecture, cue=model['cue'], epoch=model.get('epoch', 0), baseline=bool(model.get('baseline')),
            checkpoint=model.get('checkpoint'), checkpoint_sha256=model.get('checkpoint_sha256', context['study']['source_RGB_sha256']),
            model=normalized_model, module_kwargs=None if fit is None else fit['arm']['module_kwargs'],
            split='dev', sequences=list(LENGTHS), frames=4834, seed=0,
            intervention=INTERVENTION[model.get('intervention', 'normal')], metrics=selected,
            source_metrics=str(path), source_metrics_sha256=self.hash(path),
            official_CSV=str(csv_path), official_CSV_sha256=self.hash(csv_path), source_receipts=receipts)

    def family(self, context):
        family, complete, freeze = context['family'], context['complete'], context['freeze']
        values = {}
        for raw in complete['rows']:
            require(raw['id'] not in values, 'Duplicate original complete Dev case')
            values[raw['id']] = self.case(context, raw)
        normal = context['study']['normal_fits']
        expected = {(r['id'], epoch, 'normal') for r in normal for epoch in range(1, context['stage']+1)}
        expected.update((r['id'], epoch, kind) for r in normal if r['cue'] == 'ir'
                        for epoch in range(1, context['stage']+1) for kind in ['unavailable', 'IR_image_spatial_roll'])
        observed = {(r['fit_id'], r['epoch'], r.get('intervention', 'normal')) for r in complete['rows']
                    if r.get('fit_id') in {f['id'] for f in normal}}
        require(observed == expected, 'Incomplete full normal/intervention Dev curves')
        originals = {r['id']: r for r in complete['rows']}
        stage_rows = context['stage_record']['rows']
        require(len({r['id'] for r in stage_rows}) == len(stage_rows) and
                all(originals.get(r['id']) == r for r in stage_rows) and
                {r['id'] for r in stage_rows} == {r['id'] for r in complete['rows'] if
                    r.get('baseline') or r.get('fit_id') in {f['id'] for f in normal}},
                'Complete Dev changed its originally frozen normal-stage rows')
        selection = freeze['selection']
        chosen = selection['all_six_own_Dev_best' if family == 'native' else 'all_nine_own_Dev_best']
        require(all(originals.get(r['id']) == r for r in [*chosen, selection['primary']]),
                'Frozen normal choice is not the identical original complete Dev row')
        normal_best = [values[r['id']] for r in chosen]
        require(len(normal_best) == self.spec['parents'][family]['normal_fits'], 'Missing own-Dev-selected normal arm')
        primary_id = freeze['roles']['primary_IR']
        require(primary_id == selection['primary']['id'] and primary_id in {r['id'] for r in normal_best},
                'Original primary differs from its frozen selection')
        interventions = {label: dict(values[freeze['roles'][role]], source_primary_id=primary_id)
            for label, role in [('IR_unavailable', 'primary_IR_unavailable'), ('IR_image_spatial_roll', 'primary_IR_image_roll')]}
        fresh = values[freeze['roles']['source_RGB_fresh']]
        frozen_models = {r['id']: r for r in freeze['models']}
        require(len(frozen_models) == len(freeze['models']) and all(
            frozen_models.get(r['id']) == r['model'] for r in [*normal_best, fresh, *interventions.values()]),
            'Original frozen Test metadata substituted a Dev-selected model')
        for row in values.values():
            if row['intervention'] == 'IR_unavailable':
                require(row['official_CSV_sha256'] == fresh['official_CSV_sha256'] and all(
                    all(row['source_receipts'][seq][key] == fresh['source_receipts'][seq][key] for key in
                        ['track_sha256', 'detection_packet_sha256']) for seq in LENGTHS),
                    'Missing IR must exactly reproduce full fresh RGB packets, MOT and CSV')
        self.removal_curves(context, originals)
        for architecture in ARCHITECTURES[family]:
            initials = {self.fit_cache[family, r['fit_id']]['metadata']['initial_native_sha256'] for r in chosen
                        if r['mode' if family == 'native' else 'variant'] == architecture}
            require(len(initials) == 1, 'Matched cues had different initial native states')
        return dict(family=family, primary_id=primary_id, normal_best=normal_best,
            canonical_RGB=values[freeze['roles']['source_RGB_canonical']], fresh_RGB=fresh,
            interventions=interventions, normal_curves_complete=True, original_primary_frozen=True,
            interventions_complete=True, required_removal_curves_complete=True)

    def removal_curves(self, context, originals):
        family, paths = context['family'], context['paths']
        selection, stage = context['freeze']['selection'], context['stage']
        primary, choices = selection['primary'], context['complete']['removals']
        if family == 'native':
            enabled = selection['fit_registered_removals']
            labels = (context['study']['removals']['all_modes'] +
                (context['study']['removals']['temporal_only'] if primary['mode'] == 'temporal' else [])) if enabled else []
            if enabled:
                registration = self.read(paths['base']/'native_pyramid_removals_registered.json')
                require(registration['selection_sha256'] == self.hash(paths['choice']) and
                        registration['same_complete_Train_epochs'] == stage and registration['Test_read_for_selection'] is False,
                        'Native removals did not follow the original Dev choice/budget')
                fits = registration['fits']
                require({r['removal'] for r in fits} == set(labels) and len(fits) == len(labels) and
                        all(r['cue'] == 'ir' and r['mode'] == primary['mode'] for r in fits), 'Missing registered native removal')
                conditions = {r['removal']: r['id'] for r in fits}
            else:
                conditions = {}
        else:
            registration = self.read(paths['base']/'removals_registered.json')
            require(registration['study_sha256'] == self.hash(context['study_path']) and
                    registration['normal_selection_sha256'] == self.hash(paths['choice']), 'Spatial removal owner changed')
            plan = registration['plan']
            require(plan == context['complete']['removal_provenance'] and plan['Test_accuracy_used'] is False,
                    'Spatial removal plan changed')
            variant = next(v for v in context['study']['variants'] if v['name'] == primary['variant'])
            labels = [name for name, flag in [('no_spatial', 'use_spatial'), ('no_frame_quality', 'use_quality'),
                ('no_redundancy', 'use_redundancy')] if variant[flag]] if selection['fit_enabled_removals'] else []
            conditions = {label: r['fit_id'] for label, r in plan['conditions'].items()}
            require(set(conditions) == set(labels), 'Missing enabled spatial removal')
            for condition in plan['conditions'].values():
                path = paths['arms']/(condition['fit_id']+'.json')
                require(self.hash(path) == condition['source_arm_sha256'], 'Removal semantic source changed')
        require(set(choices) == set(conditions), 'Missing original removal choices')
        normal_ids = {r['id'] for r in context['study']['normal_fits']}
        expected_fits = normal_ids | set(conditions.values())
        require(all(r.get('baseline') or r.get('fit_id') in expected_fits for r in originals.values()),
                'Unregistered extra Dev fit')
        for label, fit_id in conditions.items():
            rows = [r for r in originals.values() if r.get('fit_id') == fit_id and r['intervention'] == 'normal']
            require(len(rows) == stage and {r['epoch'] for r in rows} == set(range(1, stage+1)),
                    'Incomplete required removal Dev curve')
            pair = choices[label]
            require(set(pair) == {'Dev_best', 'paired_primary_epoch'} and
                    all(originals.get(r['id']) == r and r['fit_id'] == fit_id for r in pair.values()) and
                    pair['paired_primary_epoch']['epoch'] == primary['epoch'], 'Removal frozen choice changed')

    def load(self):
        self.preflight()
        canonical = Path(self.plan['frozen_detection_sources']['rgb_dev'])/'trackeval_output/pedestrian_detailed.csv'
        self.csv(canonical, reference=True)
        self.canonical_csv_sha = self.hash(canonical)
        records = [self.family(self.family_context(family)) for family in ARCHITECTURES]
        require(all(len(v) == 1 for v in self.source_image_digests.values()), 'Parent families/epochs used different image bytes')
        first, second = [r['fresh_RGB'] for r in records]
        require(first['metrics'] == second['metrics'] and first['official_CSV_sha256'] == second['official_CSV_sha256'],
                'Cross-family fresh RGB official results differ')
        require(all(all(first['source_receipts'][s][k] == second['source_receipts'][s][k] for k in
            ['track_sha256', 'detection_packet_sha256']) for s in LENGTHS), 'Cross-family fresh RGB packets/tracks differ')
        selection = select_visual_parent(records)
        self.verify_unchanged()
        return dict(status='COMPLETE_ORIGINAL_PARENT_DEV_FILE_EVIDENCE', records=records,
            selection=selection, verified_files={str(k): v for k, v in self.digests.items()},
            actual_Test_accuracy_read=False, checkpoint_tensor_load_performed=False,
            strict_model_loading_GPU_engineering_and_release_still_required=True,
            formal_launch_ready=False, actual_learned_fusion_transfer=False, goal_met=False, supported_innovations=0)

    def verify_unchanged(self):
        saved = dict(self.digests)
        self.digests.clear()
        for path, digest in saved.items():
            require(self.hash(path) == digest, 'Verified parent input changed during the read: '+str(path))
