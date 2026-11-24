"""F4 context, Dev-bound specifications and complete predecessor release checks.

No implicit registration, fitting or CUDA. Missing full-pipeline components are
reported by preflight; actual parent qualification precedes tensor/device work.
"""
import json
import os
from pathlib import Path

from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_design import extra_cases, freeze_case_plan, select_memory
from tools.fusion_memory_dev_freeze import exact_behavior, load_dev_files
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_model_loading import DevQualifiedModels
from tools.fusion_memory_observation import LENGTHS, write_json
from tools.fusion_memory_parent_io import DevParentFiles, ParentEvidencePending, ROOT, require
from tools.neck72_runtime import require_host_memory, require_storage
from tools.neck72_runtime_study import exited

WORKFLOW = ROOT/'configs/perceptual_fusion_memory_workflow_20261125.json'
STUDY = ROOT/'configs/perceptual_fusion_memory_study_20261125.json'


def release_header(path):
    """Only these three leading strings; never materialize later Test metrics."""
    fields = {'status', 'study_sha256', 'registry_sha256'}
    values = {}
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line == '{':
                continue
            key = line.split(':', 1)[0].strip('"')
            require(key in fields and key not in values, 'Unexpected field before safe release header completed')
            item = json.loads('{'+line.rstrip(',')+'}')
            require(set(item) == {key} and isinstance(item[key], str), 'Safe release header must contain strings')
            values.update(item)
            if set(values) == fields:
                return values
    raise ValueError('Incomplete safe release header')


def release_state_ready(state, identity, header, process_exited=exited):
    status = state.get('status', '')
    require('FAIL' not in status and 'MISMATCH' not in status, 'Preserve/resolve failed spatial predecessor before F4 CUDA')
    if status != 'COMPLETE_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT':
        return False
    if header.get('status') != 'COMPLETE_NINE_ARM_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT':
        return False
    require(state.get('study_sha256') == header.get('study_sha256') and
            state.get('registry_sha256') == header.get('registry_sha256') and
            state.get('controller_pid') == identity.get('pid'), 'Spatial release versions/process identity differ')
    if not process_exited(identity['pid'], identity=identity):
        return False
    phases = state.get('phases', {})
    if not phases or any(not entries for entries in phases.values()):
        return False
    for entries in phases.values():
        for job in entries.values():
            if (job.get('status') != 'COMPLETE' or job.get('exit_code') != 0 or
                    not job.get('start_ticks') or not job.get('command')):
                return False
            if not process_exited(job['pid'], command=job['command'], identity=job):
                return False
    return True


def complete_phase_cases(plan, base, phase, schema):
    """Recompute frozen sets from complete Dev files; never read Test accuracy."""
    if phase == 'dev_normal':
        return plan['normal_cases'], {}
    require(phase in ['dev_additional', 'test_frozen'], 'Unknown formal F4 phase')
    normal_path = Path(base)/'normal_Dev_selection.json'
    original = read(normal_path)
    files = {k: r['path'] for k, r in original['normal_source_metrics'].items()}
    normal, normal_values, references = load_dev_files(plan, plan['normal_cases'], files, schema)
    selection = select_memory(plan, normal)
    additions = extra_cases(plan, selection)
    require(original['status'] == 'FROZEN_F4_NORMAL_DEV_FILES_BEFORE_INTERVENTIONS' and original['normal_scores'] == normal and
            original['selection'] == selection and original['normal_source_metrics'] == references and
            original['additional_cases'] == additions and original['Test_accuracy_read'] is False,
            'Frozen normal Dev choice changed')
    basis = dict(normal_Dev_choice_sha256=sha(normal_path))
    if phase == 'dev_additional':
        return additions, basis
    final_path = Path(base)/'frozen_Test_case_plan.json'
    frozen = read(final_path)
    require(frozen['status'] == 'FROZEN_F4_ALL_COMPLETE_DEV_METRIC_FILES_PENDING_EXECUTION_GATES' and
            frozen['original_normal_choice_sha256'] == sha(normal_path) and frozen['Test_accuracy_read'] is False,
            'Complete Dev Test choice is missing or inconsistent')
    additional_files = {k: r['path'] for k, r in frozen['additional_source_metrics'].items()}
    extra, extra_values, extra_references = load_dev_files(plan, additions, additional_files, schema)
    require(frozen['normal_source_metrics'] == references and frozen['additional_source_metrics'] == extra_references,
            'Frozen complete Dev file references changed')
    for value in extra_values.values():
        spec = read(value['manifest']['specification'])
        require(spec['normal_Dev_choice_sha256'] == sha(normal_path), 'Interventions were not bound to original normal choice')
    expected = freeze_case_plan(plan, normal, extra)
    require(all(frozen[k] == v for k, v in expected.items() if k != 'status'), 'Frozen Test case set differs from complete Dev files')
    values = {**normal_values, **extra_values}
    fresh = plan['role_visual_keys']['fresh_RGB']
    for case in additions:
        if case['intervention'] != 'IR_unavailable':
            continue
        control = next(c for c in expected['cases'] if c['visual_key'] == fresh and c['intervention'] == 'normal' and
            c['appearance_weight'] == case['appearance_weight'] and (c['profile'] == 'byte' or
                (c['profile'] == 'rgb_uniform' and c['minimum_reliability'] == case['minimum_reliability'])))
        exact_behavior(values[case['id']], values[control['id']])
    return expected['cases'], dict(basis, complete_Dev_Test_choice_sha256=sha(final_path))


class FusionMemoryWorkflow:
    def __init__(self):
        self.config, self.study = read(WORKFLOW), read(STUDY)
        require(all(sha(ROOT/p) == s for p, s in self.config['frozen_dependencies'].items()), 'Frozen F4 workflow dependencies changed')
        self.plan = read(ROOT/'configs/perceptual_mft_20261122.json')
        self.protocol = read(ROOT/self.plan['protocol'])
        self.work = Path(self.plan['work_root'])
        self.base = self.work/'fusion_memory'
        self._parents = None

    @property
    def parents(self):
        if self._parents is None:
            self._parents = DevQualifiedModels()
        return self._parents

    def predecessor_ready(self):
        base = self.work/'native_spatial'
        state_path = base/'state.json'
        if not state_path.exists():
            return False
        state = read(state_path)
        require('FAIL' not in state.get('status', ''), 'Preserve failed spatial predecessor')
        if state.get('status') != 'COMPLETE_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT':
            return False
        report = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_FORMAL.json'
        if not report.exists():
            return False
        require(state['study_sha256'] == sha(ROOT/self.study['parent_studies']['spatial']['path']) and
                state['registry_sha256'] == sha(base/'pipeline_registered.json'), 'Spatial predecessor version changed')
        return release_state_ready(state, read(base/'controller_identity.json'), release_header(report))

    def registered(self):
        required = self.config['complete_pipeline_entrypoints']
        require(all((ROOT/p).is_file() for p in required), 'Complete F4 production/probe/cost/scheduler/exporter required')
        path = self.base/'pipeline_registered.json'
        require(path.exists(), 'Complete F4 workflow registration is missing')
        value = read(path)
        require(value['status'] == 'REGISTERED_COMPLETE_F4_PIPELINE_PENDING_REAL_GPU_ENGINEERING' and
                value['study_sha256'] == sha(STUDY) and value['workflow_sha256'] == sha(WORKFLOW) and
                set(required).issubset(value['source_hashes']), 'Wrong or partial F4 pipeline registration')
        require(all(sha(ROOT/p) == s for p, s in value['source_hashes'].items()) and
                all(sha(p) == s for p, s in value['artifact_hashes'].items()), 'F4 registered code/artifact changed')
        proof_path = Path(value['complete_workflow_CPU_proof'])
        require(sha(proof_path) == value['complete_workflow_CPU_proof_sha256'], 'Integrated workflow CPU proof changed')
        proof = read(proof_path)
        require(proof['status'] == 'COMPLETE_F4_WORKFLOW_CPU_INTEGRATION' and
                proof['source_hashes'] == value['source_hashes'], 'Complete F4 integrated workflow proof required')
        return value

    def preflight(self):
        reader = DevParentFiles()
        try:
            reader.preflight()
            missing = []
        except ParentEvidencePending as pending:
            missing = pending.paths
        identity = self.work/'native_spatial/controller_identity.json'
        current = read(identity) if identity.exists() else None
        live = current is not None and not exited(current['pid'], identity=current)
        return dict(status='F4_CPU_PREFLIGHT_NO_EXECUTION', missing_parent_markers=missing,
            parent_eligibility=None, predecessor_ready=self.predecessor_ready(), spatial_controller_identity=current,
            spatial_controller_confirmed_live=live,
            missing_pipeline_entrypoints=[p for p in self.config['complete_pipeline_entrypoints'] if not (ROOT/p).is_file()],
            complete_registration_present=(self.base/'pipeline_registered.json').exists(),
            new_GPU_work=False, tensor_loads=0, new_accuracy=False, formal_launch_ready=False)

    def freeze_parent_snapshot(self):
        path = self.base/'parent_Dev_files_verified.json'
        expected = self.parents.evidence
        if path.exists():
            require(read(path) == expected, 'Previously frozen F4 parent evidence changed')
        else:
            write_json(path, expected, exclusive=True)
        return path

    def image_anchors(self, split):
        require(split in ['dev', 'test'], 'Only original formal Dev/Test anchors')
        phase = 'dev_normal_06' if split == 'dev' else 'test_frozen'
        path = self.work/'native_pyramid_specs'/(phase+'.json')
        spec = read(path)
        require(spec['split'] == split and spec['study_sha256'] == self.study['parent_studies']['native']['sha256'], 'Original Native source spec differs')
        if split == 'test':
            freeze = read(self.work/'native_pyramid_Test_selection_frozen.json')
            require(freeze['specification_sha256'] == sha(path) and freeze['complete_Dev_before_Test'] is True,
                    'Original Native Test sources lack original Dev freeze')
        result = {}
        for seq, frames in LENGTHS[split].items():
            source = Path(spec['stream_directory'])/'sequence_manifests'/(seq+'.json')
            value = read(source)
            require(value['status'] == 'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT' and
                    value['sequence'] == seq and value['split'] == split and value['frames'] == frames and
                    value['GT_read'] is False and value['specification_sha256'] == sha(path), 'Original source image anchor incomplete')
            result[seq] = dict(path=str(source), sha256=sha(source))
        return result

    def phase_cases(self, phase):
        schema, _ = official_schema(self.plan['evaluation']['TrackEval_root'])
        return complete_phase_cases(self.parents.plan, self.base, phase, schema)

    def write_specification(self, phase):
        self.registered()
        parent = self.freeze_parent_snapshot()
        cases, basis = self.phase_cases(phase)
        split = 'test' if phase == 'test_frozen' else 'dev'
        value = dict(status='FROZEN_F4_CASES_BEFORE_OBSERVATION_PRODUCTION', phase=phase, split=split,
            cases=cases, study_sha256=sha(STUDY), workflow_sha256=sha(WORKFLOW),
            pipeline_registration_sha256=sha(self.base/'pipeline_registered.json'),
            parent_Dev_evidence_sha256=sha(parent), stream_directory=str(self.base/'streams'/phase),
            image_source_anchors=self.image_anchors(split), **basis)
        path = self.base/'specifications'/(phase+'.json')
        if path.exists():
            require(read(path) == value, 'Preserve the originally registered observation specification')
        else:
            write_json(path, value, exclusive=True)
        return path

    def validate_specification(self, path):
        path = Path(path).resolve()
        require(path.parent == (self.base/'specifications').resolve(), 'Use registered F4 specification directory')
        value = read(path)
        registration = self.registered()
        require(value['status'] == 'FROZEN_F4_CASES_BEFORE_OBSERVATION_PRODUCTION' and
                value['study_sha256'] == sha(STUDY) and value['workflow_sha256'] == sha(WORKFLOW) and
                value['pipeline_registration_sha256'] == sha(self.base/'pipeline_registered.json'), 'F4 spec version mismatch')
        parent = self.base/'parent_Dev_files_verified.json'
        require(sha(parent) == value['parent_Dev_evidence_sha256'] and read(parent) == self.parents.evidence,
                'F4 parent snapshot differs from actual complete Dev')
        cases, basis = self.phase_cases(value['phase'])
        split = 'test' if value['phase'] == 'test_frozen' else 'dev'
        require(value['cases'] == cases and value['split'] == split and all(value[k] == v for k, v in basis.items()) and
                Path(value['stream_directory']).resolve() == (self.base/'streams'/value['phase']).resolve() and
                value['image_source_anchors'] == self.image_anchors(split), 'Case/phase/source anchors differ from actual Dev freeze')
        return value, registration

    def gt_hashes(self, split):
        require(split in ['dev', 'test'], 'Only registered evaluation splits')
        folder = 'train' if split == 'dev' else 'test'
        result = {}
        for seq in LENGTHS[split]:
            path = Path(self.plan['evaluation']['GT_root'])/folder/seq/'gt/gt.txt'
            require(sha(path) == self.protocol['derived_gt_sha256'][folder+'/'+seq], 'Original ProtocolV2 GT bytes changed')
            result[str(path)] = sha(path)
        return result

    def gpu_settings(self, purpose, specification):
        require(purpose in ['probe', 'produce', 'cost'], 'Unknown F4 GPU purpose')
        self.validate_specification(specification)
        require(self.predecessor_ready(), 'Wait for complete spatial export and all original processes to exit')
        if purpose != 'probe':
            proof = self.base/'gpu_engineering.json'
            require(proof.exists(), 'Actual GPU F4 engineering must complete before production/cost')
            value = read(proof)
            require(value['status'] == 'COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING' and
                    value['normal_specification_sha256'] == sha(self.base/'specifications/dev_normal.json') and
                    value['pipeline_registration_sha256'] == sha(self.base/'pipeline_registered.json'), 'Wrong GPU engineering proof')
        require_host_memory(8*2**30, maximum_workers=1)
        require_storage(self.work)
        physical = os.environ.get('CUDA_VISIBLE_DEVICES')
        require(physical in ['4', '5'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Use physical4/5 and fixed CuBLAS configuration')
        # Do not invoke any CUDA API until every preceding file/process check passes.
        from tools.native_pyramid_experiment import gpu_settings
        return gpu_settings()
