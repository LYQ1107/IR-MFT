"""Strict CPU model loading after actual complete Dev qualification.

No CUDA, image reader, NN fitting, Test-accuracy input or execution authorization.
Low-level tensor helpers are independently tested on artificial untrained states.
The actual entry always runs the original parent-file verifier first.
"""
import copy
import hashlib
from pathlib import Path

import torch

from tools.fusion_memory_design import normal_case_plan
from tools.fusion_memory_online import visual_routes
from tools.fusion_memory_parent_io import DevParentFiles, ROOT, require
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def stream_sha256(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        digest.update(block)
    return digest.hexdigest()


def load_pinned_torch(path, expected_sha256):
    """Bind deserialization and both content checks to the same open file."""
    path = Path(path)
    with path.open('rb') as stream:
        require(stream_sha256(stream) == expected_sha256, 'Checkpoint changed before tensor loading')
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        require(stream_sha256(stream) == expected_sha256, 'Open checkpoint changed during tensor loading')
    with path.open('rb') as stream:
        require(stream_sha256(stream) == expected_sha256, 'Checkpoint pathname changed during tensor loading')
    return saved


def strict_cpu_state(module, state):
    expected = module.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected), 'Strict complete tensor keys required')
    for key, reference in expected.items():
        value = state[key]
        require(torch.is_tensor(value) and value.device.type == 'cpu' and
                value.dtype == reference.dtype and value.shape == reference.shape,
                'Checkpoint tensor shape/dtype/device differs: '+key)
        require(bool(torch.isfinite(value).all()), 'Checkpoint tensor is nonfinite: '+key)
    module.load_state_dict(state, strict=True)
    module.eval().requires_grad_(False)
    require(all(torch.equal(value, state[key]) for key, value in module.state_dict().items()),
            'Tensor values changed during strict model loading')
    return module


def load_fusion_snapshot_cpu(path, expected_sha256, family, model, arm, progress):
    """Caller supplies original file-verified arm/progress; no choice is made."""
    require(family in ['native', 'spatial'], 'Only registered native/spatial families supported')
    saved = load_pinned_torch(path, expected_sha256)
    epoch = model['epoch']
    require(saved['completed_epochs'] == epoch and len(saved['history']) == epoch and
            saved['history'] == progress['history'][:epoch], 'Tensor checkpoint has different exact Train history')
    metadata = saved['metadata']
    require(metadata == progress['metadata'], 'Tensor checkpoint metadata differs from original progress')
    require(saved['source_RGB_only_by_reference'] is True, 'Fusion snapshot must reference the original RGB source')
    require(metadata['seed'] == model['seed'] == 0 and
            metadata['fit_id'] == model['fit_id'] == arm['fit_id'] and
            metadata['cue'] == model['cue'] == arm['cue'] and
            metadata['removal'] == model['removal'] == arm['removal'] and
            model['mode'] == arm['mode'] and model['geometry_fixed'] == arm['geometry_fixed'],
            'Tensor checkpoint architecture/cue/seed differs')
    if family == 'spatial':
        require(model['variant'] == arm['variant'] == metadata['variant'], 'Spatial tensor variant differs')
    else:
        require(metadata['mode'] == arm['mode'], 'Native tensor mode differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = (NativePyramidFusion if family == 'native' else SpatialNativeFusion)(**arm['module_kwargs'])
    return strict_cpu_state(module, saved['module'])


def load_source_rgb_cpu(path, expected_sha256):
    """Original single-class teacher only; CPU helper does not qualify F4."""
    from yolox.exp import get_exp
    saved = load_pinned_torch(path, expected_sha256)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
    # Source checkpoint is compactFP16; preserve integer buffers, convert floats
    # exactly to CPUFP32 as used in the existing real-RGB engineering reference.
    expected = model.state_dict()
    require(set(saved['model']) == set(expected), 'Source RGB tensor keys differ')
    state = {}
    for key, value in saved['model'].items():
        require(torch.is_tensor(value) and value.device.type == 'cpu' and
                value.dtype == (torch.float16 if expected[key].is_floating_point() else expected[key].dtype),
                'Unexpected original RGB compact tensor dtype: '+key)
        state[key] = value.to(dtype=expected[key].dtype)
    return strict_cpu_state(model, state)


class DevQualifiedModels:
    """Actual entry: pending parent files stop before any model tensor load."""
    def __init__(self, root=ROOT):
        self.reader = DevParentFiles(root)
        self.evidence = self.reader.load()
        require(self.evidence['selection']['chosen_family'] is not None,
                'No original IR parent passed all complete Dev strong/intervention controls')
        self.plan = normal_case_plan(self.evidence['selection'])
        self.contexts = {}

    def load_routes_cpu(self, cases):
        routes, _ = visual_routes(cases)
        allowed = self.plan['visuals']
        primary = self.plan['role_visual_keys']['primary_IR']
        fresh = self.plan['role_visual_keys']['fresh_RGB']
        for case in cases:
            key = case['visual_key']
            require(key in allowed and case['visual'] == allowed[key], 'Case substituted an unqualified visual object')
            require(case['intervention'] == 'normal' or key == primary, 'Only the selected IR parent receives interventions')
            require(case['minimum_reliability'] == .2 or key in [primary, fresh], 'Unregistered visual reliability-floor removal')
        self.reader.verify_unchanged()
        modules, information = {}, {}
        for key, route in routes.items():
            visual = route['visual']
            if visual['baseline']:
                modules[key] = None
                information[key] = dict(route, fusion_parameters=0, checkpoint_tensor_load=False)
                continue
            family, model = visual['family'], visual['model']
            if family not in self.contexts:
                self.contexts[family] = self.reader.family_context(family)
            context = self.contexts[family]
            checked = self.reader.fit(context, model)
            progress = self.reader.read(checked['progress_path'])
            require(visual['module_kwargs'] == checked['arm']['module_kwargs'], 'Visual module kwargs changed')
            module = load_fusion_snapshot_cpu(model['checkpoint'], model['checkpoint_sha256'],
                family, model, checked['arm'], progress)
            parameters = sum(p.numel() for p in module.parameters())
            expected = context['study']['architecture'][model['mode']+'_parameters' if family == 'native' else 'quality_parameters']
            require(parameters == expected, 'Loaded fusion parameter budget differs')
            modules[key] = module
            information[key] = dict(copy.deepcopy(route), fusion_parameters=parameters,
                checkpoint_tensor_load=True, checkpoint_sha256=model['checkpoint_sha256'],
                original_fit_metadata=copy.deepcopy(progress['metadata']),
                snapshot_history_equals_original_progress=True, original_RGB_only_by_reference=True)
        self.reader.verify_unchanged()
        require(len({id(m) for m in modules.values() if m is not None}) == sum(m is not None for m in modules.values()),
                'Loaded route modules must not alias')
        return modules, information

    def load_teacher_cpu(self):
        parent_path = self.reader.root/'configs/perceptual_correlation_repair_20261123.json'
        parent = self.reader.read(parent_path)
        digest = parent['source_RGB_sha256']
        require(all(self.reader.family_context(family)['study']['source_RGB_sha256'] == digest for family in ['native', 'spatial']),
                'F4 teacher differs from its original visual families')
        require(self.reader.hash(parent['source_RGB_checkpoint']) == digest, 'Original RGB teacher bytes differ')
        teacher = load_source_rgb_cpu(parent['source_RGB_checkpoint'], digest)
        self.reader.verify_unchanged()
        return teacher
