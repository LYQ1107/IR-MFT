#!/usr/bin/env python3
"""Actual F4 GPU qualification after the complete pipeline and predecessor exits.

The component's CPU/artificial tests cannot create this real GPU qualification.
No fitting, Test score, GT, full-sequence result or performance claim.
"""
import argparse
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.fusion_memory_engineering import (EngineeringMismatch, compare_prefix, engineering_cases,
    original_Dev_prefixes, require_active_primary_memory, unchanged_Dev_prefixes)
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_gpu_checks import known_plane_ROI
from tools.fusion_memory_observation import FixedSequenceImages, write_json
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage


def all_model_hashes(teacher, modules):
    return dict(RGB=state_sha256(teacher.state_dict()),
        fusion={name: None if module is None else state_sha256(module.state_dict()) for name, module in modules.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    workflow = FusionMemoryWorkflow()
    spec, registration = workflow.validate_specification(args.specification)
    require(spec['phase'] == 'dev_normal' and spec['split'] == 'dev', 'Only original normal Dev may qualify F4 GPU engineering')
    require(workflow.predecessor_ready(), 'Wait for complete predecessor export and actual process exits')
    output = workflow.base/'gpu_engineering.json'
    require(not output.exists(), 'Preserve existing real GPU qualification')
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(workflow.work)
    for name, folder in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')]:
        require(os.environ.get(name) == str(workflow.base/'kernel_cache'/folder),
                'Complete registered launcher must keep compiler caches inside bounded F4 work storage')
    cases = engineering_cases(workflow.parents.plan)
    originals, pins, source_rows = original_Dev_prefixes(workflow, cases)
    teacher = workflow.parents.load_teacher_cpu()
    reference_teacher = workflow.parents.load_teacher_cpu()
    modules, information = workflow.parents.load_routes_cpu(cases)
    reference_modules, _ = workflow.parents.load_routes_cpu(cases)
    # Every CPU deserialization and source check above precedes any CUDA API.
    physical, uuid = workflow.gpu_settings('probe', args.specification)
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    reference_teacher = reference_teacher.cuda().half().eval().requires_grad_(False)
    modules = {name: None if value is None else value.cuda().float().eval().requires_grad_(False) for name, value in modules.items()}
    reference_modules = {name: None if value is None else value.cuda().float().eval().requires_grad_(False) for name, value in reference_modules.items()}
    require(next(teacher.parameters()).is_cuda and next(reference_teacher.parameters()).is_cuda and
            all(value is None or next(value.parameters()).is_cuda for value in [*modules.values(), *reference_modules.values()]),
            'Real GPU tensors required, not a CPU engineering receipt')
    before = all_model_hashes(teacher, modules)
    require(all_model_hashes(reference_teacher, reference_modules) == before, 'Independent loaded GPU weights differ')
    calibration_path = workflow.work/'correlation_cache/calibration.json'
    require(sha(calibration_path) == read(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')['calibration_sha256'],
            'Original calibration changed')
    calibration = read(calibration_path)
    routes, _ = visual_routes(cases)
    results = []
    def resources():
        require_storage(workflow.work)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*2**30, 'F4 GPU probe exceeded 8GiB host RSS')
    try:
        coordinates=known_plane_ROI(next(teacher.parameters()).device)
        require(coordinates['actual_CUDA'] is True,'Known-plane CPU result cannot qualify actual GPU geometry')
        for sequence in ['BT-005', 'SN-013']:
            composed = OnlineFusionMemory(teacher, cases, modules, (800, 1440), calibration)
            reference = OnlineNativeDetector(reference_teacher, [r['model'] for r in routes.values()],
                reference_modules, (800, 1440), calibration)
            images = FixedSequenceImages(workflow.plan, 'dev', sequence)
            results.append(compare_prefix(composed, reference, images, cases, originals[sequence],
                expected_source_rows=source_rows.get(sequence), resource_check=resources))
        require_active_primary_memory(workflow.parents.plan, results)
        require(all_model_hashes(teacher, modules) == before and all_model_hashes(reference_teacher, reference_modules) == before,
                'RGB or fusion parameters/BN buffers changed during GPU engineering')
        require(all(p.grad is None for model in [teacher, reference_teacher, *modules.values(), *reference_modules.values()]
                    if model is not None for p in model.parameters()), 'GPU inference unexpectedly created gradients')
        unchanged_Dev_prefixes(pins)
        workflow.registered()
        workflow.parents.reader.verify_unchanged()
        resources()
        write_json(output, dict(status='COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING',
            normal_specification_sha256=sha(args.specification), pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
            parent_Dev_evidence_sha256=spec['parent_Dev_evidence_sha256'], physical_gpu=physical, GPU_uuid=str(uuid),
            source_versions=registration['source_hashes'], original_Dev_source_pins=pins, model_information=information,
            cases=cases, results=results, actual_GPU_tensors_verified=True, model_state_hashes=before,
            actual_GPU_ROI_coordinate_check=coordinates,
            actual_loaded_primary_fused_memory_and_positive_cost_exercised=True,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            scope='Two32-frame Dev prefixes; independent NN and original zero-MOT/KF/ID checks plus captured-current-ROI memory replay. No full Dev/Test or FPS.',
            GT_read=False, Test_accuracy_read=False, full_Dev_Test_coverage=False, new_accuracy=False,
            image_FPN_or_ROI_disk_cache=False, fixed_seed=0, additional_seed_work=False), exclusive=True)
    except Exception as error:
        write_json(workflow.base/(f'gpu_engineering.failure_{time.time_ns()}.json'),
            dict(error=repr(error), details=error.details if isinstance(error, EngineeringMismatch) else None,
                normal_specification_sha256=sha(args.specification), physical_gpu=physical, qualification=False,
                original_outputs_retained=True), exclusive=True)
        raise
    print(dict(status='COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING', physical_gpu=physical), flush=True)


if __name__ == '__main__':
    main()
