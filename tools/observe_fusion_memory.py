#!/usr/bin/env python3
"""Registered F4 production and separate CPU evaluation entry points.

Preflight is read-only. Full registration, parent eligibility, predecessor exits
and actual GPU engineering are mandatory before production CUDA is initialized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fusion_memory_evaluation import evaluate_observations, read, sha, verify_observations
from tools.fusion_memory_observation import FixedSequenceImages, observe_sequence, write_json
from tools.fusion_memory_online import OnlineFusionMemory
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.neck72_runtime import require_host_memory, require_storage


def process_identity():
    proc = Path('/proc')/str(os.getpid())
    return dict(pid=os.getpid(), start_ticks=int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19]),
                command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())


def produce(args):
    workflow = FusionMemoryWorkflow()
    spec, registration = workflow.validate_specification(args.specification)
    require(workflow.predecessor_ready(), 'Original spatial study still owns the GPU chain')
    require(args.sequence in workflow.image_anchors(spec['split']), 'A full registered sequence required')
    # Strict CPU deserialization is completed before querying/allocating CUDA.
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(workflow.work)
    teacher = workflow.parents.load_teacher_cpu()
    modules, information = workflow.parents.load_routes_cpu(spec['cases'])
    physical, uuid = workflow.gpu_settings('produce', args.specification)
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    modules = {name: None if module is None else module.cuda().float().eval().requires_grad_(False)
               for name, module in modules.items()}
    calibration_path = workflow.work/'correlation_cache/calibration.json'
    expected_calibration = read(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')['calibration_sha256']
    require(sha(calibration_path) == expected_calibration, 'Original calibration changed')
    images = FixedSequenceImages(workflow.plan, spec['split'], args.sequence)
    engine = OnlineFusionMemory(teacher, spec['cases'], modules, (800, 1440), read(calibration_path))
    anchor = spec['image_source_anchors'][args.sequence]
    require(sha(anchor['path']) == anchor['sha256'], 'Original current image anchor changed')
    expected = read(anchor['path'])['original_image_bytes_delivery_sha256']
    provenance = dict(specification_sha256=sha(args.specification), pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
        source_versions=registration['source_hashes'], physical_gpu=physical, GPU_uuid=str(uuid),
        source_RGB_sha256=read(ROOT/'configs/perceptual_correlation_repair_20261123.json')['source_RGB_sha256'],
        parent_Dev_evidence_sha256=spec['parent_Dev_evidence_sha256'],
        actual_GPU_engineering_sha256=sha(workflow.base/'gpu_engineering.json'),
        model_information=information, actual_producer_identity=process_identity(),
        strict_CPU_tensor_load_before_CUDA=True, original_RGB_parameters_frozen=True,
        GT_input=False, future_input=False)
    def resource_check():
        require_storage(workflow.work)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*2**30, 'F4 producer exceeded8GiB host bound')
    result = observe_sequence(engine, images, spec['cases'], spec['stream_directory'], expected, provenance,
                              resource_check=resource_check)
    workflow.registered()
    workflow.parents.reader.verify_unchanged()
    print(json.dumps(dict(sequence=args.sequence, frames=result['frames'], cases=len(spec['cases']),
        physical_gpu=physical, status=result['status'])), flush=True)


def evaluate(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Official F4 evaluation must be CPU only')
    workflow = FusionMemoryWorkflow()
    spec, registration = workflow.validate_specification(args.specification)
    matches = [c for c in spec['cases'] if c['id'] == args.case]
    require(len(matches) == 1, 'Exact registered F4 case ID required')
    case = matches[0]
    # Complete inference is checked before opening any GT file.
    manifest = verify_observations(args.specification, case, spec['stream_directory'],
        sha(workflow.base/'pipeline_registered.json'), registration['source_hashes'], spec['image_source_anchors'])
    gt = workflow.gt_hashes(spec['split'])
    output = workflow.base/'evaluation'/spec['phase']/case['id']
    values = evaluate_observations(args.specification, case, spec['stream_directory'],
        sha(workflow.base/'pipeline_registered.json'), registration['source_hashes'], spec['image_source_anchors'], output, workflow.plan)
    require(workflow.gt_hashes(spec['split']) == gt, 'Protocol GT bytes changed during F4 evaluation')
    require(values['manifest'] == manifest, 'Evaluated manifest differs from checked complete inference')
    workflow.registered()
    write_json(output/'driver_validation.json', dict(status='COMPLETE_REGISTERED_F4_METRIC_DRIVER_VALIDATION',
        metrics_sha256=sha(output/'metrics.json'), specification_sha256=sha(args.specification),
        parent_Dev_evidence_sha256=spec['parent_Dev_evidence_sha256'], protocol_GT_file_sha256=gt,
        pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
        actual_producer_specification_revalidated=True, GT_inference_input=False,
        no_Test_accuracy_input_for_Dev_selection=True), exclusive=True)
    print(json.dumps(dict(id=case['id'], split=spec['split'], frames=manifest['frames'], status=values['status'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'produce', 'evaluate'])
    parser.add_argument('--specification', type=Path)
    parser.add_argument('--sequence')
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.action == 'preflight':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preflight must be CPU only')
        print(json.dumps(FusionMemoryWorkflow().preflight(), indent=2), flush=True)
    else:
        require(args.specification is not None, 'Registered specification required')
        if args.action == 'produce':
            require(args.sequence is not None, 'Registered complete sequence required')
            produce(args)
        else:
            require(args.case is not None, 'Registered case required')
            evaluate(args)


if __name__ == '__main__':
    main()
