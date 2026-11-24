#!/usr/bin/env python3
"""Independent F4 B1 prefix costs after complete Dev freeze and full Test sources.

No GT/accuracy input, no fitting, no parameter choice and no bundled-condition FPS.
"""
import argparse
import gc
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_observation import FixedSequenceImages, LENGTHS, write_json
from tools.fusion_memory_online import OnlineFusionMemory
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_runtime import aggregate_cost, measure_sequence
from tools.fusion_memory_runtime_sources import runtime_references, unchanged_runtime_sources
from tools.fusion_memory_workflow import FusionMemoryWorkflow
from tools.neck72_runtime import require_host_memory, require_storage


def resource_check(work):
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*2**30, 'F4 cost worker exceeded 8GiB RSS')
    require_storage(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    workflow = FusionMemoryWorkflow()
    spec, registration = workflow.validate_specification(args.specification)
    require(spec['phase'] == 'test_frozen' and spec['split'] == 'test', 'Complete Dev freeze required before all Test costs')
    require(workflow.predecessor_ready(), 'Wait for the complete preceding GPU chain and actual exits')
    output = workflow.base/'individual_runtime.json'
    require(not output.exists(), 'Preserve existing complete individual costs')
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(workflow.work)
    teacher = workflow.parents.load_teacher_cpu()
    physical, uuid = workflow.gpu_settings('cost', args.specification)
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    calibration_path = workflow.work/'correlation_cache/calibration.json'
    calibration = read(calibration_path)
    require(sha(calibration_path) == read(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')['calibration_sha256'],
            'Original calibration changed')
    rows = []
    for case in spec['cases']:
        references, source_evidence = runtime_references(workflow, args.specification, case)
        path = workflow.base/'runtime_cases'/(case['id']+'.json')
        if path.exists():
            saved = read(path)
            require(saved['status'] == 'COMPLETE_ACTUAL_F4_INDIVIDUAL_PREFIX_COST' and saved['case'] == case and
                    saved['specification_sha256'] == sha(args.specification) and
                    saved['pipeline_registration_sha256'] == sha(workflow.base/'pipeline_registered.json') and
                    saved['source_evidence'] == source_evidence and saved['measured_frames'] == 168,
                    'Existing cost belongs to another case/source/registration')
            unchanged_runtime_sources(source_evidence)
            rows.append(saved)
            continue
        # At most one fusion computation resides on CUDA alongside the RGB teacher.
        modules, information = workflow.parents.load_routes_cpu([case])
        modules = {name: None if module is None else module.cuda().float().eval().requires_grad_(False)
                   for name, module in modules.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        for sequence in LENGTHS['test']:
            engine_factory = lambda: OnlineFusionMemory(teacher, [case], modules, (800, 1440), calibration)
            images_factory = lambda: FixedSequenceImages(workflow.plan, 'test', sequence)
            measurements.append(measure_sequence(engine_factory, images_factory, case, sequence, references[sequence],
                synchronize=torch.cuda.synchronize, resource_check=lambda: resource_check(workflow.work)))
        values = aggregate_cost(measurements)
        require(values['measured_frames'] == 168, 'All seven original 32-frame prefixes are required')
        unchanged_runtime_sources(source_evidence)
        workflow.registered()
        workflow.parents.reader.verify_unchanged()
        resource_check(workflow.work)
        fusion = sum(p.numel() for m in modules.values() if m is not None for p in m.parameters())
        row = dict(status='COMPLETE_ACTUAL_F4_INDIVIDUAL_PREFIX_COST', case=case,
            specification_sha256=sha(args.specification), pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
            source_evidence=source_evidence, measurements=measurements, **values,
            original_RGB_parameters=sum(p.numel() for p in teacher.parameters()), fusion_parameters=fusion,
            total_visual_parameters=sum(p.numel() for p in teacher.parameters())+fusion,
            original_RGB_buffer_bytes=sum(b.numel()*b.element_size() for b in teacher.buffers()),
            fusion_buffer_bytes=sum(b.numel()*b.element_size() for m in modules.values() if m is not None for b in m.buffers()),
            learned_memory_parameters=0, model_information=information,
            peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
            process_lifetime_peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
            host_RSS_scope='Cumulative process maximum, not isolated per-case host peak',
            physical_gpu=physical, GPU_uuid=str(uuid), independent_input_batch_size=1, simultaneous_configurations=1,
            sample_scope='Seven 32-frame prefixes, 8 warmup/24 timed per sequence, 168 timed frames; not full 15985-frame FPS',
            includes='Image IO/decode/RGB+IR byte hashes, preprocessing, RGB/cue/flow, heads, NMS, ROIAlign, ByteTrack+memory',
            unused_IR_bytes_hashed_in_all_controls_for_equal_source_provenance=True,
            excludes='Weight loading, offline DiffV2IR generation, verification serialization, state-byte accounting and report/final-MOT disk writing',
            actual_prefix_matches_full_production_before_and_during_timing=True,
            no_GT_or_Test_accuracy_input=True, additional_fit_or_seed_work=False, feature_or_detection_disk_cache=False)
        write_json(path, row, exclusive=True)
        rows.append(row)
        print(dict(id=case['id'], frames=168, FPS=row['FPS']), flush=True)
        del modules, information, engine_factory
        gc.collect()
        torch.cuda.empty_cache()
    require([r['case'] for r in rows] == spec['cases'], 'Every unique frozen case needs its own actual cost')
    workflow.registered()
    write_json(output, dict(status='COMPLETE_F4_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME', rows=rows,
        specification_sha256=sha(args.specification), pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),
        Test_accuracy_read=False, GT_read=False, goal_met=False, supported_innovations=0), exclusive=True)


if __name__ == '__main__':
    main()
