#!/usr/bin/env python3
"""One actual source-matched TaskUtility/B1 or historical RGB/B4 prefix cost."""
import argparse
from pathlib import Path
import resource
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from tools.fusion_memory_observation import write_json
from tools.native_pyramid_experiment import gpu_settings,load_teacher,read
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.neck72_runtime import require_host_memory,require_storage
from tools.observe_task_utility import load_modules
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_runtime import CanonicalRGBBatch,RuntimeImages,aggregate_runtime,measure_prefix
from tools.task_utility_runtime_sources import RUNTIME,runtime_references,unchanged_runtime_sources
from tools.task_utility_study import STUDY,REGISTRY,context,predecessor_ready,registered


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--specification',required=True)
    parser.add_argument('--config',required=True);args=parser.parse_args()
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    if not predecessor_ready(work):raise RuntimeError('Actual complete prior queue export and exits required before CUDA')
    evidence=runtime_references(args.specification,args.config,base,registration,plan,protocol)
    case=evidence['case']['model'];canonical=evidence['case']['canonical']
    destination=base/'individual_runtime'/(case['id']+'.json')
    if destination.exists():raise ValueError('Preserve any prior individual cost result')
    if registration['source_hashes'].get(str(RUNTIME.relative_to(ROOT)))!=sha256(RUNTIME):
        raise ValueError('The runtime clarification must be part of the complete workflow registration')
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    physical,uuid=gpu_settings()
    if canonical:
        torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False
    teacher=load_teacher(checkpoint,training=False)
    if canonical:
        modules={};information={'route':'Actual original-source B4RGB NN,no auxiliary module.'}
        factory=lambda:CanonicalRGBBatch(teacher,case,study['inference_input_hw'])
    else:
        modules,information=load_modules([case],study,base,registration,'cuda')
        factory=lambda:OnlineNativeDetector(teacher,[case],modules,study['inference_input_hw'],calibration)
    def resources():
        require_storage(work)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Own runtime exceeded8GiB')
    sequences=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for sequence in protocol['historical_test']:
        sequences.append(measure_prefix(factory,lambda:RuntimeImages(plan,'test',sequence),case,sequence,
            evidence['references'][sequence],synchronize=torch.cuda.synchronize,resources=resources))
    costs=aggregate_runtime(sequences)
    if costs['measured_frames']!=168 or costs['NN_batch_size']!=(4 if canonical else 1):
        raise ValueError('All seven actual prefixes and original batch policy are mandatory')
    unchanged_runtime_sources(evidence);registered(work);resources()
    result=dict(status='COMPLETE_TASK_UTILITY_INDIVIDUAL_SOURCE_MATCHED_PREFIX_COST',case=case,
        route='historical_canonical_RGB_B4' if canonical else 'current_image_B1',costs=costs,sequences=sequences,
        study_sha256=sha256(STUDY),runtime_clarification_sha256=sha256(RUNTIME),
        pipeline_registration_sha256=sha256(base/REGISTRY),specification_sha256=sha256(args.specification),
        config_sha256=sha256(args.config),original_file_hashes=evidence['file_hashes'],model_information=information,
        physical_gpu=physical,GPU_uuid=uuid,original_RGB_parameters=sum(p.numel() for p in teacher.parameters()),
        auxiliary_parameters=sum(p.numel() for module in modules.values() if module is not None for p in module.parameters()),
        peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        original_RGB_backbone_and_head_precision='FP16',auxiliary_precision='FP32_when_present',
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        source_and_output_equivalence_before_and_during_timing=True,methods_measured_simultaneously=1,
        complete_full_sequence_FPS_measured=False,prefix_frames_each_video=32,warmup_frames_each_video=8,
        includes_IO_decode_required_modality_hash_preprocessing_NN_NMS_hostcopy_ByteTrack=True,
        validation_serialization_and_resource_accounting_outside_timer=True,gate_observer_attached=False,
        includes_original_workers4_loader=False,offline_DiffV2IR_generation_and_weight_loading_excluded=True,
        generated_IR_changed=False,image_or_FPN_disk_cache=False,measurement_repeats=1,fixed_seed=0,
        GT_read=False,Test_accuracy_read=False,goal_met=False,supported_innovations=0,
        scope='Seven32frame prefixes,8warmup24timed each,source-equivalent independent verification thenone measured pass. '
            'Includes component clock/synchronization overhead; OS page cache may be warm after verification. '
            'Historical B4 reference uses4actual observations/batch and reports batch latency,not B1latency or original4worker fullTest throughput. '
            'All other25cases use the actual unbundled B1production path. No packet-replay FPS or historical B4/B1 alias.')
    write_json(destination,result,exclusive=True)
    print(dict(id=case['id'],batch=costs['NN_batch_size'],measured_frames=168,FPS=costs['FPS']),flush=True)


if __name__=='__main__':main()
