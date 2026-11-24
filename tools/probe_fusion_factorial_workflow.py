#!/usr/bin/env python3
"""Complete source closure, CPU failure-path tests and real Train canaries."""
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch

from tools.export_fusion_factorial import official_schema,read_table
from tools.fusion_factorial_workflow import SCIENCE,context,make_joint_arm,read
from tools.launch_fusion_factorial import source_files
from tools.native_pyramid_content_guard import PinnedTrainImageReader
from tools.native_pyramid_train_data import epoch_clips
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.train_fusion_factorial import canary


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    plan,study,protocol,calibration,work,checkpoint,base=context()
    output=work/'fusion_factorial_complete_CPU_proof.json';assert not output.exists()
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    sources={name:sha256(ROOT/name) for name in source_files(work)}
    started=time.monotonic();log=work/'fusion_factorial_complete_CPU_pytest.log'
    test_names=['rules','workflow','observations','runtime_prefix','selection','controller','export']
    command=[sys.executable,'-m','pytest','-q']+['tests/test_fusion_factorial_'+name+'.py' for name in test_names]
    with log.open('x') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    match=re.search(r'\b(\d+) passed\b',log.read_text());assert match and int(match.group(1))>=84
    schema,counts=official_schema(plan['evaluation']['TrackEval_root'])
    metric_sources={}
    for name in ['hota','clear','identity','count','_base_metric']:
        path=Path(sys.modules['trackeval.metrics.'+name].__file__).resolve()
        assert path.is_relative_to(Path(plan['evaluation']['TrackEval_root']).resolve())
        metric_sources[str(path)]=sha256(path)
    path=Path(sys.modules['trackeval._timing'].__file__).resolve()
    assert path.is_relative_to(Path(plan['evaluation']['TrackEval_root']).resolve())
    metric_sources[str(path)]=sha256(path)
    dev_csv=ROOT/'docs/fusion_research/perceptual_20261122_official_tables/strong_rgb24_dev/F_rgb_rgb_s0_e24.csv'
    dev_sha=sha256(dev_csv);converted,provenance=read_table(dev_csv,schema,protocol)
    assert len(converted)==3 and all(row['split']=='dev' for row in converted)
    assert next(r for r in converted if r['seq']=='COMBINED')['CLR_Frames']==4834
    clips=list(epoch_clips(study['Train_delivery']['lengths'],1))[:2];expected={}
    for clip in clips:
        for frame in clip.frames:
            for modality,data_root,folder,suffix in [('rgb',plan['data_root'],'visible','jpg'),
                    ('ir',plan['generated_ir_root'],'infrared_diffv2ir','png')]:
                path=Path(data_root)/'train'/clip.sequence/folder/f'{frame:06d}.{suffix}'
                expected[clip.sequence,frame,modality]=dict(sha256=sha256(path),bytes=path.stat().st_size)
    reader=PinnedTrainImageReader(plan['data_root'],plan['generated_ir_root'],protocol['train'],
        expected_content=expected,max_bytes=256*2**20,reserve_bytes=40*2**30)
    from yolox.exp import get_exp
    teacher=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model().float().eval().requires_grad_(False)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    teacher.load_state_dict(saved['model'],strict=True);del saved
    prior=read(work/'fusion_factorial_training_worker_CPU_proof.json');records=[]
    for mode in study['modes']:
        row=canary(teacher,reader,plan,study,protocol,calibration,make_joint_arm(study,mode),'cpu',2)
        old=next(r for r in prior['records'] if r['mode']==mode)
        assert row['initial_native_sha256']==old['initial_native_sha256']
        assert row['delivered_prefix_sha256']==old['delivered_prefix_sha256']
        records.append(row);print(json.dumps(dict(mode=mode,full_source_CPU_canary_passed=True)),flush=True)
    assert not torch.cuda.is_initialized() and len(reader.content_hashes)==16
    assert sources=={name:sha256(ROOT/name) for name in source_files(work)}
    assert all(sha256(p)==h for p,h in metric_sources.items()) and sha256(dev_csv)==dev_sha
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30;assert peak<=8
    require_storage(work)
    report=dict(status='COMPLETE_CPU_FACTORIAL_WORKFLOW_AND_REAL_TRAIN_CANARIES',
        source_hashes=sources,science_sha256=sha256(SCIENCE),tests_passed=int(match.group(1)),
        test_command=command,test_log_sha256=sha256(log),
        official_metric_schema_source_hashes=metric_sources,official_metric_family_counts=counts,
        real_existing_seed0_Dev_CSV_provenance=provenance,
        real_Train_worker_canaries=records,actual_Train_input_content=reader.content_manifest(),
        unique_Train_RGB_frames=8,unique_current_November_IR_frames=8,total_CPU_updates=4,
        full_resolution_B1_Train_frame_conditions=4,
        complete_followup_pipeline_implemented=True,actual_GPU_engineering=False,GPU_work=False,
        formal_fit_started=False,Dev_Test_inference_run=False,new_accuracy=False,
        Test_accuracy_used_for_selection=False,weights_saved=0,image_or_FPN_disk_copies=0,
        prior_training_worker_CPU_proof_sha256=sha256(work/'fusion_factorial_training_worker_CPU_proof.json'),
        peak_process_RSS_GiB=peak,seconds=time.monotonic()-started,
        scope='Full source set checked before/after controller failure-path, observation/freeze, artifact-reuse, prefix, and47field tests plus real Train worker canaries. CPU FP32 only. Existing Dev CSV used only for export validation. No real full Train/Dev/Test run, actual GPU/FP16 parity, new HOTA, measured FPS or independent innovation.',
        goal_met=False,supported_innovations=0)
    dump(output,report);print(report['status'],report['tests_passed'],'tests',len(sources),'sources',flush=True)


if __name__=='__main__':main()
