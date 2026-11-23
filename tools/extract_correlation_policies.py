#!/usr/bin/env python3
"""Independent-sample current-image fusion, shared across frozen snapshots."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.exp import get_exp
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def detections(output,ratio):
    if output is None:return np.empty((0,5),np.float32)
    value=output.float().cpu().numpy()
    return np.c_[value[:,:4]/ratio,value[:,4]*value[:,5]].astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--specification')
    p.add_argument('--source-check',type=int,default=0);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--ir-intervention',choices=['none','zero','spatial_roll'],default='none')
    p.add_argument('--producer-name',choices=['correlation_policy_predictions','correlation_replication_predictions'],default='correlation_policy_predictions')
    a=p.parse_args()
    if a.batch_size!=4:raise ValueError('Keep the original independent-sample baseline batch shape4')
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPU2/3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';config=json.loads(config_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    if a.sequence not in protocol['dev'if a.split=='dev'else'historical_test']:raise ValueError('Outside frozen split')
    if a.split=='test'and not(work/'correlation_selection_frozen.json').exists():raise ValueError('Freeze complete Dev choices before Test')
    calibration_path=work/'correlation_cache/calibration.json';calibration=json.loads(calibration_path.read_text())
    if calibration['config_sha256']!=sha256(config_path):raise ValueError('Training normalization changed')
    if a.source_check:
        specification=dict(models=[dict(id='source_RGB',cue='rgb',baseline=True)]+
            [dict(id='zero_'+cue,cue=cue,zero_adapter=True)for cue in ['rgb','gray','ir']],
            status='UNLABELED_SOURCE_CHECK',config_sha256=sha256(config_path))
        spec_sha256=None
    else:
        specification=json.loads(Path(a.specification).read_text());spec_sha256=sha256(a.specification)
        if specification['config_sha256']!=sha256(config_path):raise ValueError('Registered evaluation changed')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45,0)
    checkpoint=Path(config['source_RGB_checkpoint'])
    if sha256(checkpoint)!=config['source_RGB_sha256']:raise ValueError('Frozen source RGB checkpoint changed')
    model=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    model=model.cuda().half().eval().requires_grad_(False)
    loaded={};model_info={};needed={'rgb'}
    for row in specification['models']:
        name=row['id'];cue=row['cue'];needed.add(cue)
        if name in loaded:raise ValueError('Duplicate snapshot name')
        if row.get('baseline'):loaded[name]=None;model_info[name]=dict(row);continue
        variant=row.get('variant','full');architecture=config['architecture']
        adapter=CorrelationRepair(latent=architecture['latent_channels'],radius=0 if variant=='aligned'else architecture['local_radius'],
            temperature=architecture['temperature'],risk_gate=variant!='no_gate',bound=architecture['maximum_residual_RMS_fraction'])
        if not row.get('zero_adapter'):
            path=Path(row['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=False)
            if (saved['config_sha256']!=sha256(config_path)or saved['metadata']['cue']!=cue
                    or saved['completed_epochs']!=row['epoch']or saved['metadata']['variant']!=variant
                    or saved['metadata']['seed']!=row['seed']or saved['metadata']['source_RGB_sha256']!=config['source_RGB_sha256']):
                raise ValueError('Frozen trained adapter provenance mismatch')
            adapter.load_state_dict(saved['adapter'],strict=True)
            model_info[name]=dict(row,checkpoint_sha256=sha256(path),metadata=saved['metadata'])
        else:model_info[name]=dict(row)
        loaded[name]=adapter.cuda().float().eval().requires_grad_(False)
    source=Path(plan['frozen_detection_sources']['rgb_'+a.split])/'detections'/(a.sequence+'.npz')
    with np.load(source)as z:reference={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
    limit=min(len(reference['frame_ids']),a.source_check or len(reference['frame_ids']))
    producer_name=a.producer_name if a.ir_intervention=='none'else a.producer_name+'_'+a.ir_intervention
    out=work/producer_name/a.split/a.sequence
    if a.source_check:out=out/f'source_check_{a.source_check}'
    out.mkdir(parents=True,exist_ok=True)
    if(out/'complete.json').exists():raise ValueError('Completed correlation predictions exist')
    buffers={name:[]for name in loaded}
    for row in specification['models']:
        if not row.get('baseline'):
            buffers[row['id']+'_raw']=[]
            model_info[row['id']+'_raw']=dict(model_info[row['id']],output_policy='Original fused NMS, remove final-priority integration only; trained feature gate unchanged')
    parity=[];started=time.perf_counter()
    with torch.inference_mode():
        for first in range(1,limit+1,4):
            frames=list(range(first,min(first+4,limit+1)));inputs={cue:[]for cue in needed};ratios=[]
            available={cue:[]for cue in needed}
            for frame in frames:
                rgb_path=Path(plan['data_root'])/('train'if a.split=='dev'else'test')/a.sequence/'visible'/f'{frame:06d}.jpg'
                ir_path=Path(plan['generated_ir_root'])/a.split/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
                rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),-1)if'ir'in needed else None
                if rgb is None or ('ir'in needed and (ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or ir.shape!=rgb.shape[:2])):
                    raise ValueError('Actual current RGB/newIR pair missing')
                if ir is not None and a.ir_intervention=='zero':ir=np.zeros_like(ir)
                elif ir is not None and a.ir_intervention=='spatial_roll':ir=np.roll(ir,(ir.shape[0]//4,ir.shape[1]//4),(0,1))
                value,ratio=exact_half_preproc(rgb,config['inference_input_hw'],(.485,.456,.406),(.229,.224,.225))
                inputs['rgb'].append(value);ratios.append(ratio);available['rgb'].append(True)
                for cue in needed-{'rgb'}:
                    image=ir if cue=='ir'else cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                    norm=calibration['statistics'][cue];value,_=auxiliary_input(image,config['inference_input_hw'],norm['mean'],norm['std'])
                    inputs[cue].append(value);available[cue].append(bool(image.max()!=image.min()))
            # Eval-only convolution/BN and all attention reductions preserve the
            # sample dimension. A prediction never reads a neighboring sample's
            # pixels or statistics, even when independent images share a batch.
            features={cue:model.backbone(torch.from_numpy(np.stack(values)).cuda())for cue,values in inputs.items()}
            teacher=model.head(features['rgb']);teacher_scores=[];offset=0
            for value in features['rgb']:
                count=value.shape[-2]*value.shape[-1]
                score=teacher[:,offset:offset+count,4]*teacher[:,offset:offset+count,5]
                teacher_scores.append(score.reshape(len(frames),1,*value.shape[-2:]));offset+=count
            fresh=[detections(value,ratio)for value,ratio in zip(postprocess(teacher.clone(),1,.1,.7),ratios)]
            originals=[]
            for frame,current in zip(frames,fresh):
                lo,hi=reference['offsets'][frame-1:frame+1];original=np.c_[reference['boxes'][lo:hi],reference['scores'][lo:hi]]
                originals.append(original)
                parity.append(dict(frame=frame,count_equal=len(original)==len(current),array_exact=bool(np.array_equal(current,original)),
                                   maximum_difference=float(np.max(np.abs(current-original)))if current.shape==original.shape and current.size else None))
            for row in specification['models']:
                name=row['id'];cue=row['cue'];adapter=loaded[name]
                if adapter is None:
                    values=fresh
                else:
                    valid=torch.tensor(available[cue],device='cuda')
                    repaired=adapter(features['rgb'],features[cue],teacher_scores,valid)
                    prediction=model.head(repaired)
                    values=[detections(value,ratio)for value,ratio in zip(postprocess(prediction,1,.1,.7),ratios)]
                    if row.get('zero_adapter'):
                        for current,baseline in zip(values,fresh):np.testing.assert_array_equal(current,baseline)
                    buffers[name+'_raw'].extend([value.copy()for value in values])
                    if row.get('variant','full')!='no_gate':
                        values=[original.copy()if not ok or row.get('zero_adapter')else priority_union(original,[(current,1.)])[0]
                                for original,current,ok in zip(originals,values,available[cue])]
                buffers[name].extend(values)
            if frames[-1]%100<4 or frames[-1]==limit:
                progress=dict(frame=frames[-1],frames=limit,snapshots=len(loaded),physical_gpu=physical,seconds=time.perf_counter()-started)
                dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for key,values in buffers.items():
        merged=np.concatenate(values).reshape(-1,5);path=out/(key+'.npz');temporary=out/(key+'.tmp.npz')
        np.savez_compressed(temporary,frame_ids=np.arange(1,limit+1),offsets=np.r_[0,np.cumsum([len(v)for v in values])],
                            boxes=merged[:,:4],scores=merged[:,4]);os.replace(temporary,path)
        outputs[key]=dict(path=str(path),sha256=sha256(path),frames=limit,detections=len(merged),bytes=path.stat().st_size)
    dump(out/'complete.json',dict(status='SMOKE_COMPLETE'if a.source_check else'COMPLETE_CORRELATION_OBSERVATIONS',
        sequence=a.sequence,split=a.split,frames=limit,IR_intervention=a.ir_intervention,outputs=outputs,models=model_info,config_sha256=sha256(config_path),
        specification_sha256=spec_sha256,calibration_sha256=sha256(calibration_path),source_RGB_sha256=config['source_RGB_sha256'],
        global_reference=dict(path=str(source),sha256=sha256(source)),fresh_RGB_parity=parity if a.source_check else
        dict(frames=len(parity),count_equal=sum(x['count_equal']for x in parity),exact=sum(x['array_exact']for x in parity)),
        GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,independent_sample_batch_size=4,
        source_sha256=sha256(__file__),adapter_source_sha256=sha256(ROOT/'yolox/models/local_correlation_repair.py'),
        physical_gpu=physical,seconds=time.perf_counter()-started,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,
        runtime_scope='Shared current-image production of all declared snapshots, not per-method end-to-end FPS. Eval BN fixed; attention never reduces across samples.'))


if __name__=='__main__':main()
