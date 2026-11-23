#!/usr/bin/env python3
"""Same frozen detector/adapters, causal correction policies, no GT input."""
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
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.build_correlation_cache import auxiliary_input
from tools.extract_correlation_policies import detections
from yolox.exp import get_exp
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.models.temporal_correction import TemporalCorrection
from yolox.tracker.physical_motion import padded_motion_view,backward_rgb_motion
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def read_predictions(path):
    with np.load(path)as z:return {k:z[k]for k in ['frame_ids','offsets','boxes','scores']}


def frame_detections(data,frame):
    lo,hi=data['offsets'][frame-1:frame+1]
    return np.c_[data['boxes'][lo:hi],data['scores'][lo:hi]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True)
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--source-check',type=int,default=0)
    p.add_argument('--ir-intervention',choices=['none','zero','spatial_roll'],default='none');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical2/3')
    main_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(main_path.read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';config=json.loads(config_path.read_text())
    spec_path=ROOT/'configs/perceptual_temporal_correction_20261123.json';spec=json.loads(spec_path.read_text())
    if spec['config_sha256']!=sha256(config_path):raise ValueError('Frozen source model changed')
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    if a.sequence not in protocol['dev'if a.split=='dev'else'historical_test']:raise ValueError('Frozen split only')
    if a.split=='test'and not(work/'temporal_selection_frozen.json').exists():raise ValueError('Freeze all Dev/seeds/removals before Test')
    if a.source_check and(a.split!='dev'or a.ir_intervention!='none'):raise ValueError('Real Dev no-memory parity check only')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.45,0)
    checkpoint=Path(config['source_RGB_checkpoint'])
    if sha256(checkpoint)!=config['source_RGB_sha256']:raise ValueError('Source RGB checkpoint changed')
    model=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    model=model.cuda().half().eval().requires_grad_(False)
    calibration_path=work/'correlation_cache/calibration.json';calibration=json.loads(calibration_path.read_text())
    if calibration['config_sha256']!=sha256(config_path):raise ValueError('Train normalization changed')
    loaded={};gates={};memories={};references={};handles=[];model_info={}
    for row in spec['models']:
        name=row['id'];path=Path(row['checkpoint'])
        if sha256(path)!=row['checkpoint_sha256']:raise ValueError('Selected adapter changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if(saved['config_sha256']!=row['training_config_sha256']or saved['metadata']['cue']!=row['cue']
                or saved['completed_epochs']!=row['epoch']or saved['metadata']['variant']!='full'or saved['metadata']['seed']!=row['seed']):
            raise ValueError('Frozen adapter provenance mismatch')
        architecture=config['architecture'];adapter=CorrelationRepair(latent=architecture['latent_channels'],radius=architecture['local_radius'],
            temperature=architecture['temperature'],risk_gate=True,bound=architecture['maximum_residual_RMS_fraction'])
        adapter.load_state_dict(saved['adapter'],strict=True);loaded[name]=adapter.cuda().float().eval().requires_grad_(False)
        model_info[name]=dict(row,metadata=saved['metadata']);del saved
        gates[name]=[None]*3
        for level,layer in enumerate(loaded[name].levels):
            def capture(module,inputs,output,key=name,index=level):gates[key][index]=output
            handles.append(layer.gate.register_forward_hook(capture))
        for policy in spec['policies']:
            memories[name,policy]=TemporalCorrection(mode=policy,strength=spec['strength'],bound=spec['bound'],epsilon=spec['epsilon'])
        if a.split=='dev'and a.ir_intervention=='none':
            directory=work/row['reference_producer']/a.split/a.sequence;producer=json.loads((directory/'complete.json').read_text())
            for policy,key in [('raw',row['reference_model_id']+'_raw'),('priority',row['reference_model_id'])]:
                info=producer['outputs'][key]
                if sha256(info['path'])!=info['sha256']:raise ValueError('No-memory reference changed')
                references[name,policy]=read_predictions(info['path'])
    source=Path(plan['frozen_detection_sources']['rgb_'+a.split])/'detections'/(a.sequence+'.npz')
    baseline=read_predictions(source);limit=min(len(baseline['frame_ids']),a.source_check or len(baseline['frame_ids']))
    directory='temporal_correction_predictions'+(''if a.ir_intervention=='none'else'_'+a.ir_intervention)
    out=work/directory/a.split/a.sequence
    if a.source_check:out=out/f'source_check_{a.source_check}'
    out.mkdir(parents=True,exist_ok=True)
    if(out/'complete.json').exists():raise ValueError('Completed temporal predictions exist')
    buffers={f'{row["id"]}_{policy}'+suffix:[]for row in spec['models']for policy in spec['policies']for suffix in ['', '_raw']}
    exact={f'{name}_{policy}':0 for name,policy in references};fresh_exact=0;fresh_counts=0;last_view=None;started=time.perf_counter();flow_seconds=0.
    with torch.inference_mode():
        for first in range(1,limit+1,4):
            frames=list(range(first,min(first+4,limit+1)));inputs={cue:[]for cue in ['rgb','gray','ir']};available={cue:[]for cue in inputs}
            ratios=[];motions=[]
            for frame in frames:
                rgb=cv2.imread(str(Path(plan['data_root'])/('train'if a.split=='dev'else'test')/a.sequence/'visible'/f'{frame:06d}.jpg'))
                ir=cv2.imread(str(Path(plan['generated_ir_root'])/a.split/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),-1)
                if rgb is None or ir is None or ir.dtype!=np.uint8 or ir.ndim!=2 or ir.shape!=rgb.shape[:2]:raise ValueError('Actual RGB/newIR pair required')
                if a.ir_intervention=='zero':ir=np.zeros_like(ir)
                elif a.ir_intervention=='spatial_roll':ir=np.roll(ir,(ir.shape[0]//4,ir.shape[1]//4),(0,1))
                value,ratio=exact_half_preproc(rgb,config['inference_input_hw'],(.485,.456,.406),(.229,.224,.225))
                inputs['rgb'].append(value);ratios.append(ratio);available['rgb'].append(True)
                for cue,image in [('gray',cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)),('ir',ir)]:
                    norm=calibration['statistics'][cue];value,_=auxiliary_input(image,config['inference_input_hw'],norm['mean'],norm['std'])
                    inputs[cue].append(value);available[cue].append(bool(image.max()!=image.min()))
                clock=time.perf_counter();view=padded_motion_view(rgb,config['inference_input_hw'])
                if last_view is None:motions.append((None,None))
                else:
                    backward,valid=backward_rgb_motion(last_view,view,spec['motion']['FB_tolerance'])
                    motions.append((torch.from_numpy(backward.transpose(2,0,1)[None].copy()).cuda(),torch.from_numpy(valid[None,None].copy()).cuda()))
                last_view=view;flow_seconds+=time.perf_counter()-clock
            features={cue:model.backbone(torch.from_numpy(np.stack(values)).cuda())for cue,values in inputs.items()}
            teacher=model.head(features['rgb']);scores=[];offset=0
            for value in features['rgb']:
                count=value.shape[-2]*value.shape[-1];score=teacher[:,offset:offset+count,4]*teacher[:,offset:offset+count,5]
                scores.append(score.reshape(len(frames),1,*value.shape[-2:]));offset+=count
            fresh=[detections(value,ratio)for value,ratio in zip(postprocess(teacher.clone(),1,.1,.7),ratios)]
            originals=[frame_detections(baseline,frame)for frame in frames]
            fresh_exact+=sum(np.array_equal(x,y)for x,y in zip(fresh,originals));fresh_counts+=sum(len(x)==len(y)for x,y in zip(fresh,originals))
            for row in spec['models']:
                name=row['id'];cue=row['cue'];gates[name]=[None]*3;ok=torch.tensor(available[cue],device='cuda')
                repaired=loaded[name](features['rgb'],features[cue],scores,ok)
                if bool(ok.any())and any(g is None for g in gates[name]):raise RuntimeError('All current trust hooks required')
                trust=[torch.zeros_like(p)if g is None else g.sigmoid()*(1-p.detach().float().clamp(0,1))*(p.detach().float()<.6)
                       for g,p in zip(gates[name],scores)]
                for policy in spec['policies']:
                    per_frame=[]
                    for index in range(len(frames)):
                        backward,valid=motions[index]
                        per_frame.append(memories[name,policy].update([x[index:index+1]for x in features['rgb']],
                            [x[index:index+1]for x in repaired],[x[index:index+1]for x in trust],backward,valid,available[cue][index]))
                    corrected=tuple(torch.cat([f[level]for f in per_frame],0)for level in range(3))
                    prediction=model.head(corrected);raw=[detections(value,ratio)for value,ratio in zip(postprocess(prediction,1,.1,.7),ratios)]
                    priority=[priority_union(original,[(current,1.)])[0]if present else original.copy()
                              for original,current,present in zip(originals,raw,available[cue])]
                    key=name+'_'+policy;buffers[key+'_raw'].extend(raw);buffers[key].extend(priority)
                    if policy=='none'and references:
                        for output_policy,values in [('raw',raw),('priority',priority)]:
                            for frame,value in zip(frames,values):
                                reference=frame_detections(references[name,output_policy],frame);equal=np.array_equal(value,reference)
                                if a.source_check:np.testing.assert_array_equal(value,reference)
                                exact[name+'_'+output_policy]+=int(equal)
            if frames[-1]%100<4 or frames[-1]==limit:
                progress=dict(frame=frames[-1],frames=limit,physical_gpu=physical,models=3,temporal_policies=4,seconds=time.perf_counter()-started)
                dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for key,values in buffers.items():
        merged=np.concatenate(values).reshape(-1,5);path=out/(key+'.npz');temporary=out/(key+'.tmp.npz')
        np.savez_compressed(temporary,frame_ids=np.arange(1,limit+1),offsets=np.r_[0,np.cumsum([len(x)for x in values])],boxes=merged[:,:4],scores=merged[:,4])
        os.replace(temporary,path);outputs[key]=dict(path=str(path),sha256=sha256(path),frames=limit,detections=len(merged),bytes=path.stat().st_size)
    dump(out/'complete.json',dict(status='SMOKE_COMPLETE'if a.source_check else'COMPLETE_CORRELATION_OBSERVATIONS',
        sequence=a.sequence,split=a.split,frames=limit,outputs=outputs,models=model_info,config_sha256=sha256(config_path),specification_sha256=sha256(spec_path),
        no_memory_reference_exact=exact,no_memory_reference_all_exact=all(n==limit for n in exact.values())if exact else None,
        fresh_RGB_parity=dict(frames=limit,count_equal=fresh_counts,exact=fresh_exact),calibration_sha256=sha256(calibration_path),
        source_RGB_sha256=config['source_RGB_sha256'],source_sha256=sha256(__file__),
        temporal_source_sha256=sha256(ROOT/'yolox/models/temporal_correction.py'),motion_source_sha256=sha256(ROOT/'yolox/tracker/physical_motion.py'),
        physical_gpu=physical,seconds=time.perf_counter()-started,flow_seconds=flow_seconds,GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,
        history_frames=1,independent_sample_batch_size=4,IR_intervention=a.ir_intervention,
        runtime_scope='Shared all3 models/all4 temporal policies, plus reference RGB, not per-method end-to-endFPS. Independent encoder/teacher batches; correction recurrence reads only previous/current frame.'))


if __name__=='__main__':main()
