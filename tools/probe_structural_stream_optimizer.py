#!/usr/bin/env python3
"""Train-only CPU parity against two steps of the preserved legacy fitting loop."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump,sha256
from tools.structural_stream_training import TrainBatch,StreamingTrainInputs,initialize_head_adapter,initialize_encoder,optimizer_step
from yolox.models.yolo_head import YOLOXHead
from yolox.models.structural_nocycle_refinement import NoCycleRefinement
import tools.train_structural_relative as legacy


def assert_nested_equal(a,b):
    assert type(a)==type(b)
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_nested_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


class CachedSlices:
    def __init__(self,work,protocol,parent_sha):
        self.maps={};self.metadata={};self.index={};self.sources={};self.last=None
        for seq in protocol['train']:
            meta_path=work/'correlation_cache'/seq/'features/complete.json'
            m=json.loads(meta_path.read_text());assert m['config_sha256']==parent_sha and not m['Dev_rows']and not m['Test_rows']
            smeta_path=work/'structural_cache'/seq/'complete.json';s=json.loads(smeta_path.read_text())
            assert s['parent_cache_manifest_sha256']==sha256(meta_path)
            arrays={}
            for key,row in m['outputs'].items():
                if not key.startswith(('rgb_','ir_')):continue
                assert sha256(row['path'])==row['sha256']
                arrays[key]=np.load(row['path'],mmap_mode='r')
            assert sha256(m['targets']['path'])==m['targets']['sha256']
            arrays['targets']=np.load(m['targets']['path'],mmap_mode='r')
            for key,row in s['outputs'].items():
                assert sha256(row['path'])==row['sha256']
                arrays[key]=np.load(row['path'],mmap_mode='r')
            self.maps[seq]=arrays;self.metadata[seq]=m
            self.index[seq]={f:i for i,f in enumerate(m['frame_ids'])}
            self.sources[seq]=dict(cache_manifest_sha256=sha256(meta_path),structure_manifest_sha256=sha256(smeta_path))

    def features(self,sequence,frame,cue,value):
        index=self.index[sequence][frame]
        return tuple(torch.from_numpy(np.array(self.maps[sequence][f'{cue}_P{level}'][index],copy=True))for level in [3,4,5])

    def batch(self,draw,cue):
        key=(tuple(draw),cue)
        if self.last is not None and self.last[0]==key:return self.last[1]
        stack=lambda name:torch.from_numpy(np.stack([self.maps[s][name][i]for s,i in draw]))
        rgb=tuple(stack('rgb_P'+str(level)).float()for level in [3,4,5])
        aux=rgb if cue=='rgb'else tuple(stack(cue+'_P'+str(level)).float()for level in [3,4,5])
        prefix='ir'if cue=='ir'else'rgb'
        structure=tuple(stack(name)for name in ['rgb_descriptor',prefix+'_descriptor','rgb_valid',prefix+'_valid'])
        result=TrainBatch(rgb,aux,stack('targets'),structure,[(s,self.metadata[s]['frame_ids'][i])for s,i in draw])
        self.last=(key,result);return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);cv2.setNumThreads(1)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_structural_relative_20261123.json';config=json.loads(config_path.read_text())
    parent_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';parent=json.loads(parent_path.read_text())
    out=work/'structural_stream_optimizer_CPU';out.mkdir(exist_ok=False)
    source=Path(parent['source_RGB_checkpoint']);assert sha256(source)==parent['source_RGB_sha256']
    state=dict(status='REGISTERED_TWO_STEP_CPU_OPTIMIZER_PARITY',source_sha256=sha256(__file__),
        live_helper_sha256=sha256(ROOT/'tools/structural_stream_training.py'),legacy_source_sha256=sha256(ROOT/'tools/train_structural_relative.py'),
        fixed_seed=0,engineering_steps_per_path=2,no_formal_training=True,Test_read=False)
    dump(out/'state.json',state);start=time.perf_counter()
    try:
        cache=CachedSlices(work,protocol,sha256(parent_path))
        with np.load(work/'structural_stream_schedule/frame_addresses.npz')as z:schedule={k:z[k]for k in z.files}
        gpu_proof_path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_INPUT_PARITY_GPU5.json'
        gpu_proof=json.loads(gpu_proof_path.read_text());assert gpu_proof['status']=='COMPLETE_TRAIN_INPUT_PARITY_PROBE'
        for seq in protocol['train']:
            proof=next(r for r in gpu_proof['rows']if r['sequence']==seq)
            assert proof['cache_manifest_sha256']==cache.sources[seq]['cache_manifest_sha256']
        draws=[[('BT-001',0),('BT-001',96),('SN-015',0),('SN-015',96)],
               [('BT-003',0),('BT-003',96),('MSK-002',0),('MSK-002',96)]]
        assert all((seq,cache.metadata[seq]['frame_ids'][index])in
                   {(r['sequence'],r['frame'])for r in gpu_proof['rows']}for draw in draws for seq,index in draw)
        saved=torch.load(source,map_location='cpu',weights_only=False);a=config['architecture']
        torch.manual_seed(0)
        reference_head=YOLOXHead(1,width=1.25)
        reference_head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['model'].items()if k.startswith('head.')},strict=True)
        reference_head=reference_head.float().eval().requires_grad_(False)
        reference=NoCycleRefinement(latent=a['latent_channels'],radius=a['local_radius'],
            temperature=a['structural_temperature'],bound=a['feature_residual_bound_RMS_fraction'],
            risk_threshold=a['teacher_risk_threshold'],refinement='no_cycle',tolerance=a['forward_backward_tolerance_cells'])
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(0)
        head,adapter=initialize_head_adapter(saved,a,'111','cpu')
        encoder=initialize_encoder(saved,ROOT/'exps/yolox_mft26ir_visible.py','cpu')
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert_nested_equal(dict(head.state_dict()),dict(reference_head.state_dict()))
        assert_nested_equal(dict(adapter.state_dict()),dict(reference.state_dict()))
        del encoder,saved
        calibration=json.loads((work/'correlation_cache/calibration.json').read_text())
        provider=StreamingTrainInputs(plan,protocol,config,calibration,schedule,None,'cpu',feature_backend=cache.features)
        batches=[]
        for draw in draws:
            actual=provider.prepare(draw,'ir',1);expected=cache.batch(draw,'ir')
            for x,y in zip(actual.rgb+actual.auxiliary+actual.structure+(actual.target,),
                           expected.rgb+expected.auxiliary+expected.structure+(expected.target,)):assert torch.equal(x,y)
            batches.append(actual)
        print('Actual8cachedTrainframes,targets/descriptors/features and initialization exact; CPUgradient steps begin',flush=True)
        torch.set_num_threads(4)
        fit=config['fit'];ref_optimizer=torch.optim.AdamW(reference.parameters(),lr=fit['learning_rate'],weight_decay=fit['weight_decay'])
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=fit['learning_rate'],weight_decay=fit['weight_decay'])
        # Execute the preserved original run_fit rather than a second copy of
        # the new loss. Only its explicit permutation-device literal changes;
        # input callbacks supply the real cache slices on CPU. No global module
        # binding or source file used by live GPU jobs is modified.
        original=inspect.getsource(legacy.run_fit)
        assert original.count("device='cuda'")==1
        adapted=original.replace("device='cuda'","device='cpu'")
        namespace=dict(legacy.__dict__)
        namespace['batches']=lambda sequences,banks,seed,epoch:iter(draws)
        namespace['tensors']=lambda banks,draw,cue:(cache.batch(draw,cue).rgb,cache.batch(draw,cue).auxiliary,cache.batch(draw,cue).target)
        namespace['structural_tensors']=lambda banks,draw,cue:cache.batch(draw,cue).structure
        exec(compile(adapted,'<preserved_run_fit_CPU_device_only>','exec'),namespace)
        ref_out=out/'legacy_reference';ref_out.mkdir()
        meta=dict(physical_gpu='CPU_reference',config_sha256=sha256(config_path),source_hashes={
            'tools/train_structural_relative.py':sha256(ROOT/'tools/train_structural_relative.py')})
        namespace['run_fit'](SimpleNamespace(cue='ir',variant='no_cycle',seed=0,smoke_steps=2),
            reference,reference_head,ref_optimizer,{s:{'targets':cache.maps[s]['targets']}for s in protocol['train']},
            {},protocol['train'],config,config_path,ref_out,meta)
        records=[optimizer_step(adapter,head,batch,config,optimizer,i)for i,batch in enumerate(batches)]
        assert records[0]['initial_raw_RGB_head_exact']
        ref_data=torch.load(ref_out/'latest.pt',map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(np.mean([v['values']for v in records],axis=0),ref_data['history'][0]['loss'])
        assert_nested_equal(dict(adapter.state_dict()),dict(ref_data['adapter']))
        assert_nested_equal(optimizer.state_dict(),ref_data['optimizer'])
        assert not torch.cuda.is_initialized()
        report=dict(state,status='COMPLETE_TWO_STEP_CPU_OPTIMIZER_PARITY',seconds=time.perf_counter()-start,
            actual_frames=[batch.frames for batch in batches],input_fields_exact=True,
            original9adapter_tensors_initial_and_updated_exact=True,frozen_head_tensors_exact=True,
            frozen_backbone_construction_preserved_CPU_RNG=True,initialization_threads=2,CPU_step_threads=4,
            loss_components_and_clipped_gradient_norm_exact=True,AdamW_moments_steps_groups_exact=True,
            mean_loss_and_gradient=ref_data['history'][0]['loss'],frame_draw_sha256=provider.draw_hash.hexdigest(),
            legacy_function_sha256=hashlib.sha256(original.encode()).hexdigest(),CPU_function_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
            prior_actual_GPU12frame_B1_parity_sha256=sha256(gpu_proof_path),cache_sources=cache.sources,
            CUDA_context_initialized=False,Dev_read=False,no_image_or_FPN_copy=True,
            scope='TwoCPU diagnostic steps/path on8realcachedTrainframes/fourTrainvideos. The newinput path '
                  'reads actualRGB/IR and GT/descriptors, but encoder callback replays B1cache slices already '
                  'verified in the earlier12frame actualGPUprobe. This is not new liveGPU encoding or '
                  'GPUgradient parity, and does not train a formal24pass candidate or measure HOTA.',
            pending='Full streaming24pass loop and GPU optimizer parity; matched coverage-study registration and allformalTest.',
            goal_met=False,supported_innovations=0)
        dump(out/'state.json',report)
        dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_OPTIMIZER_CPU_PARITY.json',report)
        print(json.dumps(dict(status=report['status'],CPU_only=True,diagnostic_steps_per_path=2,no_formal_fit=True)),flush=True)
    except Exception as error:
        state.update(status='FAILED',error=repr(error));dump(out/'state.json',state);raise


if __name__=='__main__':main()
