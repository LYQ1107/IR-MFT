"""Source-matched individual B1/B4 prefix costs; no authorization or accuracy."""
import hashlib
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import torch

from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.run_pseudo_ir_comparison import sha256
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.utils import postprocess
from tools.extract_structural_relative import detections


class RuntimeImages:
    """Only a cost prefix; never emit a formal complete-sequence receipt."""
    def __init__(self,plan,split,sequence,frames=32):
        if split not in LENGTHS or sequence not in LENGTHS[split] or type(frames) is not int or not 1<=frames<=min(32,LENGTHS[split][sequence]):
            raise ValueError('A bounded original-video cost prefix is required')
        self.plan,self.split,self.sequence,self.frames=plan,split,sequence,frames
        self.last_frame=self.hashed_RGB_frames=self.hashed_IR_frames=self.decoded_IR_frames=0
    def read_frame(self,frame,needs_ir):
        if frame!=self.last_frame+1 or frame>self.frames:raise ValueError('Read each cost-prefix frame once in order')
        rgbpath=Path(self.plan['data_root'])/('test' if self.split=='test' else 'train')/self.sequence/'visible'/f'{frame:06d}.jpg'
        raw=rgbpath.read_bytes();rgb=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if rgb is None or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Original current RGB decode failed')
        ir=None;ir_sha=None
        if needs_ir:
            irpath=Path(self.plan['generated_ir_root'])/self.split/self.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            encoded=irpath.read_bytes();ir=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_UNCHANGED)
            if ir is None or ir.dtype!=np.uint8 or ir.ndim!=2 or ir.shape!=rgb.shape[:2]:raise ValueError('Original current IR decode/alignment failed')
            ir_sha=hashlib.sha256(encoded).hexdigest();self.hashed_IR_frames+=1;self.decoded_IR_frames+=1
        self.last_frame=frame;self.hashed_RGB_frames+=1
        return rgb,ir,[self.sequence,frame,hashlib.sha256(raw).hexdigest(),ir_sha]


class CanonicalRGBBatch:
    """Real B4 RGB NN, not packet replay or a B1 alias; independent eval samples."""
    batch_size=4
    needs_actual_IR_image=False
    def __init__(self,model,case,input_hw):
        from tools.task_utility_design import CANONICAL
        if case!=CANONICAL or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Original canonical RGB and a frozen source NN required')
        self.model,self.rows,self.input_hw=model,[dict(case)],tuple(input_hw)
        self.sequence,self.last_frame=None,0
        self.device,self.dtype=next(model.parameters()).device,next(model.parameters()).dtype
    def reset(self,sequence):self.sequence,self.last_frame=sequence,0
    @torch.no_grad()
    def step_batch(self,images,sequence,first_frame):
        if len(images)!=4 or not sequence or type(first_frame) is not int:raise ValueError('Four actual ordered RGB frames required')
        if self.sequence!=sequence:
            if first_frame!=1:raise ValueError('A new video starts at1')
            self.reset(sequence)
        if first_frame!=self.last_frame+1:raise ValueError('No skipped or replayed cost frames')
        values=[exact_half_preproc(rgb,self.input_hw,(.485,.456,.406),(.229,.224,.225)) for rgb in images]
        pixels=torch.from_numpy(np.stack([r[0] for r in values])).to(device=self.device,dtype=self.dtype)
        decoded=self.model(pixels)
        if decoded.ndim!=3 or decoded.shape[0]!=4 or decoded.shape[-1]!=6 or not bool(torch.isfinite(decoded).all()):
            raise ValueError('Complete finite single-class B4 YOLOX output required')
        selected=postprocess(decoded,1,.1,.7)
        if len(selected)!=4:raise ValueError('Every real source observation needs its own packet')
        result=[{self.rows[0]['id']:detections(output,ratio)} for output,(_,ratio) in zip(selected,values)]
        self.last_frame=first_frame+3
        return result,dict(GT_read=False,previous_outputs_rewritten=False,temporal_or_cross_sample_feature_state=False,
            independent_NN_sample_batch=4,contains_four_actual_RGB_frames=True,online_B1=False)


def packet_digest(digest,frame,values):
    digest.update(np.asarray([frame,len(values)],dtype='<i8').tobytes())
    digest.update(np.asarray(values,dtype='<f4').tobytes())


def literal_prefix(path,frames=32):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            if int(line.split(b',',1)[0])>frames:break
            result.update(line)
    return result.hexdigest()


def validate_reference(reference,case,sequence,frames):
    if (reference['sequence']!=sequence or reference['prefix_frames']!=frames or
            reference['id']!=case['id'] or len(reference['source_rows'])!=frames):
        raise ValueError('The exact original case/video prefix reference is required')
    for frame,row in enumerate(reference['source_rows'],1):
        if len(row)!=4 or row[:2]!=[sequence,frame] or any(not isinstance(s,str) or len(s)!=64 for s in row[2:]):
            raise ValueError('Original RGB/IR byte references required')
    for field in ['packet_sha256','literal_MOT_sha256']:
        if not isinstance(reference[field],str) or len(reference[field])!=64:raise ValueError('Exact original output digests required')


def _pass(engine_factory,images_factory,case,sequence,reference,frames,warmup,timed,synchronize,clock,resources):
    engine=engine_factory();images=None
    try:
        batch=getattr(engine,'batch_size',1)
        if engine.rows!=[case] or engine.sequence is not None or engine.last_frame!=0 or batch not in [1,4]:
            raise ValueError('Exactly one fresh original execution route is required')
        if frames%batch or warmup%batch:raise ValueError('Warmup and measured ranges must contain complete real batches')
        images=images_factory()
        if images.sequence!=sequence or images.frames<frames or images.last_frame!=0:raise ValueError('Fresh sequential cost images required')
        tracker=IndependentByteTrack();packets=hashlib.sha256();tracks=hashlib.sha256();samples=[];detection_count=0
        for first in range(1,frames+1,batch):
            measured=timed and first>warmup
            if measured:synchronize();t0=clock()
            inputs=[images.read_frame(f,engine.needs_actual_IR_image) for f in range(first,first+batch)]
            if measured:t1=clock()
            if batch==1:
                rgb,ir,_=inputs[0];packet,info=engine.step(rgb,ir,sequence,first);outputs=[packet]
                if info.get('future_frame_input'):raise ValueError('A B1 route cannot consume future frames')
            else:outputs,info=engine.step_batch([r[0] for r in inputs],sequence,first)
            if info['GT_read'] or info['previous_outputs_rewritten'] or len(outputs)!=batch:
                raise ValueError('No GT, output rewrite or missing real observations allowed')
            if measured:synchronize();t2=clock()
            emitted=[]
            for frame,values in zip(range(first,first+batch),outputs):
                if set(values)!={case['id']}:raise ValueError('A bundled stream is not an individual runtime')
                emitted.append(tracker.update(values[case['id']],frame))
            if measured:
                t3=clock();elapsed=t3-t0
                components=dict(IO_decode_byte_hash_seconds=t1-t0,preprocessing_NN_NMS_host_copy_seconds=t2-t1,
                    ByteTrack_seconds=t3-t2)
                if not math.isfinite(elapsed) or elapsed<=0 or any(not math.isfinite(v) or v<0 for v in components.values()):
                    raise ValueError('Measured wall times must be finite and positive')
                samples.append(dict(frames=batch,seconds=elapsed,**components))
            for offset,(rgb,ir,row) in enumerate(inputs):
                frame=first+offset;expected=reference['source_rows'][frame-1]
                if row[:3]!=expected[:3] or (engine.needs_actual_IR_image and row[3]!=expected[3]):
                    raise ValueError('Original required-modality bytes changed during runtime')
                values=outputs[offset][case['id']];packet_digest(packets,frame,values);detection_count+=len(values)
                tracks.update(''.join(mot_line(r) for r in emitted[offset]).encode())
            if first%8==1 or first+batch-1==frames:resources()
        if packets.hexdigest()!=reference['packet_sha256'] or tracks.hexdigest()!=reference['literal_MOT_sha256']:
            raise ValueError('Individual actual NN/ByteTrack prefix differs from original complete observations')
        return dict(sequence=sequence,prefix_frames=frames,NN_batch_size=batch,measured_frames=sum(s['frames'] for s in samples),
            samples=samples,packet_sha256=packets.hexdigest(),literal_MOT_sha256=tracks.hexdigest(),
            warmup_to_timing_tracker_and_detector_state_preserved=bool(timed),warmup_frames=warmup if timed else 0,
            detections=detection_count,RGB_byte_frames_hashed=images.hashed_RGB_frames,IR_byte_frames_hashed=images.hashed_IR_frames,
            IR_frames_decoded=images.decoded_IR_frames,simultaneous_methods=1,original_required_modality_bytes_exact=True)
    finally:
        close=getattr(engine,'close',None)
        if close is not None:close()


def measure_prefix(engine_factory,images_factory,case,sequence,reference,*,frames=32,warmup=8,
                   synchronize=lambda:None,clock=time.perf_counter,resources=lambda:None):
    """Verify first, then one timed pass. Reduced ranges are engineering only."""
    if type(frames) is not int or type(warmup) is not int or not 0<warmup<frames<=32:
        raise ValueError('Bounded positive prefix and warmup required')
    validate_reference(reference,case,sequence,frames);resources()
    verified=_pass(engine_factory,images_factory,case,sequence,reference,frames,warmup,False,synchronize,clock,resources)
    timed=_pass(engine_factory,images_factory,case,sequence,reference,frames,warmup,True,synchronize,clock,resources)
    if any(verified[k]!=timed[k] for k in ['packet_sha256','literal_MOT_sha256','NN_batch_size']):
        raise ValueError('Independent verified and timed executions differ')
    return dict(verification=verified,timing=timed,actual_source_and_output_equivalence_before_and_during_timing=True)


def aggregate_runtime(sequences):
    if not sequences or len({r['timing']['sequence'] for r in sequences})!=len(sequences):raise ValueError('Unique actual cost videos required')
    samples=[s for r in sequences for s in r['timing']['samples']];batches={r['timing']['NN_batch_size'] for r in sequences}
    if len(batches)!=1 or not samples or any(s['frames'] not in batches or not math.isfinite(s['seconds']) or s['seconds']<=0 for s in samples):
        raise ValueError('Positive actual measurements for one original batch policy required')
    batch=batches.pop();frames=sum(s['frames'] for s in samples);seconds=sum(s['seconds'] for s in samples)
    result=dict(NN_batch_size=batch,measured_frames=frames,measured_batches=len(samples),seconds=seconds,FPS=frames/seconds,
        amortized_wall_ms_per_frame=1000*seconds/frames,
        component_seconds={k:sum(s[k] for s in samples) for k in ['IO_decode_byte_hash_seconds','preprocessing_NN_NMS_host_copy_seconds','ByteTrack_seconds']},
        latency_scope='Individual B1 frame completion.' if batch==1 else 'B4 batch completion;amortized per-frame cost is not single-frame latency.')
    result['mean_frame_latency_ms' if batch==1 else 'mean_batch_latency_ms']=1000*statistics.mean(s['seconds'] for s in samples)
    result['median_frame_latency_ms' if batch==1 else 'median_batch_latency_ms']=1000*statistics.median(s['seconds'] for s in samples)
    return result
