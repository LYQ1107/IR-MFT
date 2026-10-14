#!/usr/bin/env python3
"""Same cu117/device-pad V3 worker with an explicit owned memory cap."""
import argparse,socket
import cv2,torch,numpy as np
from yolox.utils import postprocess
from cfp.detector_runtime import receive_arrays,send_arrays
from cfp.detector_runtime_device_pad import DevicePadDetectorFrontend


def main():
    ap=argparse.ArgumentParser()
    for name in ['fd','checkpoint','scripts-root']:ap.add_argument('--'+name,required=True,type=int if name=='fd' else str)
    a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    if torch.cuda.device_count()!=1:raise ValueError('One explicitlyselected GPU required')
    torch.cuda.set_per_process_memory_fraction(.6);torch.manual_seed(0)
    frontend=DevicePadDetectorFrontend(a.checkpoint,a.scripts_root,detector_batch=8,input_regime='causal_precomputed',preprocessing='lookup_uint8');channel=socket.socket(fileno=a.fd)
    send_arrays(channel,{},dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),parameters=frontend.detector_parameters,device=torch.cuda.get_device_name(),current_frame_repeat=8,owned_memory_fraction=.6));frames=0
    try:
        while True:
            arrays,m=receive_arrays(channel)
            if m['action']=='stop':send_arrays(channel,{},dict(frames_processed=frames,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()));break
            if m['action']=='cache_tail_validation':
                size,position=int(m['size']),int(m['position'])
                if not 1<=size<8 or not 0<=position<size:raise ValueError('Onlydeclared partialtail validation required')
                image,_=frontend.transform(arrays['rgb'],arrays['ir'],None,frontend.exp.test_size)
                one=torch.from_numpy(image)[None].cuda().half();batch=torch.zeros((size,*one.shape[1:]),device='cuda',dtype=torch.float16);batch[position]=one[0]
                with torch.inference_mode():value=postprocess(frontend.detector(batch),1,.1,.7)[position]
                raw=np.empty((0,7),np.float32) if value is None else value.float().cpu().numpy();scale=min(frontend.exp.test_size[0]/arrays['rgb'].shape[0],frontend.exp.test_size[1]/arrays['rgb'].shape[1])
                send_arrays(channel,dict(boxes=(raw[:,:4]/scale).astype(np.float32),scores=(raw[:,4]*raw[:,5]).astype(np.float32)),dict(validation_only=True,current_only_zeros_other_positions=True))
                continue
            if m['action']!='frame':raise ValueError('Onlycurrent-frame request permitted')
            result=frontend.process(arrays['rgb'],provided_ir=arrays['ir']);frames+=1
            send_arrays(channel,{k:result[k] for k in ['boxes','scores','raw_post_nms']},dict(stage_seconds=result['stage_seconds']))
    except EOFError:pass
    finally:channel.close()

if __name__=='__main__':main()
