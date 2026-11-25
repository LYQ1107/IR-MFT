"""Read-only inference diagnostics for task logits, gates and feature changes."""
import torch

from tools.native_pyramid_gate_observer import Moments


class TaskUtilityObserver:
    def __init__(self):
        self.rows={};self.handles=[];self.module_ids=set()

    def attach(self,name,module):
        if name in self.rows or id(module) in self.module_ids:
            raise ValueError('Each diagnostic stream needs a distinct frozen module instance')
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Only frozen inference modules may be observed')
        self.module_ids.add(id(module))
        fields=['gate','frame_quality','foreground_logit','utility_logit','feature_delta_RMS','feature_changed']
        row=dict(frames=0,available_frames=0,missing_frames=0,
            levels=[{k:Moments() for k in fields} for _ in range(3)])
        self.rows[name]=row
        def hook(net,args,kwargs,output):
            with torch.no_grad():
                repaired,_,information=output;row['frames']+=1
                if information['missing_exact_RGB']:
                    row['missing_frames']+=1;return
                row['available_frames']+=1;rgb=args[0] if args else kwargs['rgb']
                for result,observed,base,value in zip(row['levels'],information['levels'],rgb,repaired):
                    mask=observed['valid']
                    result['gate'].add(observed['gate'][mask]);result['frame_quality'].add(observed['frame_quality'])
                    result['foreground_logit'].add(observed['foreground_logits'][mask])
                    result['utility_logit'].add(observed['utility_logits'][mask])
                    delta=value.detach().float()-base.detach().float()
                    result['feature_delta_RMS'].add(delta.square().mean(1,keepdim=True)[mask].sqrt())
                    result['feature_changed'].add(delta.ne(0).any(1,keepdim=True)[mask].float())
        self.handles.append(module.register_forward_hook(hook,with_kwargs=True))

    def summary(self,name):
        row=self.rows[name]
        return dict(frames=row['frames'],available_frames=row['available_frames'],missing_frames=row['missing_frames'],
            levels=[{k:v.summary() for k,v in level.items()} for level in row['levels']],
            scope='Valid current spatial cells,frame quality once perframe. Missing observations have no invented logits. No tensors retained; not calibrated reliability,causal efficacy or individualFPS.')

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.module_ids.clear()
