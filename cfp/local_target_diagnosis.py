"""Offline-only stopped-target discriminability and mean-target controls."""
import torch
from torch import nn
from torch.nn import functional as F
from .local_dense_frozen_controls import FrozenSceneControl


class TeacherLocalFIFO(nn.Module):
    def __init__(self,teacher):
        super().__init__();self.teacher=teacher;self.control=FrozenSceneControl('local_flexible')
    def forward(self,history,history_fraction,history_global,times,history_valid,frame,candidates,candidate_fraction,candidate_global):
        n,length,k,d=history.shape
        fraction=history_fraction*history_valid[:,:,None]
        past=self.teacher(history.reshape(-1,k,d),fraction.reshape(-1,k))
        current=self.teacher(candidates,candidate_fraction)
        return self.control(history=past['tokens'].reshape(n,length,k,d),history_fraction=fraction,
            history_global=history_global,times=times,history_valid=history_valid,frame=frame,
            candidates=current['tokens'],candidate_fraction=candidate_fraction,candidate_global=candidate_global)


class FitMeanPrediction(nn.Module):
    def __init__(self,student,mean):
        super().__init__();self.student=student;self.register_buffer('mean',F.normalize(mean,dim=-1))
    def forward(self,**inputs):
        past={k:inputs[k] for k in ['history','history_fraction','history_global','times','history_valid','frame']}
        context=self.student.history_context(past['history'],past['history_fraction'],past['history_global'],past['times'],past['history_valid'],past['frame'])
        representation=self.mean[None].expand(len(past['history']),-1,-1)*context['available'][:,None,None]
        current=self.student.current(inputs['candidates'],inputs['candidate_fraction'],inputs['candidate_global'])
        age=torch.where(context['available'],context['age'][:,-1],torch.zeros_like(context['age'][:,-1]))
        scores=self.student.relation_scores(representation,current,context['available'],age)
        return dict(scores=scores,predicted=representation,current=current,history_available=context['available'])
