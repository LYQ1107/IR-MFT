"""Identity prediction on a candidate graph shared by bounded-memory controls."""
import numpy as np
import torch

from .association import AppearanceTracker
from .predictive import bounded_identity_prototype
from yolox.tracker import matching


class PredictiveAssociationTracker(AppearanceTracker):
    def __init__(self,args,predictor=None,memory='predictor',**kwargs):
        if memory not in ['predictor','last','ema','fifo','all_history_ema','constant_zero','constant_half']:
            raise ValueError('Unknown bounded-memory association control')
        if memory=='predictor' and predictor is None:raise ValueError('Trained temporal model required')
        super().__init__(args,mode='rgb',**kwargs)
        self.predictor=predictor;self.memory_mode=memory
        self.prediction_counts=dict(predicted_rows=0,missing_history_rows=0,modified_edges=0)

    def association_cost(self,tracks,detections):
        raw=matching.iou_distance(tracks,detections)
        geometry=matching.fuse_score(raw,detections) if not self.args.mot20 else raw
        if geometry.size==0 or self.appearance_weight==0:return geometry
        selected=[];histories=[]
        for i,track in enumerate(tracks):
            history=list(track.identity_history)
            if not history:
                self.prediction_counts['missing_history_rows']+=1;continue
            if len(history)>8 or any(frame>=self.frame_id for frame,_,_ in history):raise ValueError('Predict before current identity writes from at most8 past observations')
            selected.append(i);histories.append(history)
        if not selected:return geometry
        dimension=len(histories[0][0][1]);features=np.zeros((len(selected),8,dimension),np.float32);times=np.zeros((len(selected),8),np.int64);valid=np.zeros((len(selected),8),bool)
        for i,history in enumerate(histories):
            for j,(frame,rgb,_) in enumerate(history):features[i,j]=rgb;times[i,j]=frame;valid[i,j]=True
        inputs=[torch.from_numpy(v) for v in [features,times,valid,np.full(len(selected),self.frame_id,np.int64)]]
        with torch.inference_mode():
            if self.memory_mode=='predictor':
                self.predictor.eval();prototype,available=self.predictor(*inputs)
            elif self.memory_mode in ['last','ema','fifo']:prototype,available=bounded_identity_prototype(*inputs,mode=self.memory_mode,ema=self.ema)
            else:prototype,available=bounded_identity_prototype(*inputs,mode='last')
        prototype=prototype.numpy();result=geometry.copy()
        candidate_valid=np.array([d.current_rgb is not None for d in detections],bool)
        for row,i in enumerate(selected):
            if not available[row]:continue
            track=tracks[i]
            if self.memory_mode=='all_history_ema':p=track.memory_rgb
            else:p=prototype[row]
            costs=np.array([float(np.clip((1-p@d.current_rgb)/2,0,1)) if d.current_rgb is not None else 1. for d in detections],np.float64)
            if self.memory_mode=='constant_zero':costs[:]=0.
            elif self.memory_mode=='constant_half':costs[:]=.5
            # This graph is independent of predicted/last/EMA identity distances.
            eligible=candidate_valid & (raw[i]<=1-self.proximity_iou)
            result[i,eligible]=(1-self.appearance_weight)*geometry[i,eligible]+self.appearance_weight*costs[eligible]
            self.prediction_counts['predicted_rows']+=1;self.prediction_counts['modified_edges']+=int(eligible.sum())
        return result
