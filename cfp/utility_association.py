"""Utility-gated association under a fixed RGB-defined candidate set."""
import numpy as np
import torch

from .association import AppearanceTracker
from .utility import decision_evidence, utility_mix
from .candidate_context import complete_candidate_context
from yolox.tracker import matching


class UtilityAssociationTracker(AppearanceTracker):
    def __init__(self, args, model=None, kind='cmue', calibration=None, threshold=.02,
                 no_calibration=False, no_fallback=False, **kwargs):
        if kind not in ('cmue','ordinary','quality','fixed_matched','untrained','fixed_constant_ir','fixed_row_mean_ir'):
            raise ValueError('Unknown utility association control')
        super().__init__(args, mode='rgb', **kwargs)
        if kind not in ('fixed_matched','fixed_constant_ir','fixed_row_mean_ir') and model is None:
            raise ValueError('Gate model is required')
        self.utility_model=model
        self.utility_kind=kind
        self.calibration=calibration or dict(slope=1.,intercept=0.)
        self.utility_threshold=threshold
        self.no_calibration=no_calibration
        self.no_fallback=no_fallback
        self.candidate_context_version=1
        self.teacher_association_weight=.3
        self.utility_counts=dict(evaluated_rows=0,accepted_rows=0,rejected_rows=0,missing_ir_rows=0,weight_sum=0.,weight_squared_sum=0.)

    def association_cost(self, tracks, detections):
        baseline=super().association_cost(tracks,detections)
        if not tracks or not detections or self.appearance_weight==0:
            return baseline
        raw_geometry=matching.iou_distance(tracks,detections)
        geometry=matching.fuse_score(raw_geometry,detections) if not self.args.mot20 else raw_geometry
        contexts=[]
        for i,track in enumerate(tracks):
            if track.memory_rgb is None:continue
            rgb=np.array([float(np.clip((1-track.memory_rgb@d.current_rgb)/2,0,1)) if d.current_rgb is not None else 1. for d in detections],np.float64)
            valid=np.array([d.current_rgb is not None for d in detections],bool)
            selected=np.flatnonzero(valid & (raw_geometry[i]<=1-self.proximity_iou) & (rgb<=self.appearance_max_distance))
            if not len(selected):continue
            ir=rgb[selected].copy();available=np.zeros(len(selected),bool)
            for k,j in enumerate(selected):
                d=detections[j]
                if track.memory_ir is not None and d.current_ir is not None:
                    ir[k]=float(np.clip((1-track.memory_ir@d.current_ir)/2,0,1));available[k]=True
            if self.utility_kind=='fixed_constant_ir':
                ir[available]=.5
            elif self.utility_kind=='fixed_row_mean_ir' and available.any():
                ir[available]=float(ir[available].mean())
            if not available.any():
                self.utility_counts['missing_ir_rows']+=1
                continue
            evidence=decision_evidence(rgb[selected],ir,raw_geometry[i,selected],np.array([detections[j].score for j in selected]),available,
                     self.frame_id-track.start_frame+1,self.frame_id-track.last_identity_frame,[r[1] for r in track.identity_history])
            if self.candidate_context_version==2 and self.utility_kind not in ('fixed_matched','fixed_constant_ir','fixed_row_mean_ir'):
                complete=complete_candidate_context(track,detections,raw_geometry[i],geometry[i],self.frame_id,
                             teacher_weight=self.teacher_association_weight,match_threshold=self.args.match_thresh,
                             proximity_iou=self.proximity_iou,appearance_max_distance=self.appearance_max_distance)
                if complete is None:continue
                evidence=complete['evidence']
            contexts.append((i,selected,rgb[selected],ir,evidence))
        if not contexts:return baseline
        if self.utility_kind in ('fixed_matched','fixed_constant_ir','fixed_row_mean_ir'):values=np.zeros(len(contexts))
        else:
            self.utility_model.eval()
            with torch.inference_mode():
                values=self.utility_model(torch.from_numpy(np.stack([row[4] for row in contexts]))).cpu().numpy()
            if not np.isfinite(values).all():raise ValueError('Nonfinite utility prediction')
        result=baseline.copy()
        for (i,selected,rgb,ir,_),value in zip(contexts,values):
            if self.utility_kind in ('ordinary','quality','untrained'):
                weight=float(.5/(1+np.exp(-np.clip(value,-80,80))))
                appearance=(1-weight)*rgb+weight*ir
            elif self.utility_kind in ('fixed_matched','fixed_constant_ir','fixed_row_mean_ir'):
                weight=.5;appearance=.5*rgb+.5*ir
            else:
                utility=float(value if self.no_calibration else self.calibration['slope']*value+self.calibration['intercept'])
                appearance,weight=utility_mix(rgb,ir,utility,threshold=self.utility_threshold)
                if weight==0 and self.no_fallback:
                    weight=.5;appearance=.5*rgb+.5*ir
            self.utility_counts['evaluated_rows']+=1
            self.utility_counts['accepted_rows' if weight>0 else 'rejected_rows']+=1
            self.utility_counts['weight_sum']+=weight;self.utility_counts['weight_squared_sum']+=weight**2
            if weight>0:
                # Preserve unchanged/missing-IR edges in the original precision.
                # utility_mix returns FP32; copying such edges would round RGB costs.
                changed=ir!=rgb
                result[i,selected[changed]]=(1-self.appearance_weight)*geometry[i,selected[changed]]+self.appearance_weight*appearance[changed]
            # Rejected rows remain the original RGB baseline values bit-exact.
        return result
