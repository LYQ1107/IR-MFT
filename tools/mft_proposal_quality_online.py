"""Actual-image native visual detector, frozen crop encoders and quality tracker."""
import numpy as np
import torch
from torch.nn import functional as F
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.mft_proposal_quality_evaluation import SharedConfirmation,make_tracker
from tools.run_mft_tracklet_permanence_r2 import encode
from yolox.tracker.native_cue import native_crop,native_input


@torch.no_grad()
def RGB_only(encoders,image,boxes):
    values=[];model=encoders['rgb']
    for lo in range(0,len(boxes),32):
        pairs=[native_crop(image,box)for box in boxes[lo:lo+32]]
        pixels=torch.from_numpy(np.stack([p[0]for p in pairs]));rect=torch.from_numpy(np.stack([p[1]for p in pairs]))
        raw=model.projection(model.backbone(native_input(pixels,rect,'cuda')))
        values.append(F.normalize(model.neck(raw),dim=-1).cpu().numpy())
    return np.concatenate(values).astype(np.float32)if values else np.empty((0,512),np.float32)


class ProposalQualityOnline:
    def __init__(self,gpu_models,calibration,views,heads,thresholds,speed,archive_models,archive_thresholds,cases,visual_intervention='normal'):
        assert visual_intervention in ['normal','unavailable','IR_image_spatial_roll']
        teacher,module,visual,encoders,physical,uuid=gpu_models
        row=dict(visual['model']);row['intervention']=visual_intervention
        if visual_intervention=='unavailable':row['expect_RGB_exact']=True
        self.detector=OnlineNativeDetector(teacher,[row],{row['id']:module},(800,1440),calibration)
        self.visual_id=row['id'];self.encoders=encoders;self.visual_intervention=visual_intervention
        self.cases=list(cases);shared=SharedConfirmation()
        self.trackers={c['id']:make_tracker(c,views,heads,thresholds,speed,archive_models,archive_thresholds,shared)for c in cases}
        self.sequence=None;self.last_evidence=None;self.closed=False

    @property
    def needs_actual_IR_image(self):return self.visual_intervention!='unavailable'

    @torch.no_grad()
    def step(self,rgb,ir,sequence,frame):
        assert not self.closed
        packets,info=self.detector.step(rgb,ir,sequence,frame);d=packets[self.visual_id]
        if self.visual_intervention=='unavailable':r=RGB_only(self.encoders,rgb,d[:,:4]);i=None
        else:
            image=ir
            if self.visual_intervention=='IR_image_spatial_roll':image=np.roll(ir,(ir.shape[0]//4,ir.shape[1]//4),(0,1))
            values=encode(self.encoders,rgb,image,d[:,:4]);r,i=values['rgb'],values['ir']
        self.last_evidence=dict(detections=d.copy(),rgb=r.copy(),ir=None if i is None else i.copy())
        rows={c['id']:self.trackers[c['id']].update(sequence,frame,d,r,i)for c in self.cases};self.sequence=sequence
        return rows,{c['id']:d for c in self.cases},dict(visual=info,actual_FP32_postBN_crop_features=True,
            detector_and_quality_IR_intervention=self.visual_intervention,GT_labels_parsed_or_used=False,
            future_frame_input=False,previous_outputs_rewritten=False)

    def close(self):
        self.closed=True;self.detector.states.clear();self.last_evidence=None
