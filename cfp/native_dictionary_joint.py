"""GT-free recent actual observations and full-set ID binding at inference."""
import numpy as np
import torch
from .joint_identity_decoder import decode_joint_assignment
from .native_local_identity import RecentIdentityControl
from yolox.tracker import matching


def native_joint_inputs(tracks, detections, frame, capacity, rng, length=4, maximum_age=1500, dimension=384):
    n, m = len(tracks), len(detections)
    if n > capacity:
        raise ValueError('No silent whole-set truncation or fallback')
    past=np.zeros((n,length,dimension),np.float32);times=np.zeros((n,length),np.int64);valid=np.zeros((n,length),bool)
    for i,track in enumerate(tracks):
        rows=[r for r in track.identity_history if frame-maximum_age<=r[0]][-length:]
        for j,(t,rgb,_) in enumerate(rows):
            if not 0<t<frame or rgb.shape!=(dimension,):raise ValueError('Exact strictly-past native coordinate source required')
            past[i,j],times[i,j],valid[i,j]=rgb,t,True
    current=np.zeros((m,dimension),np.float32);available=np.zeros(m,bool)
    for j,detection in enumerate(detections):
        if detection.current_rgb is not None:
            if detection.current_rgb.shape!=(dimension,):raise ValueError('Current proposal coordinate source differs')
            current[j]=detection.current_rgb;available[j]=True
    inputs=dict(history=torch.from_numpy(past),times=torch.from_numpy(times),valid=torch.from_numpy(valid),
        frame=torch.full((n,),frame,dtype=torch.int64),detections=torch.from_numpy(current),detection_valid=torch.from_numpy(available),
        geometry=torch.from_numpy(matching.iou_distance(tracks,detections).astype(np.float32)),
        track_state=torch.tensor([[np.log1p(frame-t.start_frame)/np.log(15001),np.log1p(frame-t.end_frame)/np.log(15001),float(t.state==2)] for t in tracks],dtype=torch.float32).reshape(n,3),
        confidence=torch.tensor([d.score for d in detections],dtype=torch.float32),codes=torch.from_numpy(rng.permutation(capacity)[:n].astype(np.int64)))
    with torch.no_grad():
        prototype,_=RecentIdentityControl('fifo',length,maximum_age)(inputs['history'],inputs['times'],inputs['valid'],inputs['frame'])
        inputs['appearance']=((1-prototype@inputs['detections'].T)/2).clamp(0,1)
    return inputs


def install_dictionary_joint(tracker,model,seed=0):
    model.eval();base=type(tracker)
    class JointTracker(base):
        def association_cost(self,tracks,detections):
            baseline=super().association_cost(tracks,detections)
            if not baseline.size or any(not t.is_activated for t in tracks) or any(d.score<self.args.track_thresh for d in detections):return baseline
            inputs=native_joint_inputs(tracks,detections,self.frame_id,self.joint_model.capacity,self.joint_rng)
            with torch.inference_mode():
                output=self.joint_model(**inputs);selected=decode_joint_assignment(output['logits'],inputs['detection_valid']).numpy()
            result=np.full_like(baseline,1.01)
            for detection,track in enumerate(selected):
                if track>=0:result[track,detection]=0.
            # An unextractable current crop retains original geometry decisions.
            unavailable=~inputs['detection_valid'].numpy();result[:,unavailable]=baseline[:,unavailable]
            self.joint_counts['scored_frames']+=1;self.joint_counts['scored_tracks']+=len(tracks)
            self.joint_counts['null_detections']+=int((selected<0).sum());self.joint_counts['unavailable_fallback_columns']+=int(unavailable.sum())
            return result
    tracker.__class__=JointTracker;tracker.joint_model=model;tracker.joint_rng=np.random.default_rng(seed)
    tracker.joint_counts=dict(scored_frames=0,scored_tracks=0,null_detections=0,unavailable_fallback_columns=0)
    return tracker
