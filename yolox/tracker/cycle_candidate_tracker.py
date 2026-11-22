"""Causal object hypotheses for repeated low-score body observations."""
import numpy as np

from yolox.tracker.foreground_byte_tracker import ObjectFileTrack
from yolox.tracker.residual_foreground_tracker import ResidualForegroundTracker
from yolox.tracker.matching import ious


class CycleBirthMixin:
    def __init__(self,config):
        config=dict(config,candidate_birth=False)
        super().__init__(config)
        self.provisional={}
        if 'low_track_threshold' in config:
            self.inner.args.track_thresh=float(config['low_track_threshold'])
            self.inner.det_thresh=float(config['low_track_threshold'])+.1
        self.statistics.update(cycle_candidates=0,cycle_confirmations=0,cycle_rejections=0)

    def update(self,frame,data):
        rows=super().update(frame,data)
        if not self.config.get('cycle_birth',False):return rows
        point=data['foreground_transport'];source='rgb';target=self.config.get('cycle_foreground','joint')
        a,b=map(int,data['offsets'][frame-1:frame+1])
        used={t.evidence['index'] for t in self.inner.tracked_stracks if t.frame_id==frame}
        minimum=float(self.config.get('candidate_minimum_score',.3));required=int(self.config.get('candidate_observations',3))
        if required not in {2,3}:raise ValueError('Only registered2/3-observation candidates')
        current={o.evidence['index']:o for o in self.created if o.evidence['index'] not in used and o.score>=minimum}
        def lookup(lag):
            c,d=map(int,data['offsets'][frame-lag-1:frame-lag+1])
            v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1]);p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
            matrix=point[f'vote_{lag}_{source}_{target}'][v0:v1].reshape(d-c,b-a).astype(np.float64)
            return c,matrix,point[f'good_{lag}_rgb'][p0:p1],point[f'total_{lag}_rgb'][p0:p1]
        def verified(context,index,lag):
            start,matrix,good,total=lookup(lag);row=context['index']-start;column=index-a
            if not matrix.size or row<0 or row>=len(matrix):return False
            value=matrix[row];strength=float(value[column]);other=np.delete(value,column)
            purity=(strength-(float(other.max()) if len(other) else 0.))/max(strength,1e-8)
            if strength<self.config.get('candidate_minimum_vote',.3) or purity<.5:return False
            if int(value.argmax())!=column or good[row]<4 or good[row]/max(float(total[row]),1.)<.6:return False
            alternatives=matrix[:,column].copy();alternatives[row]=0
            if float(alternatives.max())>.8*strength:return False
            quality_index=1 if target in {'ir','joint'} else 0
            def quality(i):
                q=point['quality'][i];return q[quality_index] if q[quality_index]>=.15 else q[0]
            return min(quality(context['index']),quality(index))>=.15
        updated={}
        for index,observation in current.items():
            history=[observation.evidence]
            if frame>1:
                parents=[p for p in self.provisional.values() if p[-1]['frame']==frame-1 and verified(p[-1],index,1)]
                if len(parents)==1:history=parents[0]+history
            history=history[-required:]
            self.statistics['cycle_candidates']+=1
            if len(history)<required:updated[index]=history;continue
            if self.config.get('candidate_cycle_check',True) and required==3 and not verified(history[0],index,2):
                self.statistics['cycle_rejections']+=1;updated[index]=history;continue
            active=[t for t in self.inner.tracked_stracks if t.frame_id==frame]
            if active and float(ious(observation.tlbr[None],np.stack([t.tlbr for t in active])).max())>.7:
                updated[index]=history;continue
            if observation.tlwh[2]*observation.tlwh[3]<=100:continue
            observation.activate(self.inner.kalman_filter,frame)
            observation.is_activated=True;observation.object_observations=history
            self.inner.tracked_stracks.append(observation)
            rows.append((int(frame),observation.tlwh.copy(),observation.track_id,observation.score))
            self.statistics['cycle_confirmations']+=1;self.statistics['candidate_births']+=1
            if target in {'ir','joint'} and all(c['quality'][1]>=.15 for c in history):self.statistics['IR_supported_births']+=1
        self.provisional=updated
        return rows


class CycleForegroundTracker(CycleBirthMixin,ResidualForegroundTracker):
    pass
