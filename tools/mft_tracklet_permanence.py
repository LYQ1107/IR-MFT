"""Online identity links between removed and first-published tracklets.

Original ByteTrack costs, geometry and published trajectory prefixes stay fixed.
No network, GT, file IO or future descriptor access belongs to this component.
"""
from types import SimpleNamespace
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from yolox.tracker import matching


def unit(value):
    value = np.asarray(value,dtype=np.float32)
    norm = np.linalg.norm(value)
    if value.ndim != 1 or not np.isfinite(value).all() or norm < 1e-9:
        return None
    return value/norm


class MemoryTrack(STrack):
    def __init__(self,box,score,observation,counter):
        super().__init__(box,score)
        self.observation,self.counter = observation,counter
        self.rgb,self.ir,self.last_box = None,None,None
        self.accepted = 0

    def next_id(self):
        self.counter[0] += 1
        return self.counter[0]

    def accept(self,obs):
        if obs is None:
            return
        self.last_box = obs['box'].copy()
        if not obs['reliable']:
            return
        rgb,ir = unit(obs['rgb']),unit(obs['ir'])
        if rgb is None or ir is None:
            return
        self.rgb = rgb if self.rgb is None else unit(.8*self.rgb+.2*rgb)
        self.ir = ir if self.ir is None else unit(.8*self.ir+.2*ir)
        self.accepted += 1

    def activate(self,kalman_filter,frame_id):
        super().activate(kalman_filter,frame_id)
        self.accept(self.observation)

    def update(self,new_track,frame_id):
        super().update(new_track,frame_id)
        self.accept(new_track.observation)

    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id)
        self.accept(new_track.observation)


def cases():
    return [dict(id='TP_native_zero',buffer=25,alias=False,use_IR=False,maximum_gap=250),
            dict(id='TP_retention250',buffer=250,alias=False,use_IR=False,maximum_gap=250),
            dict(id='TP_late_RGB',buffer=25,alias=True,use_IR=False,maximum_gap=250),
            dict(id='TP_late_RGB_IR',buffer=25,alias=True,use_IR=True,maximum_gap=250),
            dict(id='TP_retention250_late_RGB_IR',buffer=250,alias=True,use_IR=True,maximum_gap=1000)]


class TrackletPermanence:
    def __init__(self,case,thresholds):
        assert case in cases()
        self.case,self.thresholds = dict(case),dict(thresholds)
        self.sequence,self.core = None,None

    def reset(self,sequence):
        self.sequence = sequence
        counter = [0]
        args = SimpleNamespace(track_thresh=.6,track_buffer=self.case['buffer']*30//25,match_thresh=.9,mot20=False)
        self.core = BYTETracker(args,frame_rate=25,track_factory=lambda box,score,obs:MemoryTrack(box,score,obs,counter))
        assert self.core.max_time_lost == self.case['buffer']
        self.published,self.last_published,self.closed = {},{},{}
        self.removed_prefix = 0
        self.statistics = dict(identity_links=0,eligible_removed_memories=0,candidate_pairs=0,
            RGB_gate_passes=0,IR_gate_passes=0,primary_association_costs_changed=0,published_IDs_reassigned=0)

    def archive(self,frame):
        removed = self.core.removed_stracks
        assert len(removed) >= self.removed_prefix
        for track in removed[self.removed_prefix:]:
            identity = self.published.get(track.track_id)
            if identity is not None and track.accepted >= 3 and track.rgb is not None and track.last_box is not None:
                self.closed[identity] = dict(rgb=track.rgb.copy(),ir=track.ir.copy(),box=track.last_box.copy(),
                    last_frame=self.last_published[identity],internal_track=track.track_id)
            track.rgb,track.ir,track.observation = None,None,None
        self.removed_prefix = len(removed)
        self.closed = {key:value for key,value in self.closed.items()if frame-value['last_frame'] <= self.case['maximum_gap']}

    def link(self,new,frame,active_roots):
        if not self.case['alias'] or not new:
            return {}
        old = [(key,value)for key,value in sorted(self.closed.items())if key not in active_roots]
        if not old:
            return {}
        scores = np.full((len(new),len(old)),-np.inf,np.float64)
        for i,track in enumerate(new):
            if track.accepted < 2 or track.rgb is None:
                continue
            for j,(identity,memory) in enumerate(old):
                gap = frame-memory['last_frame']
                assert gap > 0
                if not 2 <= gap <= self.case['maximum_gap']:
                    continue
                old_box,current_box = memory['box'],track.last_box
                a,b = np.maximum(old_box[2:]-old_box[:2],1.),np.maximum(current_box[2:]-current_box[:2],1.)
                scale = b/a
                if (scale < .5).any() or (scale > 2.).any():
                    continue
                distance = np.linalg.norm(((old_box[:2]+old_box[2:])-(current_box[:2]+current_box[2:]))/2)/np.linalg.norm(a)
                if distance > .5+self.thresholds['Train_unknown_speed_upper']*gap:
                    continue
                self.statistics['candidate_pairs'] += 1
                rgb,ir = float(track.rgb@memory['rgb']),float(track.ir@memory['ir'])
                if rgb < self.thresholds['RGB_cosine_threshold']:
                    continue
                self.statistics['RGB_gate_passes'] += 1
                if self.case['use_IR'] and ir < self.thresholds['IR_cosine_threshold']:
                    continue
                self.statistics['IR_gate_passes'] += int(self.case['use_IR'])
                scores[i,j] = rgb
        result = {}
        for i,track in enumerate(new):
            if not np.isfinite(scores[i]).any():
                continue
            j = int(np.argmax(scores[i]))
            if int(np.argmax(scores[:,j])) != i:
                continue
            row = np.delete(scores[i],j);column = np.delete(scores[:,j],i)
            second_row = float(row.max())if len(row)else -np.inf
            second_column = float(column.max())if len(column)else -np.inf
            if scores[i,j]-max(second_row,second_column) < .1:
                continue
            identity = old[j][0]
            assert identity not in result.values() and track.track_id not in self.published
            result[track.track_id] = identity
            del self.closed[identity]
            self.statistics['identity_links'] += 1
        return result

    def update(self,sequence,frame,detections,rgb=None,ir=None):
        if self.sequence != sequence:
            assert frame == 1
            self.reset(sequence)
        assert frame == self.core.frame_id+1
        values = np.asarray(detections,dtype=np.float32)
        assert values.ndim == 2 and values.shape[1] == 5 and np.isfinite(values).all()
        evidence = None
        if self.case['alias']:
            rgb,ir = np.asarray(rgb),np.asarray(ir)
            assert rgb.ndim == ir.ndim == 2 and rgb.shape == ir.shape and len(rgb) == len(values)
            assert np.isfinite(rgb).all() and np.isfinite(ir).all()
            overlap = matching.ious(values[:,:4],values[:,:4])if len(values)else np.empty((0,0))
            np.fill_diagonal(overlap,0)
            maximum = overlap.max(1)if len(values)else np.empty(0)
            evidence = [dict(rgb=rgb[i],ir=ir[i],box=values[i,:4].copy(),reliable=bool(values[i,4]>.6 and maximum[i]<=.5))for i in range(len(values))]
        tracks = self.core.update(values.copy(),(1,1),(1,1),evidence=evidence)
        visible = [t for t in tracks if t.tlwh[2]*t.tlwh[3] > 100]
        self.archive(frame)
        active = {self.published[t.track_id]for t in visible if t.track_id in self.published}
        new = [t for t in visible if t.track_id not in self.published]
        links = self.link(new,frame,active)
        for track in new:
            self.published[track.track_id] = links.get(track.track_id,track.track_id)
        result = []
        for track in visible:
            identity = self.published[track.track_id]
            assert identity not in [row[2]for row in result]
            self.last_published[identity] = frame
            result.append((frame,track.tlwh.copy(),identity,track.score))
        self.statistics['eligible_removed_memories'] = len(self.closed)
        for track in self.core.tracked_stracks+self.core.lost_stracks:
            track.observation = None
        return result
