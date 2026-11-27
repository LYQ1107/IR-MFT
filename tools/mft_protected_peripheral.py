"""Immutable trusted tracking with spatially unexplained peripheral evidence.

Every trusted observation survives. Publishers receive read-only snapshots, not
the mutable teacher. Auxiliary tracks never enter its association or memory.
Confidence mass is an uncalibrated heuristic, not an expected object count.
"""
import hashlib
from types import MappingProxyType
import numpy as np
from tools.mft_birth_only import BirthOnlyTeacher,BirthOnlyQuality
from tools.mft_proposal_quality import ProposalConfirmation,current_features
from tools.mft_proposal_quality_evaluation import ShapeTeacher
from tools.mft_scene_coverage import KINDS,coverage_features
from tools.native_pyramid_stream_tracks import mot_line


def covered_fraction(candidates,regions):
    """Exact union of rectangles clipped to each candidate, without GT."""
    boxes=np.asarray(candidates,np.float64).reshape(-1,4)
    regions=np.asarray(regions,np.float64).reshape(-1,4)
    assert np.isfinite(boxes).all()and np.isfinite(regions).all()
    assert (boxes[:,2:]>boxes[:,:2]).all()and (regions[:,2:]>regions[:,:2]).all()
    result=np.zeros(len(boxes),np.float64)
    for index,box in enumerate(boxes):
        lo=np.maximum(box[:2],regions[:,:2]);hi=np.minimum(box[2:],regions[:,2:])
        keep=(hi>lo).all(axis=1);rect=np.column_stack([lo[keep],hi[keep]])
        if not len(rect):continue
        xs=np.unique(rect[:,[0,2]]);area=0.
        for left,right in zip(xs[:-1],xs[1:]):
            active=rect[(rect[:,0]<right)&(rect[:,2]>left)]
            if not len(active):continue
            intervals=active[:,[1,3]];intervals=intervals[np.argsort(intervals[:,0],kind='stable')]
            bottom,top=intervals[0];height=0.
            for a,b in intervals[1:]:
                if a>top:height+=top-bottom;bottom,top=a,b
                else:top=max(top,b)
            height+=top-bottom;area+=(right-left)*height
        fraction=area/np.prod(box[2:]-box[:2])
        assert -1e-12<=fraction<=1+1e-12
        result[index]=np.clip(fraction,0,1)
    return result


def _readonly(a):
    a=np.asarray(a).copy();a.setflags(write=False);return a


def _ratio(probability,candidates,assigned,weight):
    unseen=float((probability[candidates]*weight[candidates]).sum())
    seen=float(probability[assigned].sum())
    return unseen/(unseen+seen)if unseen+seen>0 else 0.


class ProtectedStreams:
    """One unchanged trusted parent and one independent original-score shadow."""
    def __init__(self,views,heads,thresholds,speed,archive_models=None,archive_thresholds=None,*,standalone=False):
        self.views,self.heads=views,heads;self.standalone=standalone
        self.raw=BirthOnlyTeacher(speed);self.shadow=ProposalConfirmation()
        if standalone:self.main=ShapeTeacher(speed)
        else:self.main=BirthOnlyQuality(dict(id='BO_paired_q999_c3',kind='paired',quantile='q999',confirmation=3),
            views,heads,thresholds,speed,archive_models,archive_thresholds)
        self.sequence=None;self.frame=0

    def update(self,sequence,frame,d,rgb,ir=None):
        if sequence==self.sequence and frame==self.frame:
            for original,current in zip(self.packet,[d,rgb,ir]):
                if original is None or current is None:assert original is current
                else:np.testing.assert_array_equal(original,current)
            return self.value
        assert sequence!=self.sequence and frame==1 or sequence==self.sequence and frame==self.frame+1
        streak=self.shadow.update(sequence,frame,d);self.raw.update(sequence,frame,d,rgb,ir)
        rows=self.main.update(sequence,frame,d,rgb,ir)
        teacher=self.main.teacher
        assigned=np.asarray(list(teacher.current_proposal_assignments(frame).values()),np.int64)
        main_proposals={}
        published=None if self.standalone else self.main.parent.linker.published
        for t in teacher.core.tracked_stracks:
            if t.is_activated and t.measurement_frame==frame:
                identity=int(t.track_id)if published is None else published.get(int(t.track_id))
                if identity is not None:main_proposals[int(identity)]=int(t.proposal_index)
        assert all(int(row[2])in main_proposals for row in rows)
        unused=[e['proposal_index']for e in self.raw.core.last_birth_events if not e['original_allowed']]
        original_assigned=list(self.raw.current_proposal_assignments(frame).values())
        _,old=coverage_features(d,rgb,ir,self.views,self.heads,streak,unused,original_assigned)
        x=current_features(d,rgb,ir,self.views)
        probabilities={k:self.heads[k].probabilities(x)for k in ['RGB','paired','duplicatedRGB']}
        if ir is None:probabilities['paired']=probabilities['RGB'].copy()
        rolled=current_features(d,rgb,None if ir is None else np.roll(ir,1,axis=0),self.views)
        probabilities.update(IR_roll=self.heads['paired'].probabilities(rolled)if ir is not None else probabilities['RGB'].copy(),
            IR_missing=probabilities['RGB'].copy(),count=np.ones(len(d)),score=np.asarray(d[:,4],np.float64))
        explained=covered_fraction(d[:,:4],d[assigned,:4])
        potential=(d[:,4]>.3)&(d[:,4]<.7)&(streak>=3);potential[assigned]=False
        candidate_indices=np.flatnonzero(potential);weights=1-explained
        needs={
            'original':{k:old[k]['deficit']for k in KINDS},
            'expanded':{k:_ratio(p,candidate_indices,assigned,weights)for k,p in probabilities.items()},
            'expanded_without_explanation':{k:_ratio(p,candidate_indices,assigned,np.ones(len(d)))for k,p in probabilities.items()}}
        assert all(v['RGB']==v['IR_missing']for v in needs.values())
        records=[];seen=set()
        for t in self.shadow.core.tracked_stracks:
            if not t.is_activated or t.measurement_frame!=frame:continue
            index=int(t.proposal_index);assert index not in seen;seen.add(index)
            measured=d[index,:4];size=measured[2:]-measured[:2];filtered=t.tlwh.copy()
            box=np.r_[filtered[:2]+filtered[2:]/2-size/2,size]
            records.append(dict(shadow_id=int(t.track_id),proposal_index=index,streak=int(t.streak),
                score=float(d[index,4]),box_xyxy=_readonly(measured),tlwh=_readonly(box)))
        value=MappingProxyType(dict(sequence=sequence,frame=frame,main_rows=tuple((f,_readonly(box),int(i),float(s))for f,box,i,s in rows),
            main_proposals=MappingProxyType(main_proposals),main_assigned=frozenset(map(int,assigned)),
            shadow_records=tuple(MappingProxyType(r)for r in records),explained=_readonly(explained),
            expanded_candidates=_readonly(candidate_indices),
            needs=MappingProxyType({k:MappingProxyType(v)for k,v in needs.items()}),
            main_statistics=MappingProxyType(dict(self.main.statistics)),raw_statistics=MappingProxyType(dict(self.raw.statistics))))
        self.sequence,self.frame=sequence,frame;self.packet=[None if a is None else _readonly(a)for a in [d,rgb,ir]]
        self.value=value;return value


class PeripheralPublisher:
    """Keep all trusted rows; causally allocate auxiliary IDs and handoffs."""
    def __init__(self,case):self.case=dict(case);self.sequence=None

    def reset(self,sequence):
        self.sequence=sequence;self.frame=0;self.main_alias={};self.owner={};self.used=set();self.maximum=0
        self.aux_alias={};self.last_aux_publication={}
        self.main_projection=hashlib.sha256();self.handoff_packet=hashlib.sha256()
        self.statistics=dict(frames=0,coverage_gated_frames=0,auxiliary_birth_publications=0,
            auxiliary_rows=0,main_rows=0,spatial_rejections=0,auxiliary_duplicate_rejections=0,
            proposal_ownership_rejections=0,identity_handoffs=0,numeric_ID_collisions=0)

    def allocate(self,preferred=None):
        if preferred is not None and preferred not in self.used:identity=int(preferred)
        else:
            if preferred is not None:self.statistics['numeric_ID_collisions']+=1
            identity=self.maximum+1
            while identity in self.used:identity+=1
        assert identity>0 and identity not in self.used
        self.used.add(identity);self.maximum=max(self.maximum,identity);return identity

    def update(self,value):
        sequence=value['sequence'];frame=value['frame']
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        assert frame==self.frame+1
        kind=self.case['kind'];mode=self.case.get('intervention');explain=self.case.get('explain',True)
        need_mode=self.case.get('need_mode','original')
        if not explain and need_mode=='expanded':need_mode='expanded_without_explanation'
        if kind=='reference':gate=False
        elif kind=='unconditional':gate=True
        else:
            assert kind in ['count','RGB','paired','duplicatedRGB']and mode in [None,'roll','missing']
            effective='IR_roll'if mode=='roll'else'IR_missing'if mode=='missing'else kind
            gate=value['needs'][need_mode][effective]>=self.case['threshold']
        self.statistics['coverage_gated_frames']+=int(gate)
        by_proposal={r['proposal_index']:r for r in value['shadow_records']}
        output=[];current_public=set()
        for f,box,base_id,score in value['main_rows']:
            assert f==frame and base_id>0
            if base_id not in self.main_alias:
                record=by_proposal.get(value['main_proposals'][base_id]);sid=None if record is None else record['shadow_id']
                auxiliary=None if sid is None else self.aux_alias.get(sid)
                handoff=(self.case.get('handoff',True)and auxiliary is not None and auxiliary not in self.owner
                    and frame-self.last_aux_publication[sid]<=2)
                if handoff:
                    public=auxiliary;self.statistics['identity_handoffs']+=1
                    self.handoff_packet.update(np.asarray([frame,base_id,sid,public],dtype='<i8').tobytes())
                else:public=self.allocate(base_id)
                assert public not in self.owner;self.owner[public]=base_id;self.main_alias[base_id]=public
            public=self.main_alias[base_id]
            assert self.owner[public]==base_id and public not in current_public
            current_public.add(public);output.append((f,box.copy(),public,score))
            self.main_projection.update(mot_line((f,box,self.owner[public],score)).encode())
        assert len(self.main_alias)==len(set(self.main_alias.values()))
        records=sorted(value['shadow_records'],key=lambda r:(r['shadow_id']not in self.aux_alias,-r['streak'],-r['score'],r['shadow_id']))
        auxiliary_boxes=[]
        for record in records:
            index=record['proposal_index'];sid=record['shadow_id'];existing=sid in self.aux_alias
            if index in value['main_assigned']:
                self.statistics['proposal_ownership_rejections']+=1;continue
            if kind=='reference':continue
            if not existing and not(gate and .3<record['score']<.7 and record['streak']>=3):continue
            if existing and self.aux_alias[sid]in current_public:continue
            if explain and value['explained'][index]>=.5:
                self.statistics['spatial_rejections']+=1;continue
            if not existing and explain and len(auxiliary_boxes)and covered_fraction(record['box_xyxy'][None],auxiliary_boxes)[0]>=.5:
                self.statistics['auxiliary_duplicate_rejections']+=1;continue
            if np.prod(record['tlwh'][2:])<=100:continue
            if not existing:
                self.aux_alias[sid]=self.allocate();self.statistics['auxiliary_birth_publications']+=1
            public=self.aux_alias[sid];assert public not in current_public
            current_public.add(public);self.last_aux_publication[sid]=frame
            output.append((frame,record['tlwh'].copy(),public,record['score']));auxiliary_boxes.append(record['box_xyxy'])
            self.statistics['auxiliary_rows']+=1
        assert len({row[2]for row in output})==len(output)
        self.statistics['frames']+=1;self.statistics['main_rows']+=len(value['main_rows']);self.frame=frame
        return output


def policies():
    rows=[dict(id='PP_reference',kind='reference',need_mode='original',threshold=None),
        dict(id='PP_unconditional',kind='unconditional',need_mode='original',threshold=None)]
    for mode in ['original','expanded']:
        for threshold in [.1,.25]:
            for kind in ['count','RGB','paired','duplicatedRGB']:
                rows.append(dict(id='PP_'+mode+'_'+kind+'_t'+str(threshold).replace('.',''),kind=kind,
                    need_mode=mode,threshold=threshold))
    return rows


class ProtectedPeripheralTracker:
    def __init__(self,case,views,heads,thresholds,speed,archive_models=None,archive_thresholds=None,*,standalone=False,shared=None):
        self.publisher=PeripheralPublisher(case);self.standalone=standalone
        self.streams=ProtectedStreams(views,heads,thresholds,speed,archive_models,archive_thresholds,standalone=standalone)if shared is None else shared

    def update(self,sequence,frame,d,rgb,ir=None):return self.publisher.update(self.streams.update(sequence,frame,d,rgb,ir))

    @property
    def statistics(self):
        return dict(self.publisher.statistics,standalone=self.standalone,
            original_main_projection_MOT_sha256=self.publisher.main_projection.hexdigest(),
            identity_handoff_packet_sha256=self.publisher.handoff_packet.hexdigest(),
            main_alias_count=len(self.publisher.main_alias),aux_alias_count=len(self.publisher.aux_alias),
            maximum_published_ID=self.publisher.maximum,trusted_state_modified_by_auxiliary=False)
