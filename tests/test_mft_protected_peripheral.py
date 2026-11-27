import hashlib
import unittest
import numpy as np
from tools.mft_protected_peripheral import PeripheralPublisher,covered_fraction
from tools.mft_peripheral_frame_bank import FrameBankBuilder,SequentialFrameBank
from tools.native_pyramid_stream_tracks import mot_line


def record(identity,index,box,score=.45,streak=3):
    box=np.asarray(box,np.float32);size=box[2:]-box[:2]
    return dict(shadow_id=identity,proposal_index=index,score=score,streak=streak,
        box_xyxy=box.copy(),tlwh=np.r_[box[:2],size])


def packet(frame,rows,proposals,records=(),explained=None,gate=.3):
    n=max([*proposals.values(),*(r['proposal_index']for r in records),-1])+1
    value=dict(sequence='A',frame=frame,main_rows=rows,main_proposals=proposals,
        main_assigned=frozenset(proposals.values()),shadow_records=records,
        explained=np.zeros(n)if explained is None else np.asarray(explained),
        needs={k:{p:gate for p in ['count','score','RGB','paired','duplicatedRGB','IR_roll','IR_missing']}
            for k in ['original','expanded','expanded_without_explanation']})
    return value


def case(**options):return dict(kind='paired',need_mode='original',threshold=.25,**options)


class PeripheralTests(unittest.TestCase):
    def test_union_containment_and_empty_geometry(self):
        box=np.array([[0,0,100,100],[200,0,220,20]],float)
        regions=np.array([[0,0,30,100],[70,0,100,100]],float)
        np.testing.assert_array_equal(covered_fraction(box,regions),[.6,0])
        np.testing.assert_array_equal(covered_fraction(box,[]),[0,0])
        np.testing.assert_array_equal(covered_fraction([[0,0,100,100]],[[0,0,80,100],[20,0,80,100]]),[.8])
        p=PeripheralPublisher(case());row=(1,np.array([0,0,30,100]),1,.9)
        out=p.update(packet(1,[row],{1:0},[record(10,1,[0,0,100,100])],explained=[1,.6]))
        self.assertEqual(len(out),1);self.assertEqual(p.statistics['spatial_rejections'],1)

    def test_no_auxiliary_literal_MOT_and_projection(self):
        p=PeripheralPublisher(case());expected=hashlib.sha256()
        for frame in range(1,10):
            rows=[(frame,np.array([frame,0,30,20],float),1,.9),
                (frame,np.array([100,0,30,20],float),7,.8)]
            expected.update(''.join(map(mot_line,rows)).encode())
            before=[r[1].copy()for r in rows]
            actual=p.update(packet(frame,rows,{1:0,7:1},gate=0))
            self.assertEqual(''.join(map(mot_line,actual)),''.join(map(mot_line,rows)))
            for x,r in zip(before,rows):np.testing.assert_array_equal(x,r[1])
        self.assertEqual(p.main_projection.hexdigest(),expected.hexdigest())
        self.assertEqual(p.statistics['auxiliary_birth_publications'],0)

    def test_causal_handoff_future_numeric_collision_and_owned_duplicate(self):
        p=PeripheralPublisher(case());main=(1,np.array([0,0,20,20]),1,.9)
        first=p.update(packet(1,[main],{1:0},[record(100,1,[100,0,120,20])]))
        self.assertEqual([x[2]for x in first],[1,2])
        rows=[(2,main[1],1,.9),(2,np.array([100,0,20,20]),7,.8)]
        second=p.update(packet(2,rows,{1:0,7:1},[record(100,1,[100,0,120,20],score=.8)]))
        self.assertEqual([x[2]for x in second],[1,2]);self.assertEqual(p.statistics['identity_handoffs'],1)
        rows=[(3,main[1],1,.9),(3,np.array([100,0,20,20]),7,.8),(3,np.array([200,0,20,20]),2,.9)]
        third=p.update(packet(3,rows,{1:0,7:1,2:2}))
        self.assertEqual([x[2]for x in third],[1,2,3]);self.assertEqual(first[1][2],2)
        self.assertEqual(p.main_alias,{1:1,7:2,2:3});self.assertEqual(p.statistics['numeric_ID_collisions'],1)
        fourth=p.update(packet(4,[(4,x[1],x[2],x[3])for x in rows],{1:0,7:1,2:2},
            [record(100,3,[300,0,320,20])]))
        self.assertEqual(len(fourth),3);self.assertEqual(len({r[2]for r in fourth}),3)

    def test_handoff_gap_limit_disable_and_sequence_reset(self):
        for options in [{},{'handoff':False}]:
            p=PeripheralPublisher(case(**options))
            p.update(packet(1,[],{},[record(10,0,[0,0,20,20])]))
            p.update(packet(2,[],{}));p.update(packet(3,[],{}))
            out=p.update(packet(4,[(4,np.array([0,0,20,20]),7,.9)],{7:0},[record(10,0,[0,0,20,20],score=.9)]))
            self.assertEqual(out[0][2],7);self.assertEqual(p.statistics['identity_handoffs'],0)
            value=packet(1,[(1,np.array([0,0,20,20]),1,.9)],{1:0});value['sequence']='B'
            new=p.update(value);self.assertEqual(new[0][2],1);self.assertEqual(p.aux_alias,{})

    def test_unknown_foreground_part_is_not_published_when_half_covered(self):
        p=PeripheralPublisher(case());main=(1,np.array([0,0,10,20]),1,.9)
        out=p.update(packet(1,[main],{1:0},[record(10,1,[0,0,20,20])],explained=[1,.5]))
        self.assertEqual(len(out),1)
        q=PeripheralPublisher(case(explain=False))
        out=q.update(packet(1,[main],{1:0},[record(10,1,[0,0,20,20])],explained=[1,.5]))
        self.assertEqual(len(out),2)

    def test_frame_bank_float64_ID_and_causal_order_roundtrip(self):
        builder=FrameBankBuilder();original=PeripheralPublisher(case());expected=[]
        for frame in range(1,5):
            rows=[(frame,np.array([.1234567890123+frame,0,20,20],np.float64),1,.912345678901)]
            value=packet(frame,rows,{1:0},[record(100,1,[100+frame,0,120+frame,20])])
            builder.append(value);expected.append(''.join(map(mot_line,original.update(value))))
        reader=SequentialFrameBank('A',builder.arrays());restored=PeripheralPublisher(case())
        with self.assertRaises(AssertionError):reader.next(2)
        for frame in range(1,5):
            value=reader.next(frame);actual=restored.update(value)
            self.assertEqual(''.join(map(mot_line,actual)),expected[frame-1])
            self.assertEqual(float(value['main_rows'][0][1][0]),.1234567890123+frame)
            with self.assertRaises(TypeError):value['main_proposals'][1]=9
            with self.assertRaises(ValueError):value['main_rows'][0][1][0]=0
        self.assertEqual(original.main_projection.hexdigest(),restored.main_projection.hexdigest())


if __name__=='__main__':unittest.main()
