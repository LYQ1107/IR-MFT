import unittest
import numpy as np
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_coverage_recall import CoverageRecallTeacher
from tools.native_pyramid_stream_tracks import mot_line
from tools.diagnose_mft_birth_only import state_equal


class RecallTests(unittest.TestCase):
    def test_disabled_recall_exact_frozen_birth_only_state(self):
        rgb=np.zeros((3,512),np.float32);rgb[:,0]=1
        old=BirthOnlyTeacher(.2);new=CoverageRecallTeacher(.2)
        for frame in range(1,30):
            d=np.array([[frame,0,frame+25,20,.9],[100+frame,0,125+frame,20,.65],
                [200+frame,0,225+frame,20,.45]],np.float32)
            if frame in [7,8,15]:d=d[:1]
            mask=(d[:,4]>.6)&(d[:,4]<.7);old.qualification=mask;new.qualification=mask.copy()
            a=old.update('A',frame,d,rgb[:len(d)],rgb[:len(d)])
            b=new.update('A',frame,d,rgb[:len(d)],rgb[:len(d)])
            self.assertEqual(''.join(map(mot_line,a)),''.join(map(mot_line,b)));state_equal(old,new)
        self.assertEqual(new.statistics['low_score_births'],0)

    def test_weak_birth_confirmation_keeps_original_score_and_existing_track(self):
        rgb=np.zeros((2,512),np.float32);rgb[:,0]=1
        old=BirthOnlyTeacher(.2);new=CoverageRecallTeacher(.2)
        for frame in range(1,9):
            d=np.array([[frame,0,frame+25,20,.9],[100+frame,0,125+frame,20,.45]],np.float32)
            before=d.copy();new.recall_qualification=np.array([False,frame in [3,4]])
            a=old.update('A',frame,d,rgb,rgb);b=new.update('A',frame,d,rgb,rgb)
            np.testing.assert_array_equal(d,before)
            self.assertEqual(mot_line(a[0]),mot_line(next(v for v in b if v[2]==1)))
            if frame>=4:
                weak=next(v for v in b if v[2]==2);self.assertEqual(float(weak[3]),float(d[1,4]))
        self.assertEqual(new.statistics['low_score_births'],1)
        self.assertEqual(new.statistics['weak_pending_confirmations'],1)

    def test_used_secondary_proposal_cannot_birth_and_reset_clears_state(self):
        rgb=np.zeros((1,512),np.float32);rgb[:,0]=1;new=CoverageRecallTeacher(.2)
        for frame in range(1,7):
            score=.9 if frame<3 else .45
            d=np.array([[frame,0,frame+25,20,score]],np.float32)
            new.recall_qualification=np.array([frame>=3]);rows=new.update('A',frame,d,rgb,rgb)
            self.assertEqual(len(rows),1);self.assertEqual(rows[0][2],1)
        self.assertEqual(new.statistics['low_score_births'],0)
        empty=np.empty((0,5),np.float32);r=np.empty((0,512),np.float32)
        self.assertEqual(new.update('B',1,empty,r,r),[])
        self.assertEqual(new.core.frame_id,1);self.assertEqual(new.statistics['extra_qualified_births'],0)


if __name__=='__main__':unittest.main()
