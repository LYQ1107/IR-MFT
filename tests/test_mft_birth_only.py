import inspect
import unittest
import numpy as np
from tools.mft_birth_only import BirthOnlyByteTracker,BirthOnlyTeacher,birth_mask
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_proposal_quality import ForegroundView,QualityHead
from yolox.tracker.byte_tracker import BYTETracker
from tools.native_pyramid_stream_tracks import mot_line


class BirthOnlyTests(unittest.TestCase):
    def test_port_only_changes_birth_check(self):
        old=inspect.getsource(BYTETracker.update)
        new=inspect.getsource(BirthOnlyByteTracker.update)
        expected=old.replace('if track.score < self.det_thresh:', 'if not self.birth_allowed(track):')
        self.assertNotEqual(old,expected)
        self.assertEqual(new,expected)

    def test_extra_birth_keeps_scores_and_existing_association(self):
        rgb=np.zeros((2,512),np.float32);rgb[:,0]=1
        reference=CandidateTeacher(.2);port=BirthOnlyTeacher(.2)
        for frame in range(1,6):
            d=np.array([[10+frame,10,40+frame,30,.9],
                [100+frame,10,130+frame,30,.65]],np.float32)
            before=d.copy();raw=reference.update('A',frame,d,rgb,rgb)
            port.qualification=np.array([False,True]);rows=port.update('A',frame,d,rgb,rgb)
            np.testing.assert_array_equal(d,before)
            existing=next(r for r in rows if r[2]==1)
            self.assertEqual(mot_line(existing),mot_line(raw[0]))
            if frame>=2:
                weak=next(r for r in rows if r[2]==2)
                self.assertEqual(float(weak[3]),float(d[1,4]))
            self.assertEqual(port.core.args.track_thresh,.6)
        self.assertEqual(port.statistics['extra_qualified_births'],1)

    def test_zero_gate_exact_and_missing_quality_ir_exact_rgb(self):
        rng=np.random.default_rng(0);rgb=rng.normal(size=(2,512)).astype(np.float32)
        reference=CandidateTeacher(.2);port=BirthOnlyTeacher(.2)
        for frame in range(1,9):
            d=np.array([[10+frame,10,40+frame,30,.9],
                [100+frame,10,130+frame,30,.65]],np.float32)
            a=reference.update('A',frame,d,rgb,rgb);b=port.update('A',frame,d,rgb,rgb)
            self.assertEqual(''.join(map(mot_line,a)),''.join(map(mot_line,b)))
        direction=np.zeros(512);direction[0]=1
        views={k:ForegroundView(np.zeros(512),direction)for k in ['RGB','IR']}
        heads={k:QualityHead(np.zeros(11),np.ones(11),k)for k in ['RGB','paired']}
        heads['RGB'].coefficients[0]=1
        heads['paired'].coefficients[0]=-1
        thresholds={k:{'q99':{'confirmed':.55}}for k in heads}
        c=dict(kind='paired',confirmation=3,quantile='q99',intervention='missing')
        a=birth_mask(c,d,rgb,rgb,views,heads,thresholds,np.array([3,3]))
        b=birth_mask(dict(kind='RGB',confirmation=3,quantile='q99'),d,rgb,None,views,heads,thresholds,np.array([3,3]))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(a[1]);self.assertFalse(a[0])


if __name__=='__main__':unittest.main()
