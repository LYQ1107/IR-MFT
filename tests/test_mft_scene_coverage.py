import unittest
import numpy as np
from tools.mft_scene_coverage import coverage_features,Dev_analysis_labels,uncovered_target
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_proposal_quality import ForegroundView,QualityHead


class CoverageTests(unittest.TestCase):
    def test_causal_eligibility_and_literal_missing_fallback(self):
        d=np.array([[0,0,20,20,.9],[50,0,70,20,.65],[100,0,120,20,.65]],np.float32)
        r=np.zeros((3,512),np.float32);r[:,0]=1
        v={k:ForegroundView(np.zeros(512),r[0])for k in ['RGB','IR']}
        h={k:QualityHead(np.zeros(11),np.ones(11),k)for k in ['RGB','paired','duplicatedRGB']}
        h['paired'].coefficients[-1]=10
        before=[x.copy()for x in [d,r]]
        chosen,x=coverage_features(d,r,None,v,h,np.array([3,3,2]),[1,2],[0])
        np.testing.assert_array_equal(chosen,[1]);self.assertEqual(x['count']['deficit'],.5)
        self.assertEqual(x['paired'],x['RGB']);self.assertEqual(x['IR_missing'],x['RGB'])
        for a,b in zip([d,r],before):np.testing.assert_array_equal(a,b)
        with self.assertRaises(AssertionError):coverage_features(d,r,r,v,h,np.ones(3),[1],[1])

    def test_dev_adapter_preserves_unknown_and_forbids_fitting_roles(self):
        d=np.array([[0,0,20,20,.65],[50,0,70,20,.65],[100,0,120,20,.65]],np.float32)
        gt=np.array([[1,1,0,0,20,20,1,1],[1,2,50,0,20,20,0,1]],float)
        y=Dev_analysis_labels(d,gt,frame=1,split='dev',purpose='read_only_coverage_diagnostic')
        np.testing.assert_array_equal(y,[1,-1,0])
        np.testing.assert_array_equal(y,current_detection_labels(d,gt,frame=1,split='train'))
        with self.assertRaises(AssertionError):current_detection_labels(d,gt,frame=1,split='dev')
        with self.assertRaises(AssertionError):Dev_analysis_labels(d,gt,frame=1,split='test',purpose='read_only_coverage_diagnostic')
        with self.assertRaises(AssertionError):Dev_analysis_labels(d,gt,frame=1,split='dev',purpose='fit')
        a=uncovered_target(d,y,gt,[],np.arange(3));self.assertEqual(a['uncovered_known_positive'],1)
        self.assertEqual(a['unknown'],1);self.assertEqual(a['known_background'],1)
        b=uncovered_target(d,y,gt,[[0,0,20,20]],np.arange(3));self.assertEqual(b['uncovered_known_positive'],0)


if __name__=='__main__':unittest.main()
