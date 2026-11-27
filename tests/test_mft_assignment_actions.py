import unittest
from types import SimpleNamespace
import numpy as np
from tools.mft_assignment_actions import (ActionTeacher, alternatives, force_actions,
    action_features, view_features, FEATURE_NAMES)
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.diagnose_mft_birth_only import state_equal
from tools.native_pyramid_stream_tracks import mot_line


def track():
    return SimpleNamespace(accepted=3,rgb=np.array([1.,0.],np.float32),ir=np.array([0.,1.],np.float32),
        anchor_rgb=np.array([1.,0.],np.float32),anchor_ir=np.array([0.,1.],np.float32),last_box=np.array([0,0,20,20]),
        memory_quality=.8,frame_id=1,start_frame=1,rgb_variance=.01,ir_variance=.1)


def detection(rgb,ir):
    return SimpleNamespace(score=.8,observation=dict(rgb=np.asarray(rgb,np.float32),ir=np.asarray(ir,np.float32),
        ir_available=True,quality=.7))


class ActionTests(unittest.TestCase):
    def test_all_three_actions_keep_cardinality_and_zero_is_literal(self):
        for base,n,m,kind in [(np.array([[.2,.25],[.25,.2]]),2,2,'closed_swap'),
                              (np.array([[.2,.25]]),1,2,'detection_replacement'),
                              (np.array([[.2],[.25]]),2,1,'track_replacement')]:
            pairs,actions=alternatives([track()for _ in range(n)],[None]*m,base,'primary')
            action=next(v for v in actions if v.kind==kind)
            result=force_actions(base,'primary',pairs,[action])
            from yolox.tracker import matching
            observed=matching.linear_assignment(result,thresh=.9)[0]
            self.assertEqual(len(observed),len(pairs));self.assertEqual(set(map(tuple,observed)),set(action.after))
            self.assertIs(force_actions(base,'primary',pairs,[]),base)
            self.assertTrue(np.all(result[np.array(result)>1]==2))

    def test_primary_and_secondary_geometry_gates_and_templates(self):
        base=np.array([[.4,.55]])
        self.assertEqual(len(alternatives([track()],[None,None],base,'secondary')[1]),0)
        t=track();t.accepted=2
        self.assertEqual(len(alternatives([t],[None,None],np.array([[.2,.25]]),'primary')[1]),0)

    def test_cues_symmetric_and_rgb_control_independent_of_ir(self):
        t=[track(),track()];d=[detection([1,0],[0,1]),detection([0,1],[1,0])]
        base=np.array([[.2,.25],[.25,.2]])
        pairs,acts=alternatives(t,d,base,'primary');a=acts[0]
        x=action_features(t,d,base,2,'primary',a);self.assertEqual(x.shape,(41,))
        from tools.mft_assignment_actions import Action
        reversed_action=Action(a.kind,tuple(reversed(a.before)),tuple(reversed(a.after)),a.regret)
        np.testing.assert_array_equal(x,action_features(t,d,base,2,'primary',reversed_action))
        d[0].observation['ir']=np.array([1.,0.],np.float32)
        y=action_features(t,d,base,2,'primary',a)
        np.testing.assert_array_equal(view_features(x,'RGB'),view_features(y,'RGB'))
        duplicate=view_features(x,'duplicatedRGB')
        self.assertEqual(duplicate[FEATURE_NAMES.index('IR_after')],duplicate[FEATURE_NAMES.index('RGB_after')])

    def test_actual_teacher_rows_states_and_locked_anchor(self):
        records=[];a=ActionTeacher(.1,lambda *v:records.append(v));b=CandidateTeacher(.1)
        anchored=None
        for f in range(1,30):
            d=np.array([[f,0,f+20,20,.9],[100-f,40,120-f,60,.85]],np.float32)
            x=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)if f<=3 else np.array([[.8,.6,0.],[.1,.99,0.]],np.float32)
            before=d.copy();old=b.update('A',f,d,x,x);new=a.update('A',f,d,x,x)
            self.assertEqual(''.join(map(mot_line,old)),''.join(map(mot_line,new)));state_equal(a,b)
            np.testing.assert_array_equal(d,before)
            if f==3:anchored=a.core.tracked_stracks[0].anchor_rgb.copy()
            if f>3:np.testing.assert_array_equal(a.core.tracked_stracks[0].anchor_rgb,anchored)
        self.assertTrue(records);self.assertGreater(a.core.tracked_stracks[0].rgb_variance,0)


if __name__=='__main__':unittest.main()
