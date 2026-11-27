import unittest
from types import SimpleNamespace
import numpy as np
from tools.mft_deformation_association import AssociationTeacher, measured_scale_cost, policies
from tools.mft_birth_only import BirthOnlyTeacher
from tools.diagnose_mft_birth_only import state_equal
from tools.native_pyramid_stream_tracks import mot_line


def obj(box, score=.8):
    return SimpleNamespace(tlbr=np.asarray(box, np.float64), score=score)


class AssociationTests(unittest.TestCase):
    def test_current_scale_handles_shape_change_and_retains_center_separation(self):
        tracks = [obj([-5, -5, 5, 5]), obj([-20, -5, 80, 5])]
        detections = [obj([-50, -5, 50, 5]), obj([25, -5, 35, 5])]
        c = measured_scale_cost(tracks, detections)
        self.assertEqual(c[0, 0], 0)
        self.assertEqual(c[1, 1], 0)
        self.assertGreater(c[0, 1], 0)
        self.assertGreater(c[1, 0], 0)
        np.testing.assert_array_equal(detections[0].tlbr, [-50, -5, 50, 5])
        self.assertEqual(measured_scale_cost([], detections).shape, (0, 2))
        self.assertEqual(measured_scale_cost(tracks, []).shape, (2, 0))

    def test_disabled_exact_original_rows_and_hidden_state(self):
        old = BirthOnlyTeacher(.1)
        new = AssociationTeacher(.1, policies()[0])
        for frame in range(1, 40):
            boxes = [[frame, 0, frame+20, 20, .91]]
            if frame % 7 != 0: boxes.append([100-frame, 40, 130-frame, 60, .76])
            d = np.asarray(boxes, np.float32)
            x = np.tile(np.array([[1., .5, .2]], np.float32), (len(d), 1))
            before = d.copy()
            a = old.update('A', frame, d, x, x)
            b = new.update('A', frame, d, x, x)
            self.assertEqual(''.join(map(mot_line, a)), ''.join(map(mot_line, b)))
            state_equal(old, new)
            np.testing.assert_array_equal(d, before)

    def test_score_mode_changes_only_its_own_core_and_keeps_original_score(self):
        reference = AssociationTeacher(.1, policies()[0])
        candidate = AssociationTeacher(.1, policies()[1])
        d = np.array([[0, 0, 20, 20, .85]], np.float32)
        x = np.array([[1., .5]], np.float32)
        a = reference.update('A', 1, d, x, x)
        b = candidate.update('A', 1, d, x, x)
        self.assertFalse(reference.core.args.mot20)
        self.assertTrue(candidate.core.args.mot20)
        self.assertEqual(a[0][3], b[0][3])
        self.assertEqual(b[0][3], float(d[0, 4]))
        state_equal(reference, candidate)


if __name__ == '__main__':
    unittest.main()
