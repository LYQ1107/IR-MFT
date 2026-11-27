import hashlib
import unittest

import numpy as np

from tests.test_mft_protected_peripheral import packet, record, case
from tools.mft_trusted_shadow_bridge import TrustedShadowBridge
from tools.mft_protected_peripheral import PeripheralPublisher
from tools.native_pyramid_stream_tracks import mot_line


class BridgeTests(unittest.TestCase):
    def test_trust_born_identity_survives_current_low_confidence_measurement(self):
        p = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=3))
        expected = hashlib.sha256()
        for frame in range(1, 4):
            rows = [(frame, np.array([0, 0, 20, 20]), 7, .9)]
            expected.update(''.join(map(mot_line, rows)).encode())
            out = p.update(packet(frame, rows, {7: 0}, [record(10, 0, [0, 0, 20, 20], .9)], gate=0))
            self.assertEqual([r[2] for r in out], [7])
        out = p.update(packet(4, [], {}, [record(10, 0, [1, 0, 21, 20], .2)], gate=0))
        self.assertEqual([r[2] for r in out], [7])
        self.assertEqual(out[0][3], .2)
        self.assertEqual(p.main_projection.hexdigest(), expected.hexdigest())
        self.assertEqual(p.statistics['trusted_bridge_rows'], 1)

    def test_expiry_reconfirmation_disagreement_and_no_id_merge(self):
        p = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=1))
        p.update(packet(1, [(1, np.array([0, 0, 20, 20]), 7, .9)], {7: 0}, [record(10, 0, [0, 0, 20, 20], .9)], gate=0))
        for frame in range(2, 7):
            self.assertEqual(p.update(packet(frame, [], {}, [record(10, 0, [0, 0, 20, 20], .2)], gate=0))[0][2], 7)
        self.assertEqual(p.update(packet(7, [], {}, [record(10, 0, [0, 0, 20, 20], .2)], gate=0)), [])
        out = p.update(packet(8, [(8, np.array([0, 0, 20, 20]), 9, .9)], {9: 0}, [record(10, 0, [0, 0, 20, 20], .9)], gate=0))
        self.assertEqual(out[0][2], 9)
        self.assertEqual(p.update(packet(9, [], {}, [record(10, 0, [0, 0, 20, 20], .2)], gate=0))[0][2], 9)
        self.assertEqual(p.main_alias, {7: 7, 9: 9})

    def test_consecutive_agreement_and_spatial_exclusion(self):
        p = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=3))
        for frame in range(1, 4):
            p.update(packet(frame, [(frame, np.array([0, 0, 20, 20]), 7, .9)], {7: 0},
                            [record(10 if frame != 2 else 11, 0, [0, 0, 20, 20], .9)], gate=0))
        self.assertEqual(p.update(packet(4, [], {}, [record(10, 0, [0, 0, 20, 20], .2)], gate=0)), [])
        q = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=1))
        q.update(packet(1, [(1, np.array([0, 0, 20, 20]), 7, .9)], {7: 0}, [record(10, 0, [0, 0, 20, 20], .9)], gate=0))
        out = q.update(packet(2, [(2, np.array([0, 0, 20, 20]), 8, .9)], {8: 1},
                              [record(10, 0, [0, 0, 20, 20], .2)], explained=[.5, 1], gate=0))
        self.assertEqual([r[2] for r in out], [8])

    def test_disabled_is_literal_parent_and_prefix_is_causal(self):
        a = TrustedShadowBridge(case(bridge_gap=0, bridge_agreement=3))
        b = PeripheralPublisher(case())
        for frame in range(1, 20):
            rows = [(frame, np.array([0, 0, 20, 20]), 1, .9)] if frame % 3 else []
            value = packet(frame, rows, {1: 0} if rows else {}, [record(10, 0, [0, 0, 20, 20], .9 if rows else .4)])
            self.assertEqual(''.join(map(mot_line, a.update(value))), ''.join(map(mot_line, b.update(value))))
        original = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=1))
        prefixes = []
        for frame in range(1, 5):
            rows = [(frame, np.array([0, 0, 20, 20]), 1, .9)] if frame == 1 else []
            prefixes.append(''.join(map(mot_line, original.update(packet(frame, rows, {1: 0} if rows else {}, [record(10, 0, [0, 0, 20, 20], .9 if rows else .2)], gate=0)))))
        other = TrustedShadowBridge(case(bridge_gap=5, bridge_agreement=1))
        self.assertEqual(prefixes[0], ''.join(map(mot_line, other.update(packet(1, [(1, np.array([0, 0, 20, 20]), 1, .9)], {1: 0}, [record(10, 0, [0, 0, 20, 20], .9)], gate=0)))))
        reset = packet(1, [], {}, gate=0); reset['sequence'] = 'B'
        self.assertEqual(original.update(reset), [])
        self.assertEqual(original.bindings, {})


if __name__ == '__main__':
    unittest.main()
