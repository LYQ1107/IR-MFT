"""Causal short-lived continuation of trusted identities by measured shadows.

Exact repeated agreement on the current proposal binds a shadow to a trusted
identity. A missing trusted output can then use that shadow's current measured
box briefly. This never merges two trusted identities, changes their states,
interpolates boxes, or rewrites an earlier output.
"""
import numpy as np

from tools.mft_protected_peripheral import PeripheralPublisher, covered_fraction


class TrustedShadowBridge(PeripheralPublisher):
    def reset(self, sequence):
        super().reset(sequence)
        self.agreements = {}
        self.bindings = {}
        self.main_shadow = {}
        self.statistics.update(trusted_bridge_rows=0, trusted_bridge_bindings=0,
                               trusted_bridge_spatial_rejections=0,
                               trusted_bridge_duplicate_rejections=0)

    def update(self, value):
        if value['sequence'] != self.sequence:
            assert value['frame'] == 1
            self.reset(value['sequence'])
        frame = value['frame']
        assert frame == self.frame + 1
        maximum_gap = self.case.get('bridge_gap', 0)
        if maximum_gap == 0:
            return super().update(value)
        assert maximum_gap in (5, 25)
        agreement_required = self.case.get('bridge_agreement', 3)
        assert agreement_required in (1, 3)
        current_main = set(value['main_proposals'])
        by_proposal = {r['proposal_index']: r for r in value['shadow_records']}

        # Current disagreement revokes an old binding before any continuation.
        matches = []
        for base, proposal in value['main_proposals'].items():
            record = by_proposal.get(proposal)
            if record is None:
                continue
            sid = record['shadow_id']
            old_sid = self.main_shadow.get(base)
            if old_sid is not None and old_sid != sid:
                old_binding = self.bindings.get(old_sid)
                if old_binding is not None and old_binding['base'] == base:
                    self.bindings.pop(old_sid)
                old_agreement = self.agreements.get(old_sid)
                if old_agreement is not None and old_agreement['base'] == base:
                    self.agreements.pop(old_sid)
            old = self.bindings.get(sid)
            if old is not None and old['base'] != base:
                self.bindings.pop(sid)
            matches.append((base, sid))

        eligible = []
        for record in value['shadow_records']:
            sid = record['shadow_id']
            binding = self.bindings.get(sid)
            if binding is None or binding['base'] in current_main:
                continue
            if not 1 <= frame - binding['last'] <= maximum_gap:
                continue
            if record['proposal_index'] in value['main_assigned']:
                continue
            if not record['score'] > .1 or np.prod(record['tlwh'][2:]) <= 100:
                continue
            if value['explained'][record['proposal_index']] >= .5:
                self.statistics['trusted_bridge_spatial_rejections'] += 1
                continue
            eligible.append(record)

        # Reserve those proposals for the bound identity, before ordinary
        # peripheral publication allocates a new identity for the same box.
        reserved = {r['shadow_id'] for r in eligible}
        filtered = dict(value)
        filtered['shadow_records'] = tuple(r for r in value['shadow_records']
                                           if r['shadow_id'] not in reserved)
        output = super().update(filtered)
        used = {row[2] for row in output}
        accepted_boxes = []
        for record in filtered['shadow_records']:
            sid = record['shadow_id']
            if self.last_aux_publication.get(sid) == frame:
                accepted_boxes.append(record['box_xyxy'])
        for record in sorted(eligible, key=lambda r: (-self.bindings[r['shadow_id']]['last'],
                                                     r['shadow_id'])):
            binding = self.bindings[record['shadow_id']]
            public = self.main_alias[binding['base']]
            assert self.owner[public] == binding['base']
            if public in used:
                self.statistics['trusted_bridge_duplicate_rejections'] += 1
                continue
            if accepted_boxes and covered_fraction(record['box_xyxy'][None], accepted_boxes)[0] >= .5:
                self.statistics['trusted_bridge_spatial_rejections'] += 1
                continue
            output.append((frame, record['tlwh'].copy(), public, record['score']))
            used.add(public)
            accepted_boxes.append(record['box_xyxy'])
            self.statistics['trusted_bridge_rows'] += 1

        # Bind only after the ordinary publisher has fixed every trusted alias.
        for base, sid in matches:
            self.main_shadow[base] = sid
            previous = self.agreements.get(sid)
            count = previous['count'] + 1 if (previous is not None
                    and previous['base'] == base and previous['last'] == frame - 1) else 1
            self.agreements[sid] = dict(base=base, count=count, last=frame)
            public = self.main_alias[base]
            auxiliary = self.aux_alias.get(sid)
            if count >= agreement_required and (auxiliary is None or auxiliary == public):
                if sid not in self.bindings:
                    self.statistics['trusted_bridge_bindings'] += 1
                self.bindings[sid] = dict(base=base, last=frame)
        assert len(output) == len({row[2] for row in output})
        return output


def policies():
    base = dict(kind='paired', need_mode='original', threshold=.25)
    return [dict(base, id='SB_reference', bridge_gap=0, bridge_agreement=3),
            dict(base, id='SB_c3_gap5', bridge_gap=5, bridge_agreement=3),
            dict(base, id='SB_c3_gap25', bridge_gap=25, bridge_agreement=3),
            dict(base, id='SB_c1_gap25', bridge_gap=25, bridge_agreement=1)]
