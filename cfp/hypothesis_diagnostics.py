"""Read-only causal public-ID/birth audit for isolated trajectory hypotheses."""


def birth_key(track):
    proposal = getattr(track, 'initial_proposal_index', None)
    return (int(track.start_frame), int(getattr(track, 'initial_identity_frame', 0)),
            None if proposal is None else int(proposal))


class HypothesisBirthAudit:
    def __init__(self):
        self.first_emitted = {}
        self.reused_births = set()
        self.counts = dict(emitted_reused_id_frames=0, distinct_reused_births=0,
                          live_hypothesis_birth_conflict_frames=0, counter_divergence_frames=0,
                          maximum_counter_gap=0)
        self.examples = []

    def observe(self, frame, hypotheses, output):
        counters = [h.next_identity for h in hypotheses]
        gap = max(counters) - min(counters)
        self.counts['maximum_counter_gap'] = max(self.counts['maximum_counter_gap'], gap)
        self.counts['counter_divergence_frames'] += int(gap > 0)
        names = {}
        for hypothesis in hypotheses:
            for track in hypothesis.state.tracked_stracks + hypothesis.state.lost_stracks:
                names.setdefault(int(track.track_id), set()).add(birth_key(track))
        self.counts['live_hypothesis_birth_conflict_frames'] += int(any(len(v) > 1 for v in names.values()))
        for track in output:
            identity, key = int(track.track_id), birth_key(track)
            old = self.first_emitted.setdefault(identity, key)
            if old != key:
                self.counts['emitted_reused_id_frames'] += 1
                event = (identity, key)
                if event not in self.reused_births:
                    self.reused_births.add(event)
                    if len(self.examples) < 10:
                        self.examples.append(dict(frame=int(frame), identity=identity,
                                                  first_birth=old, selected_birth=key))
        self.counts['distinct_reused_births'] = len(self.reused_births)

    def report(self):
        return dict(counts=dict(self.counts), examples=list(self.examples),
                    scope='Observed public-ID reuse across different birth observations; no GT labels, '
                          'future input, output correction or physical identity assertion.')
