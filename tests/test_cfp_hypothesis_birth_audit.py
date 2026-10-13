from types import SimpleNamespace

from cfp.hypothesis_diagnostics import HypothesisBirthAudit


def test_birth_reuse_event_is_deduplicated_and_read_only():
    first = SimpleNamespace(track_id=1, start_frame=1, initial_identity_frame=1, initial_proposal_index=2)
    later = SimpleNamespace(track_id=1, start_frame=3, initial_identity_frame=3, initial_proposal_index=9)
    a = SimpleNamespace(next_identity=1, state=SimpleNamespace(tracked_stracks=[first], lost_stracks=[]))
    b = SimpleNamespace(next_identity=2, state=SimpleNamespace(tracked_stracks=[later], lost_stracks=[]))
    audit = HypothesisBirthAudit(); audit.observe(1, [a], [first]); audit.observe(3, [a, b], [later]); audit.observe(4, [a, b], [later])
    report = audit.report()
    assert report['counts']['distinct_reused_births'] == 1
    assert report['counts']['emitted_reused_id_frames'] == 2
    assert report['counts']['counter_divergence_frames'] == 2
    assert first.start_frame == 1 and later.start_frame == 3
