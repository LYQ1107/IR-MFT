import cv2
import numpy as np

from yolox.tracker.perceptual_features import (
    foreground, hog, letterbox_tensor_array, local_transport,
)
from yolox.tracker.perceptual_tracker import PerceptualTracker


def test_local_transport_recovers_known_translation_and_rejects_empty_frame():
    rng = np.random.default_rng(9)
    previous = np.zeros((120, 160), np.uint8)
    previous[25:85, 35:115] = rng.integers(40, 250, (60, 80), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 5], [0, 1, 3]]), (160, 120))
    boxes = np.array([[35, 25, 115, 85]], np.float32)
    result = local_transport(previous, current, boxes, previous.shape)
    np.testing.assert_allclose(result[0, :2], [5, 3], atol=.4)
    assert result[0, 2] > .7
    missing = local_transport(np.zeros_like(previous), np.zeros_like(previous), boxes, previous.shape)
    assert not missing.any()


def test_shape_contrast_is_polarity_independent_and_finite():
    gray = np.full((32, 64), 30, np.uint8)
    cv2.ellipse(gray, (32, 16), (19, 7), 15, 0, 360, 180, -1)
    first, quality = foreground(gray)
    inverse, inverse_quality = foreground(255 - gray)
    np.testing.assert_allclose(first, inverse, atol=1e-5)
    np.testing.assert_allclose(quality, inverse_quality, atol=1e-5)
    assert first[16, 32] > first[0, 0]
    assert np.isfinite(hog(gray)).all()


def test_empty_saliency_retains_finite_mean_padded_crop():
    image = np.full((50, 100, 3), 100, np.uint8)
    empty, quality = foreground(np.full((25, 50), 80, np.uint8))
    assert quality == 0
    crop = letterbox_tensor_array(image, [-20, 10, 80, 40], empty)
    assert crop.shape == (3, 128, 256)
    assert np.isfinite(crop).all()


def crossing_data():
    centers = [(30, 70), (45, 55), (60, 40), (75, 25)]
    boxes = np.array([[x-10, 20, x+10, 40] for frame in centers for x in frame], np.float32)
    appearance = np.tile(np.eye(2, dtype=np.float32), (4, 1))
    return dict(boxes=boxes, scores=np.full(8, .95), offsets=np.arange(0,9,2),
                embedding=appearance, ir_mask_embedding=appearance,
                rgb_mask_embedding=appearance, ir_hog=appearance,
                rgb_hog=appearance, color=appearance,
                flow_rgb=np.tile([[15,0,1,0],[-15,0,1,0]],(4,1)),
                flow_ir=np.tile([[15,0,1,0],[-15,0,1,0]],(4,1)))


def test_appearance_keeps_unique_identity_through_crossing():
    data = crossing_data()
    tracker = PerceptualTracker(dict(appearance_weight=.65, color_weight=0,
                                    appearance_scale=.2, maximum_center_distance=3))
    outputs = [tracker.update(f, data) for f in range(1,5)]
    assert all(len({r[2] for r in out})==2 for out in outputs)
    last = {row[2]:row[1][0]+row[1][2]/2 for row in outputs[-1]}
    assert last[1] > last[2]


def test_future_evidence_does_not_change_already_emitted_prefix():
    first = crossing_data()
    second = {k:v.copy() for k,v in first.items()}
    # At frame 2 only flow from source frame 1 may be used. Flow from
    # source frame 2 describes frame 3, and is deliberately corrupted.
    for key in ['flow_rgb','flow_ir']:
        second[key][2:] = 1000
    for key in ['embedding','ir_mask_embedding','rgb_mask_embedding','boxes']:
        second[key][4:] = -1000
    cfg = dict(flow='consensus',appearance_weight=.5,protected_memory=True,
               maximum_center_distance=3)
    a,b=PerceptualTracker(cfg),PerceptualTracker(cfg)
    for frame in [1,2]:
        out_a,out_b=a.update(frame,first),b.update(frame,second)
        for ra,rb in zip(out_a,out_b):
            np.testing.assert_array_equal(ra[1],rb[1])
            assert ra[2:]==rb[2:]
    assert a.statistics==b.statistics


def test_gap_output_requires_current_transport_without_mutating_observation():
    data = crossing_data()
    data={k:(v[:1].copy() if k!='offsets' else np.array([0,1,1])) for k,v in data.items()}
    cfg=dict(flow='robust_consensus',supported_gap_output=True,gap_requires_aux=True)
    tracker=PerceptualTracker(cfg)
    assert len(tracker.update(1,data))==1
    assert len(tracker.update(2,data))==1
    assert tracker.tracks[0].last_frame==1
    assert tracker.statistics['supported_gap_outputs']==1
    invalid={k:v.copy() for k,v in data.items()}
    invalid['flow_rgb'][:]=0;invalid['flow_ir'][:]=0
    other=PerceptualTracker(cfg)
    other.update(1,invalid)
    assert not other.update(2,invalid)


def test_adapter_starts_at_RGB_identity_without_sample_context():
    import torch
    from torch.nn import functional as F
    from yolox.tracker.evidence_adapter import EvidenceAdapter
    torch.manual_seed(3)
    raw=torch.randn(4,512);evidence=torch.randn(4,1170)
    model=EvidenceAdapter().eval()
    torch.testing.assert_close(model.unnormalized(raw,evidence),raw,rtol=0,atol=0)
    torch.testing.assert_close(model(raw,evidence),F.normalize(raw,dim=-1),rtol=0,atol=0)
    with torch.no_grad():model.net[-1].weight.normal_(std=.01)
    torch.testing.assert_close(model(raw,evidence)[0],model(raw[:1],evidence[:1])[0],rtol=1e-5,atol=1e-6)


def test_RGB_metric_control_cannot_read_auxiliary_quality_or_views():
    from yolox.tracker.evidence_adapter import pack_evidence
    rng=np.random.default_rng(4)
    data={k:rng.normal(size=(2,dim)).astype(np.float32) for k,dim in
          [('embedding',512),('ir_mask_embedding',512),('rgb_mask_embedding',512),
           ('ir_hog',144),('rgb_hog',144),('quality',2)]}
    first=pack_evidence(data,'rgb')
    data['ir_hog'][:]=1000;data['ir_mask_embedding'][:]=-1000;data['quality'][:,1]=1e8
    second=pack_evidence(data,'rgb')
    np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])


def test_RGB_backend_runs_without_any_auxiliary_array():
    data=crossing_data()
    for key in ['ir_mask_embedding','ir_hog','flow_ir']:del data[key]
    tracker=PerceptualTracker(dict(appearance_view='rgb_mask',shape='rgb',flow='rgb',
                                   protected_memory=True))
    assert len(tracker.update(1,data))==2
    assert len(tracker.update(2,data))==2


def test_safe_byte_zero_cue_preserves_original_lifecycle_and_output():
    from types import SimpleNamespace
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker
    boxes=[];scores=[];offsets=[0]
    for frame in range(1,66):
        for identity in range(3):
            if identity==1 and 15<=frame<=46:continue
            if identity==2 and 20<=frame<=23:continue
            x=20+identity*60+.3*frame
            boxes.append([x,20,x+25,40]);scores.append(.4 if frame%7==0 else .95)
        offsets.append(len(boxes))
    n=len(boxes);features=np.tile(np.array([[1.,0.]],np.float32),(n,1))
    data=dict(boxes=np.array(boxes,np.float32),scores=np.array(scores,np.float32),offsets=np.array(offsets),
              embedding=features,rgb_hog=features,color=features)
    BaseTrack._count=0
    original=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
    expected=[]
    for f in range(1,66):
        a,b=data['offsets'][f-1:f+1];d=np.column_stack([data['boxes'][a:b],data['scores'][a:b]])
        expected.append([(t.track_id,t.tlwh.copy(),t.score) for t in original.update(d.copy(),(1,1),(1,1))])
    BaseTrack._count=0
    revised=EvidenceByteTracker(dict(appearance_view='raw',shape='rgb'))
    assert revised.inner.max_time_lost==25
    for f in range(1,66):
        actual=revised.update(f,data);reference=expected[f-1]
        assert len(actual)==len(reference)
        for row,ref in zip(actual,reference):
            assert row[2]==ref[0] and row[3]==ref[2]
            np.testing.assert_array_equal(row[1],ref[1])


def test_evidence_ranking_never_opens_a_rejected_IoU_edge():
    from types import SimpleNamespace
    from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker
    owner=EvidenceByteTracker(dict(appearance_penalty=.5))
    tracks=[SimpleNamespace(appearance=np.array([1.,0.])),SimpleNamespace(appearance=np.array([0.,1.]))]
    observations=[SimpleNamespace(evidence=dict(index=i,appearance=v)) for i,v in enumerate([np.array([0.,1.]),np.array([1.,0.])])]
    base=np.array([[.55,1.],[.54,.56]])
    cost=owner.rank_cost(tracks,observations,base,1,'primary')
    assert cost[0,1]==1.
    assert not np.any((base>.9)&(cost<=.9))
