import cv2
import numpy as np
import pytest

from yolox.tracker.foveated_causal_attention import CausalAttention


def inputs(seed, count=6):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0,256,(90,160,3),dtype=np.uint8),
             rng.integers(0,256,(90,160),dtype=np.uint8)) for _ in range(count)]


def test_future_frames_cannot_change_emitted_attention():
    source = inputs(1)
    changed = source[:3] + inputs(2,3)
    outputs = []
    for sequence in [source,changed]:
        state = CausalAttention()
        outputs.append([state.update(i+1,rgb,ir,[],[]) for i,(rgb,ir) in enumerate(sequence)])
    for frame in range(3):
        for cue in outputs[0][frame]:
            np.testing.assert_array_equal(outputs[0][frame][cue],outputs[1][frame][cue])


def test_missing_IR_has_exact_complete_RGB_fallback():
    state = CausalAttention()
    for frame,(rgb,ir) in enumerate(inputs(3),1):
        result = state.update(frame,rgb,np.zeros_like(ir),[],[])
        for cue in ['ir_self','ir_physical','ir_consensus']:
            np.testing.assert_array_equal(result[cue],result['rgb'])


def test_reset_restores_first_frame_and_rejects_timestamp_reuse():
    rgb,ir = inputs(4,1)[0]
    state = CausalAttention()
    first = state.update(1,rgb,ir,[],[])
    with pytest.raises(ValueError):
        state.update(1,rgb,ir,[],[])
    state.reset()
    reset = state.update(1,rgb,ir,[],[])
    for cue in first:
        np.testing.assert_array_equal(reset[cue],first[cue])
