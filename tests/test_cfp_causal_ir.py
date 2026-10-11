"""Scientific causality and recovered-source integrity tests."""
from pathlib import Path

import numpy as np
import pytest

from cfp.pseudo_ir import CausalPseudoIR


SCRIPTS = Path('/data3/liuyeqiang/IR-MFT_assets/scripts')


@pytest.mark.skipif(not SCRIPTS.exists(), reason='External recovered scripts required')
def test_future_perturbation_cannot_change_prefix_and_sequence_reset():
    rng = np.random.default_rng(19)
    frames = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(6)]
    altered = frames[:3] + [np.zeros_like(frames[0]) for _ in range(3)]
    a, b = CausalPseudoIR(SCRIPTS), CausalPseudoIR(SCRIPTS)
    outputs_a = [a(frame) for frame in frames]
    outputs_b = [b(frame) for frame in altered]
    assert all(np.array_equal(x, y) for x, y in zip(outputs_a[:3], outputs_b[:3]))
    fresh = CausalPseudoIR(SCRIPTS)
    assert np.array_equal(fresh(frames[0]), outputs_a[0])
    assert a.frames_seen == 6
    with pytest.raises(ValueError, match='dimensions'):
        a(np.zeros((63, 96, 3), np.uint8))


@pytest.mark.skipif(not SCRIPTS.exists(), reason='External recovered scripts required')
def test_background_write_occurs_after_response_and_input_is_immutable():
    image = np.zeros((64, 96, 3), np.uint8)
    changed = image.copy()
    changed[20:45, 30:65] = 220
    stream = CausalPseudoIR(SCRIPTS, alpha=.2)
    stream(image)
    old = stream.background.copy()
    expected_gray = stream.source.rgb_to_pseudo_gray_temporal(changed, old.astype(np.uint8))
    import cv2
    expected = cv2.applyColorMap(expected_gray, cv2.COLORMAP_INFERNO)
    before = changed.copy()
    actual = stream(changed)
    assert np.array_equal(actual, expected)
    assert np.array_equal(changed, before)
    assert np.allclose(stream.background, .8 * old + .2 * changed)


def test_changed_external_generator_is_rejected(tmp_path):
    (tmp_path / 'trans-sn0612.py').write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match='pinned original'):
        CausalPseudoIR(tmp_path)


@pytest.mark.skipif(not SCRIPTS.exists(), reason='External recovered scripts required')
@pytest.mark.parametrize('mode', ['local', 'unified_v2'])
def test_stateless_adapters_exactly_match_original_process_image(tmp_path, mode):
    pytest.importorskip('pywt')
    import cv2
    frame = np.random.default_rng(21).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), frame)
    expected_dir = tmp_path / 'original'
    expected_dir.mkdir()
    adapter = CausalPseudoIR(SCRIPTS, mode)
    adapter.source.process_image(str(path), str(expected_dir))
    expected = cv2.imread(str(expected_dir / path.name))
    assert np.array_equal(adapter(frame), expected)


@pytest.mark.skipif(not SCRIPTS.exists(), reason='External recovered scripts required')
def test_temporal_optimization_exact_pixel_parity():
    original = CausalPseudoIR(SCRIPTS, optimize=False)
    optimized = CausalPseudoIR(SCRIPTS, optimize=True)
    rng = np.random.default_rng(25)
    for _ in range(5):
        frame = rng.integers(0, 256, (128, 192, 3), dtype=np.uint8)
        assert np.array_equal(original(frame), optimized(frame))
    # Many separate components exercise the lookup replacement and area bounds.
    mask = np.zeros((540, 960), np.uint8)
    for y in range(10, 530, 35):
        for x in range(10, 950, 35):
            mask[y:y + 5, x:x + 5] = 255
    assert np.array_equal(original.source.filter_components(mask), optimized.source.filter_components(mask))


@pytest.mark.skipif(not SCRIPTS.exists(), reason='External recovered scripts required')
def test_prefix_only_state_can_resume_without_changing_any_output():
    frames = [np.random.default_rng(i).integers(0, 256, (64, 96, 3), dtype=np.uint8) for i in range(7)]
    sequential = CausalPseudoIR(SCRIPTS)
    expected = [sequential(frame) for frame in frames]
    prefix = CausalPseudoIR(SCRIPTS)
    for frame in frames[:3]:
        prefix.advance_background(frame)
    resumed = CausalPseudoIR(SCRIPTS)
    resumed.background = prefix.background.copy()
    resumed.frames_seen = 3
    assert all(np.array_equal(resumed(frame), target) for frame, target in zip(frames[3:], expected[3:]))
