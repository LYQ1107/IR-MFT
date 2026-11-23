import numpy as np

from tools.structural_correspondence import descriptor, match


def test_local_self_similarity_is_affine_polarity_invariant():
    image = np.random.default_rng(8).random((32, 40)).astype(np.float32)
    a, _ = descriptor(image)
    b, _ = descriptor(-1.7*image+.3)
    np.testing.assert_allclose(a, b, atol=2e-6)


def test_forward_offset_recovers_known_translation_without_wrap():
    image = np.random.default_rng(5).random((40, 48)).astype(np.float32)
    translated = np.pad(image, ((2, 0), (0, 3)), mode='reflect')[:40, 3:51]
    result = match(image, translated, 4)
    interior = np.s_[10:-10, 10:-10]
    expected = np.broadcast_to([2, -3], result['offset'][interior].shape)
    np.testing.assert_array_equal(result['offset'][interior], expected)
    assert result['inverse_consistent'][interior].all()


def test_constant_views_cannot_certify_a_correspondence():
    result = match(np.ones((24, 24), np.float32), np.ones((24, 24), np.float32)*3)
    assert not result['textured'].any()
    assert not result['inverse_consistent'].any()
