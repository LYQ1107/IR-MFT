import numpy as np

from cfp.crop_retrieval import embedding_crop_retrieval


def test_missing_view_is_explicit_empty_and_gallery_is_strictly_past():
    records = [dict(view='gt', frame=1, identity=1), dict(view='gt', frame=2, identity=1)]
    result = embedding_crop_retrieval(np.array([[1., 0.], [1., 0.]], np.float32), records,
        np.ones(2, bool), gaps=(1,))
    assert result['gt']['1']['missing_target'] == 1 and result['gt']['1']['supported_queries'] == 1
    assert result['detector']['1']['queries'] == 0 and result['detector']['1']['rank1_percent'] is None


def test_unavailable_gallery_feature_cannot_beat_true_past_identity():
    records = [dict(view='detector', frame=1, identity=-1), dict(view='detector', frame=1, identity=1),
               dict(view='detector', frame=2, identity=1)]
    embedding = np.array([[1., 0.], [.8, .6], [1., 0.]], np.float32)
    result = embedding_crop_retrieval(embedding, records, np.array([False, True, True]), gaps=(1,))
    assert result['detector']['1']['rank1_percent'] == 100 and result['detector']['1']['unknown_selected'] == 0
    # Do not mutate an immutable caller feature bank.
    np.testing.assert_array_equal(embedding[0], np.array([1., 0.]))
