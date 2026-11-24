import pytest

from tools.benchmark_fusion_factorial import track_prefix


def test_prefix_preserves_original_bytes_and_stops_at_future_frame(tmp_path):
    path=tmp_path/'tracks.txt'
    prefix=b'1,4,1.000,2.000,3.000,4.000,0.800000,-1,-1,-1\n2,4,2.000,2.000,3.000,4.000,0.900000,-1,-1,-1\n'
    path.write_bytes(prefix+b'33,4,99,99,3,4,0.8,-1,-1,-1\n')
    assert track_prefix(path,32)==prefix


def test_empty_early_frames_do_not_justify_reindexing_source_tracks(tmp_path):
    path=tmp_path/'tracks.txt';path.write_bytes(b'8,9,0,0,20,20,0.8,-1,-1,-1\n')
    assert track_prefix(path,7)==b''
    assert track_prefix(path,8).startswith(b'8,9,')


@pytest.mark.parametrize('raw',[b'0,1,0,0,20,20,0.8,-1,-1,-1\n',
                               b'2,1,0,0,20,20,0.8,-1,-1,-1\n1,1,0,0,20,20,0.8,-1,-1,-1\n'])
def test_invalid_frame_order_cannot_support_a_prefix_identity_claim(tmp_path,raw):
    path=tmp_path/'tracks.txt';path.write_bytes(raw)
    with pytest.raises(ValueError):track_prefix(path,32)
