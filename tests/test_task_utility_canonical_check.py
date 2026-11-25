"""Artificial MOT rows; no official accuracy or actual model qualification."""
import pytest

from tools.task_utility_canonical_check import compare_original_tracks


def line(frame,identity,x='1.000'):
    return f'{frame},{identity},{x},2.000,20.000,30.000,0.900000,-1,-1,-1\n'


def test_video_wide_ID_bijection_accepts_offset_without_modifying_either_file(tmp_path):
    old,new=tmp_path/'original.txt',tmp_path/'replay.txt'
    a=line(1,23)+line(1,24)+line(2,23)+line(2,24);b=line(1,1)+line(1,2)+line(2,1)+line(2,2)
    old.write_text(a);new.write_text(b);value=compare_original_tracks(old,new)
    assert value['rows']==4 and value['identity_count']==2 and not value['literal_files_exact']
    assert value['original_minus_replayed_ID_offsets']==[22] and not value['original_or_replayed_rows_rewritten']
    assert old.read_text()==a and new.read_text()==b


@pytest.mark.parametrize('new_rows',[
    line(1,1)+line(2,2), # One original identity split in time.
    line(1,1)+line(2,1,x='1.001'),
    line(1,1),
    line(1,0)+line(2,0),
])
def test_renaming_check_rejects_association_coordinate_coverage_and_invalid_ID_changes(tmp_path,new_rows):
    old,new=tmp_path/'original.txt',tmp_path/'replay.txt'
    old.write_text(line(1,23)+line(2,23));new.write_text(new_rows)
    with pytest.raises(ValueError):compare_original_tracks(old,new)


def test_different_original_IDs_cannot_merge_into_one_replayed_ID(tmp_path):
    old,new=tmp_path/'original.txt',tmp_path/'replay.txt'
    old.write_text(line(1,23)+line(2,24));new.write_text(line(1,1)+line(2,1))
    with pytest.raises(ValueError):compare_original_tracks(old,new)
