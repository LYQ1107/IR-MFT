"""Read-only comparison of per-video MOT identity names; never rewrite outputs."""
import hashlib
import itertools
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import sha256


def compare_original_tracks(original,replayed):
    """Require every non-ID field exact and one fixed bijection for the whole video."""
    original,replayed=Path(original),Path(replayed);forward={};reverse={};rows=0;offsets=set()
    original_sha,replayed_sha=sha256(original),sha256(replayed)
    with original.open() as left,replayed.open() as right:
        for number,(old,new) in enumerate(itertools.zip_longest(left,right),1):
            if old is None or new is None:raise ValueError(f'Original/replay row coverage differs at row{number}')
            a,b=old.rstrip('\n').split(','),new.rstrip('\n').split(',')
            if len(a)!=10 or len(b)!=10 or a[:1]+a[2:]!=b[:1]+b[2:]:
                raise ValueError(f'Original/replay non-ID MOT fields differ at row{number}')
            oi,ni=int(a[1]),int(b[1])
            if min(oi,ni)<1 or forward.get(oi,ni)!=ni or reverse.get(ni,oi)!=oi:
                raise ValueError(f'Whole-video identity mapping is not bijective at row{number}')
            forward[oi]=ni;reverse[ni]=oi;offsets.add(oi-ni);rows+=1
    if sha256(original)!=original_sha or sha256(replayed)!=replayed_sha:
        raise ValueError('Original/replayed tracks changed during comparison')
    mapping=sorted(forward.items())
    return dict(rows=rows,original_track=str(original),replayed_track=str(replayed),
        original_track_sha256=original_sha,replayed_track_sha256=replayed_sha,literal_files_exact=original_sha==replayed_sha,
        all_non_ID_fields_literal_exact=True,whole_video_ID_bijection=True,identity_count=len(mapping),
        original_minus_replayed_ID_offsets=sorted(offsets),
        identity_mapping_sha256=hashlib.sha256(json.dumps(mapping,separators=(',',':')).encode()).hexdigest(),
        original_or_replayed_rows_rewritten=False,GT_read=False,
        scope='Read-only video-wide identity-name comparison; separate official47 reproduction remains mandatory.')
