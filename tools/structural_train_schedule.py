#!/usr/bin/env python3
"""Prepare broader Train-frame addresses without allocating images or FPNs."""
import configparser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump, sha256


def make_schedule(lengths, initial, passes=24, samples=192, seed=0):
    """Same first pass, then a fixed cyclic deck; distinct frames within a pass."""
    if seed != 0 or passes < 1 or samples < 1 or set(lengths) != set(initial) or not lengths:
        raise ValueError('Use seed0 and matched nonempty Train video maps')
    result = {}
    for index, sequence in enumerate(sorted(lengths)):
        length = int(lengths[sequence])
        first = np.asarray(initial[sequence], dtype=np.int32)
        if (length < samples or len(first) != samples or len(np.unique(first)) != samples or
                first.min() < 1 or first.max() > length):
            raise ValueError('Each first pass must contain valid unique existing cache addresses')
        remaining = np.setdiff1d(np.arange(1,length+1,dtype=np.int32),first)
        # This is a single fixed seed0 study, not additional model seed trials.
        rng = np.random.default_rng(20261123+index)
        deck = np.r_[first,rng.permutation(remaining)]
        result[sequence] = deck[np.arange(passes*samples).reshape(passes,samples) % length]
        assert np.array_equal(result[sequence][0],first)
        assert all(len(np.unique(row)) == samples for row in result[sequence])
        assert len(np.unique(result[sequence])) == min(length,passes*samples)
    return result


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol_path = ROOT/plan['protocol']
    protocol = json.loads(protocol_path.read_text())
    work = Path(plan['work_root'])
    lengths, initial, sources = {}, {}, {}
    for sequence in protocol['train']:
        info_path = Path(plan['evaluation']['GT_root'])/'train'/sequence/'seqinfo.ini'
        info = configparser.ConfigParser()
        info.read(info_path)
        lengths[sequence] = info.getint('Sequence','seqLength')
        path = work/'correlation_cache'/sequence/'features/complete.json'
        meta = json.loads(path.read_text())
        assert meta['status'] == 'COMPLETE_TRAIN_FROZEN_FPN' and not meta['Dev_rows'] and not meta['Test_rows']
        initial[sequence] = meta['frame_ids']
        sources[sequence] = dict(cache_manifest_sha256=sha256(path),seqinfo_sha256=sha256(info_path))
    assert sum(lengths.values()) == 27251
    schedule = make_schedule(lengths,initial)
    out = work/'structural_stream_schedule'
    out.mkdir(exist_ok=False)
    path = out/'frame_addresses.npz'
    np.savez_compressed(path,**schedule)
    rows = [dict(sequence=sequence,full_frames=lengths[sequence],passes=24,frames_per_pass=192,
                 sample_exposures=int(addresses.size),unique_frames=int(len(np.unique(addresses))),
                 first_pass_exact_to_existing_cache=bool(np.array_equal(addresses[0],initial[sequence])),
                 all_passes_have_distinct_frames=True) for sequence,addresses in schedule.items()]
    report = dict(status='PREPARED_TRAIN_ADDRESS_SCHEDULE_NO_TRAINING_ACTIVATED',rows=rows,
        source_sha256=sha256(__file__),protocol_sha256=sha256(protocol_path),sources=sources,
        frame_addresses=dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)),
        fixed_seed=0,additional_seed_work=False,
        old_unique_frames=1152,new_unique_frames=sum(r['unique_frames'] for r in rows),
        sample_exposures=27648,updates_at_batch4=6912,
        fixed_elements='Same six Train videos,192slots/video/pass,24passes and two-video/two-frame batch layout. '
                       'First pass exactly matches existing cached-frame addresses. All cue/control runs must share these addresses.',
        scope='Only frame-count metadata and old Train-cache manifests were read. No GT label rows, '
              'Dev/Test rows, images, features, GPU work or fits. This schedule alone is not a trained candidate.',
        requirements_before_fitting='Prove live B1 frozen-encoder, descriptors, targets and initialization parity '
            'on existing cached frames; register matched cue controls and all formal Test cases. '
            'No full-image/FPN cache may be created by this schedule.',
        inference='Additional unique observations could help; no accuracy improvement is claimed or assumed.',
        goal_met=False,supported_innovations=0)
    dump(out/'complete.json',report)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_TRAIN_STREAM_SCHEDULE.json',report)
    print(json.dumps(dict(status=report['status'],unique_frames=report['new_unique_frames'],
                          sample_exposures=27648,address_bytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
