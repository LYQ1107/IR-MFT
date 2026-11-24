"""Pin native fitting to the successful full-Train campaign's exact image bytes."""
from pathlib import Path

from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.run_pseudo_ir_comparison import sha256
import json


class PinnedTrainImageReader(ContentVerifiedTrainImageReader):
    def __init__(self, *args, expected_content, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_content = expected_content

    def read(self, sequence, frame, modality):
        key = sequence, int(frame), modality
        if key not in self.expected_content:
            raise ValueError('Observation is outside the pinned Train content manifest')
        result = super().read(sequence, frame, modality)
        actual = self.content_hashes[key]['sha256']
        expected = self.expected_content[key]['sha256']
        if actual != expected:
            raise RuntimeError(f'Original Train image differs from completed reference: {key}; '
                f'expected_SHA256={expected}; actual_SHA256={actual}')
        return result


def make_image_reader(work, plan, protocol, lengths, cue):
    if set(lengths) != set(protocol['train']) or cue not in {'rgb', 'gray', 'ir'}:
        raise ValueError('Only the fixed Train videos and cues are allowed')
    paths, digests = {}, set()
    for variant in ['111', '011', '101', '110']:
        complete_path = Path(work) / 'structural_full_train_fit' / f'ir_{variant}_s0/full_train_complete.json'
        complete = json.loads(complete_path.read_text())
        assert complete['status'] == 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
        assert complete['seed'] == 0 and complete['logical_Train_epochs_completed'] == 3
        manifest = complete['source_image_content_manifest']
        assert sha256(manifest['path']) == manifest['sha256']
        paths[variant] = dict(path=manifest['path'], sha256=manifest['sha256'],
                              full_train_complete_sha256=sha256(complete_path))
        digests.add(manifest['sha256'])
    assert len(digests) == 1, 'Four successful structural IR conditions must consume identical image bytes'
    records = json.loads(Path(paths['111']['path']).read_text())
    expected = {(r['sequence'], r['frame'], r['modality']): dict(sha256=r['sha256'], bytes=r['bytes']) for r in records}
    assert len(expected) == len(records) == 2*sum(lengths.values())
    assert set(expected) == {(seq, frame, modality) for seq, count in lengths.items()
                            for frame in range(1, count+1) for modality in ['rgb', 'ir']}
    expected = {key: value for key, value in expected.items() if cue == 'ir' or key[2] == 'rgb'}
    reader = PinnedTrainImageReader(plan['data_root'], plan['generated_ir_root'], protocol['train'],
                                   expected_content=expected)
    return reader, dict(reference_successful_IR_content_manifests=paths, expected_input_entries=len(expected),
        expected_manifest_sha256=next(iter(digests)), expected_data_not_labels_or_model_inputs=True,
        actual_read_counts_remain_in_reader_statistics=True)
