from pathlib import Path

import pytest
import torch

from tools.neck72_frontend_serialization import (ROOT, PROOF, REGISTRY, PROOF_SOURCES,
                                               load_verified_frontend_model)
from tools.prove_neck72_frontend_serialization import compare_states
from tools.run_pseudo_ir_comparison import dump, sha256


@pytest.fixture
def proven(tmp_path):
    original = {'weight': torch.tensor([1.00005]), 'num_batches_tracked': torch.tensor(4097)}
    compact = {k: v.half() if v.is_floating_point() else v.clone() for k, v in original.items()}
    old, packed = tmp_path / 'original.pt', tmp_path / 'compact.pt'
    torch.save({'model': original}, old); torch.save({'model': compact}, packed)
    protocols = {}; sources = {}
    for split, weight in [('dev', old), ('test', packed)]:
        folder = tmp_path / split; folder.mkdir()
        metadata = dict(ckpt=str(weight), source_sha256=sha256(weight), seed=0, variant='rgb',
                        input_size=[800, 1440], conf=.1, nms=.7)
        path = folder / 'metrics.json'; dump(path, dict(protocol=metadata))
        protocols[split] = dict(path=str(path), sha256=sha256(path), protocol=metadata)
        sources['rgb_'+split] = str(folder)
    row = dict(frontend='rgb', source_protocols=protocols,
        original=dict(file=str(old), file_sha256=sha256(old)),
        compact=dict(file=str(packed), file_sha256=sha256(packed)), comparison=compare_states(original, compact))
    proof = dict(status='COMPLETE_NECK72_FRONTEND_SERIALIZATION_COMPARISON', all_frontends_exact=True,
        GPU_work=False, CNN_forward=False, rows=[row], source_hashes={p: sha256(ROOT / p) for p in PROOF_SOURCES})
    dump(tmp_path / PROOF, proof)
    dump(tmp_path / REGISTRY, dict(frontend_serialization_proof_sha256=sha256(tmp_path / PROOF)))
    return tmp_path, dict(frozen_detection_sources=sources), protocols['dev']['protocol'], proof, old, packed


def test_dev_uses_exact_compact_state_without_reading_the_original_archive(proven):
    work, plan, metadata, proof, old, packed = proven
    old.unlink()  # This historical file is provenance; runtime never opens it.
    model, source = load_verified_frontend_model(plan, 'rgb', 'dev', metadata, work)
    assert model['weight'].dtype == torch.float16
    assert model['num_batches_tracked'].dtype == torch.int64 and model['num_batches_tracked'].item() == 4097
    assert source['checkpoint'] == str(packed) and source['accuracy_source_checkpoint'] == str(old)
    assert not source['original_Dev_archive_read_by_runtime']


def test_changed_compact_bytes_are_rejected_before_model_use(proven):
    work, plan, metadata, proof, old, packed = proven
    packed.write_bytes(packed.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Checkpoint bytes differ'):
        load_verified_frontend_model(plan, 'rgb', 'dev', metadata, work)


def test_changed_normalization_or_source_metadata_cannot_reuse_the_proof(proven):
    work, plan, metadata, proof, old, packed = proven
    with pytest.raises(AssertionError):
        load_verified_frontend_model(plan, 'rgb', 'dev', dict(metadata, conf=.2), work)


def test_equal_flag_without_matching_canonical_digests_is_insufficient(proven):
    work, plan, metadata, proof, old, packed = proven
    proof['rows'][0]['comparison']['canonical_state_sha256']['compact'] = '0'*64
    dump(work / PROOF, proof)
    dump(work / REGISTRY, dict(frontend_serialization_proof_sha256=sha256(work / PROOF)))
    with pytest.raises(AssertionError):
        load_verified_frontend_model(plan, 'rgb', 'dev', metadata, work)
