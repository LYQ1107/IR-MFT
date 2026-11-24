"""Reuse only an existing, proven identical serialized detector state."""
from pathlib import Path

from tools.neck72_runtime import read
from tools.prove_neck72_frontend_serialization import load_model_bytes
from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]
PROOF = 'neck72_frontend_serialization_proof.json'
REGISTRY = 'neck72_fresh_runtime_registered.json'
PROOF_SOURCES = {'tools/prove_neck72_frontend_serialization.py',
                 'tests/test_neck72_frontend_serialization.py'}


def load_verified_frontend_model(plan, arm, split, metadata, work):
    if arm not in {'new', 'rgb'} or split not in {'dev', 'test'}:
        raise ValueError('Unknown frozen frontend/split')
    proof_path = Path(work) / PROOF
    registry = read(Path(work) / REGISTRY)
    assert registry['frontend_serialization_proof_sha256'] == sha256(proof_path)
    proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_NECK72_FRONTEND_SERIALIZATION_COMPARISON'
    assert proof['all_frontends_exact'] and not proof['GPU_work'] and not proof['CNN_forward']
    assert set(proof['source_hashes']) == PROOF_SOURCES
    assert all(sha256(ROOT / p) == h for p, h in proof['source_hashes'].items())
    matches = [row for row in proof['rows'] if row['frontend'] == arm]
    assert len(matches) == 1
    row = matches[0]; comparison = row['comparison']
    assert comparison['all_state_tensors_exact'] and comparison['compared_tensors'] > 0
    assert not comparison['missing_keys'] and not comparison['extra_keys']
    assert comparison['canonical_state_sha256']['original'] == comparison['canonical_state_sha256']['compact']
    source = row['source_protocols'][split]
    prefix = 'new_diffv2ir' if arm == 'new' else 'rgb'
    manifest = Path(plan['frozen_detection_sources'][prefix+'_'+split]) / 'metrics.json'
    assert source['path'] == str(manifest) and sha256(manifest) == source['sha256']
    assert source['protocol'] == metadata and metadata['seed'] == 0
    compact = row['compact']
    assert row['source_protocols']['test']['protocol']['ckpt'] == compact['file']
    assert row['source_protocols']['test']['protocol']['source_sha256'] == compact['file_sha256']
    # Read and hash precisely the bytes deserialized, with no temporary weight
    # copy and no repeated read of the multi-GiB provenance-only Dev archive.
    model, loaded = load_model_bytes(compact['file'], compact['file_sha256'])
    return model, dict(serialization_revision=2,
        checkpoint=compact['file'], checkpoint_sha256=compact['file_sha256'],
        accuracy_source_checkpoint=metadata['ckpt'], accuracy_source_checkpoint_sha256=metadata['source_sha256'],
        original_Dev_archive=row['original']['file'], original_Dev_archive_sha256=row['original']['file_sha256'],
        original_Dev_archive_read_by_runtime=False,
        canonical_half_state_sha256=comparison['canonical_state_sha256']['original'],
        canonical_half_parameters_BN_and_integer_buffers_exact=True,
        serialization_proof_sha256=sha256(proof_path), actual_checkpoint_bytes_read=loaded['bytes_read'],
        preprocessing_metadata_unchanged=True,
        scope='Same complete nn.Module.half state, existing compact artifact, no model/checkpoint selection. '
              'Fresh B1 image/source/track equivalence remains an independent gate.')
