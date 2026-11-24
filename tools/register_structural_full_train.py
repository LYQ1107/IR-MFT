#!/usr/bin/env python3
"""Register three complete Train traversals and matched controls before fitting."""
import configparser
import copy
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_full_train_schedule import make_full_train_schedule, table_digest


def main():
    main_path = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = json.loads(main_path.read_text())
    protocol_path = ROOT / plan['protocol']
    protocol = json.loads(protocol_path.read_text())
    base_path = ROOT / 'configs/perceptual_structural_relative_20261123.json'
    base = json.loads(base_path.read_text())
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    policy = json.loads(policy_path.read_text())
    assert policy['fixed_seed'] == 0 and not policy['additional_seed_fits']
    assert policy['structural_base_config_sha256'] == sha256(base_path)
    lengths, metadata = {}, {}
    for seq in protocol['train']:
        path = Path(plan['evaluation']['GT_root']) / 'train' / seq / 'seqinfo.ini'
        ini = configparser.ConfigParser()
        ini.read(path)
        lengths[seq] = ini.getint('Sequence', 'seqLength')
        metadata[seq] = dict(seqinfo_sha256=sha256(path), frames=lengths[seq])
    assert len(lengths) == 6 and sum(lengths.values()) == 27251
    sequences, table, summaries = make_full_train_schedule(lengths)
    work = Path(plan['work_root'])
    schedule_dir = work / 'structural_full_train_schedule'
    schedule_dir.mkdir(exist_ok=False)
    address_path = schedule_dir / 'frame_addresses.npz'
    np.savez_compressed(address_path, addresses=table)
    proof = dict(status='REGISTERED_ALL27251_TRAIN_FRAMES_PER_LOGICAL_EPOCH',
        source_sha256=sha256(__file__), schedule_source_sha256=sha256(ROOT / 'tools/structural_full_train_schedule.py'),
        main_plan_sha256=sha256(main_path), protocol_sha256=sha256(protocol_path),
        sequence_order=sequences, lengths=lengths, source_metadata=metadata,
        address_file=dict(path=str(address_path), sha256=sha256(address_path), bytes=address_path.stat().st_size),
        semantic_address_sha256=table_digest(sequences, table), logical_epochs=summaries,
        macro_passes=72, updates=20736, sample_exposures=82944,
        fixed_seed=0, GT_label_rows_read=False, Dev_read=False, Test_read=False,
        scope='Address metadata only. Every Train frame appears in each logical traversal. '
              'Within-video distinct frame pairs preserve the original swap loss; two pairs may share a video. '
              'Each traversal has397repeated exposures for pairing/288update macro alignment. '
              'No images/FPNs copied and no formal model trained.',
        goal_met=False, supported_innovations=0)
    dump(schedule_dir / 'complete.json', proof)
    dump(ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_SCHEDULE.json', proof)
    config = copy.deepcopy(base)
    config.update(status='REGISTERED_FULL_TRAIN_DERIVATIVE',
        parent_structural_config_sha256=sha256(base_path),
        Train_frames=27251, epochs=72, snapshots=[24, 48, 72], updates=20736,
        macro_pass_definition='288batch4updates;24macro passes complete one entire Train traversal with397padding exposures.',
        logical_Train_epochs=3, cache=None,
        novelty='Same S/C/R architecture and losses; additional Train coverage/budget is an experimental condition, not an innovation.',
        evaluation='All3logicalTrain checkpoints,raw/priority on all4834Dev;same six choices for every cue/removal. '
                   'Primary stays the fullIR111architecture. Freeze Test choices using complete Dev only. '
                   'All engineering-passed formal configurations get full7Test even if Dev is negative.',
        scheduling='GPU4/5 only after the already registered factorial Test NN producers and queued liveGPU optimizer proof complete.',
        Test_read='Prior Test results are known; new Train/Dev fitting/selection never consumes Test accuracy.')
    config['fit']['batch'] = 'Four frames in two distinct-frame within-video pairs; natural-length coverage with fixed padding. All six fits share exact addresses.'
    config_path = ROOT / 'configs/perceptual_structural_fulltrain_20261123.json'
    assert not config_path.exists()
    dump(config_path, config)
    fits = [dict(cue=cue, components='111', physical_gpu=gpu, seed=0)
            for cue, gpu in [('rgb', '4'), ('gray', '5'), ('ir', '4')]]
    fits += [dict(cue='ir', components=bits, physical_gpu=gpu, seed=0)
             for bits, gpu in [('011', '5'), ('101', '4'), ('110', '5')]]
    source_Dev = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_ANALYSIS.json'
    coverage = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_TRAIN_COVERAGE.json'
    study = dict(status='REGISTERED_FULL_TRAIN_STUDY_NO_FITS_ACTIVATED',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        main_plan_sha256=sha256(main_path), protocol_sha256=sha256(protocol_path),
        parent_structural_config_sha256=sha256(base_path), configuration=str(config_path.relative_to(ROOT)),
        configuration_sha256=sha256(config_path), single_seed_policy_sha256=sha256(policy_path),
        schedule_proof_sha256=sha256(schedule_dir / 'complete.json'), semantic_address_sha256=proof['semantic_address_sha256'],
        source_Train_coverage_audit_sha256=sha256(coverage), source_complete_Dev_analysis_sha256=sha256(source_Dev),
        hypothesis='Existing1152frames cover4.227percent of Train and repeat24times; complete Dev curves peak early. '
                   'Test the same architecture on complete Train observations with matched controls. '
                   'This changes coverage, natural-length batch sampling and total updates; it is not a pure coverage-only attribution.',
        fits=fits, fixed_seed=0, additional_seed_work=False, trainable_parameters=145600,
        logical_Train_epochs_each=3, optimizer_macro_passes_each=72, updates_each=20736,
        snapshots=[24, 48, 72], training_input_hw=base['training_input_hw'],
        frozen_elements='Original NovemberIR files, protocol, RGBcheckpoint, frozen shared encoder/head, architecture, '
            'losses, calibration, optimizer LR/clip/weight decay, inference resolution, ByteTrack parameters. '
            'Warmup remains288updates; cosine total horizon increases to20736.',
        selection='FullIR111 primary selected by maximum completeDevHOTA over3snapshots and raw/priority. '
            'Tie: earlier macro pass, then raw. R/G receive the same6opportunities. '
            'Each removal has a primary-epoch/policy role and its independently Dev-selected best role. '
            'Never promote a Test winner or sum removal deltas.',
        mandatory_Test='All formal trained six configurations, including negative ones, cover7/7Test15985frames. '
            'Also correctIR/imageRoll/unavailable and fresh/canonicalRGB; freeze all choices after completeDev.',
        engineering='20actualGPUupdate smoke per fit, initialRGBexact, finite gradients; queued actualB1GPU optimizer parity must complete first. '
            'Source8zero/prefix32/full source and denseRGBgeometry proofs before accuracy.',
        scheduling='Do not launch until the liveGPU parity schedule is COMPLETE and registered downstream evaluation tooling is ready. '
            'Only physical4/5, one fit worker/card; current factorial pipeline is not modified or interrupted.',
        storage='Reuse originals, no image/FPN disk caches. EncodedRAM<=8GiB/worker with40GiB insertion reserve; '
            'work cap64GiB, disk free reserve40GiB. Only lightweight results/code in Git.',
        prior_Test_exposure='Previous fixed111Test29.142 and baseline29.529 are already known. '
            'This is exploratory work; design motivation is the pre-existing Train coverage audit and completeDev evidence. '
            'The new model/checkpoint choices use Train/Dev only.',
        originality='Training on more frames is not a contribution; S/C/R remain unproven mechanisms requiring matched attribution and prior-art assessment.',
        goal_met=False, supported_innovations=0)
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    assert not study_path.exists()
    dump(study_path, study)
    print(json.dumps(dict(status=study['status'], fits=len(fits), unique_Train_frames=27251,
                         logical_epochs=3, updates_each=20736, address_bytes=address_path.stat().st_size)))


if __name__ == '__main__':
    main()
