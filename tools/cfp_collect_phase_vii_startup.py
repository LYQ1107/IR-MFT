#!/usr/bin/env python3
"""Publish verified A0 startup evidence, leaving all unperformed work explicit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from cfp.provenance import sha256, write_json


A=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase7')
D=Path('docs/research')


def read_complete(path):
    data=json.loads(path.read_text())
    if data.get('status')!='COMPLETE':raise ValueError('Complete diagnostic required: '+str(path))
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inventory',type=Path,default=A/'reference_inventory_v3.json')
    args=parser.parse_args()
    raw=read_complete(A/'inherited_raw_rank_reproduction_v1.json')
    cue=read_complete(A/'inherited_cue_rank_reproduction_v1.json')
    extended=read_complete(A/'identity_geometry_decomposition_v1.json')
    expected_raw=read_complete(D/'PHASE_VI_RAW_COORDINATE_CONTROLS.json')
    expected_cue=read_complete(D/'PHASE_VI_RETAINED_CUE_UTILITY.json')
    if raw['videos']!=expected_raw['videos'] or cue['videos']!=expected_cue['videos']:
        raise ValueError('Inherited scientific diagnostic differs; investigate before publishing')
    if raw['query_manifest_sha256']!=cue['query_manifest_sha256'] or raw['query_manifest_sha256']!=extended['query_manifest_sha256']:
        raise ValueError('Different diagnostic supports')
    for sequence,result in extended['videos'].items():
        if result['input_and_cost_audit']['baseline_cost_max_error']>1e-6:
            raise ValueError('Reconstructed ordinary cost drift: '+sequence)
        if (result['retrieval']['gru']['known_positive_queries']!=raw['videos'][sequence]['frozen_ordinary_gru']['known_positive'] or
                abs(result['retrieval']['gru']['rank1_percent']-raw['videos'][sequence]['frozen_ordinary_gru']['rank1_percent'])>1e-10):
            raise ValueError('GRU candidate-ranking denominator drift: '+sequence)
    report=dict(status='STAGE_COMPLETE_GOAL_IN_PROGRESS', task_book_sha256=sha256(D/'PHASE_VII_TASK_BOOK.md'),
                sources={str(p):sha256(p) for p in [A/'inherited_raw_rank_reproduction_v1.json',A/'inherited_cue_rank_reproduction_v1.json',
                                                   A/'identity_geometry_decomposition_v1.json',D/'PHASE_VI_RAW_COORDINATE_CONTROLS.json',
                                                   D/'PHASE_VI_RETAINED_CUE_UTILITY.json']},
                reproduction=dict(all_six_raw_metric_vectors_exact=True,all_six_same_supported_cue_metric_vectors_exact=True),
                results=extended, offline_retained_past_cue_oracle=cue,
                scope='First fixed-state six-train-video diagnosis only. No encoder training, two-crop completion, new OOF tracking or full-online improvement is claimed.')
    support_path=A/'identity_geometry_decomposition_v2.json'
    support=read_complete(support_path) if support_path.exists() and json.loads(support_path.read_text()).get('status')=='COMPLETE' else None
    if support:
        if support['query_manifest_sha256']!=raw['query_manifest_sha256']:
            raise ValueError('History support experiment changed query bank')
        report['history_support_extension']=dict(path=str(support_path),sha256=sha256(support_path),result=support)
    write_json(D/'PHASE_VII_A0_STARTUP_MEASURED.json',report)
    rows=['# Phase VII identity evidence: first measured decomposition','',
          'Status: IN_PROGRESS. Inherited raw/GRU and same-supported retained-cue metrics were recalculated; all six videos exactly match the original scientific metric payloads. '
          'This is actual accepted-history detector-feature diagnosis with inherited exposure, not completed GT/detector-crop ReID training or new online MOT evidence.','',
          '| Video | Known-positive queries | Raw mean R1 | GRU pure R1 | Geometry row R1 | Real blended row R1 | Cost reconstruction max error |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for seq,value in extended['videos'].items():
        r=value['retrieval'];audit=value['input_and_cost_audit']
        rows.append(f"| {seq} | {r['gru']['known_positive_queries']} | {r['fifo_mean']['rank1_percent']:.4f} | {r['gru']['rank1_percent']:.4f} | {r['geometry_only']['rank1_percent']:.4f} | {r['original_row_cost']['rank1_percent']:.4f} | {audit['baseline_cost_max_error']:.3g} |")
    rows += ['', 'SN-015 uses the same10,710 known-positive queries for the first pure-ranking contrast. The separate retained-cue support uses8,631 queries: mean76.989920%, GT-selected retained-cue oracle83.489746%. '
             'Oracle cue filtering is offline analysis only; unknown history does not prove wrong identity or irrecoverability.','',
             'The GRU pure rank is35.845005%, while real geometry/confidence blending gives62.296919%, close to geometry-only62.324930%. '
             'Reconstruction reproduces real recorded costs within1.2e-7 on SN-015. Feature norms are within about1.2e-7 of1, and normalized prototype similarities reproduce saved evidence. '
             'Thus normalization or a different checkpoint does not explain this contrast; the tracker uses a different score and the geometry prior can largely dominate these fixed-state row decisions. '
             'This does not prove that removing GRU preserves full endogenous tracking: Hungarian competition, thresholds, later stages, births and changed histories are outside row retrieval.','',
             'PF-001 pure GRU96.534309% versus mean60.519% contradicts a blanket claim that the GRU always discards useful identity evidence. '
             'The residual GRU also moves far from original normalized history: SN-015 mean cosine to last0.22285 and to FIFO0.19086. These are measured outputs, not proof of the unique causal source of degradation.','',
             'Training source tools/cfp_build_prediction_queries.py uses an earlier ordinary RGB-tracker producer, known unique current identities, positive-present filtering, sampled past cutoff gaps1/5/15/30 and at least two same-frame IDs. '
             'The current diagnostic uses historical GRU-produced actual state graphs, retains unknown candidates and evaluates naturally missing histories/candidates separately. '
             'SN-015 was calibration for the temporal model and seen by the frozen visual encoder. Candidate label support, generating tracker, requested temporal cutoff and scene exposure differ; their individual contributions remain to be discriminated.','',
             'Next: stratify correct/mixed/unknown retained histories and common candidate support, construct the explicit two-crop identity diagnostic platform, and complete30 source audits before choosing visual/local/joint training architectures. '
             'Do not begin large JEPA training from the pure-rank discrepancy alone. PairAUC/EER/mAP/R1/R5 and gap/density/overlap-proxy breakdowns are stored in PHASE_VII_A0_STARTUP_MEASURED.json. '
             'Overlap/gap are proxies, not ground-truth occlusion or turning labels.','']
    if support:
        rows += ['Further actual-history support and appearance-region diagnosis (v2)','',
                 '| Video/support | Known-positive queries | FIFO R1 | GRU R1 |','|---|---:|---:|---:|']
        for seq,value in support['videos'].items():
            for label,groups in value['by_observable_proxy'].items():
                if not label.startswith('retained_history_support/'):continue
                mean,gru=groups['fifo_mean'],groups['gru']
                if mean['known_positive_queries']:
                    rows.append(f"| {seq}/{label.split('/')[-1]} | {mean['known_positive_queries']} | {mean['rank1_percent']:.4f} | {gru['rank1_percent']:.4f} |")
        rows += ['', 'On SN-015 all2,716 actual histories whose retained proposal labels are verified correct, FIFO99.5950% versus GRU52.6510%. '
                 'Contamination therefore cannot by itself explain the weak isolated GRU representation on this subset. '
                 'A restricted native appearance region has7,081 positive-present queries, FIFO94.1110% versus GRU84.5078%. '
                 'Changing the candidate pool changes both problem difficulty and denominator; these94/84 values must not be directly compared to the10,710-query62/35 values as a method improvement. '
                 'The remainder is consistent with scene/target/producer distribution shift, but does not uniquely prove one mechanism.','']
    (D/'PHASE_VII_REID_COMPARISON.md').write_text('\n'.join(rows))
    plan=json.loads(Path('configs/cfp/phase_vii_reference_inventory_v1.json').read_text())
    inventory=json.loads(args.inventory.read_text())
    if inventory['status']!='INVENTORY_COMPLETE':raise ValueError('Terminal inventory required')
    registry=dict(status='IN_PROGRESS',scope='All30 requested entries; exact available Git commits are inventoried, semantic audits are incomplete.',repositories={})
    for name,slug,relative in plan['repositories']:
        path=Path(plan['references_root'])/relative
        entry=dict(inventory['repositories'].get(name,{}),name=name,github='https://github.com/'+slug,
                   requested_local_path=str(path),audit_status='NOT_AUDITED')
        if (path/'.git').exists():
            entry['commit']=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        entry['required_unverified_fields']=['paper_year_scene_contribution','license_and_weight_access','training_entry_and_sampler',
            'inference_entry_and_temporal_scope','core_model_files_and_shapes','loss_and_positive_negative_construction',
            'pretrained_sources','GPU_CUDA_dependencies','IR_MFT_compatibility','mechanisms_to_port_and_exclude','novelty_overlap']
        registry['repositories'][name]=entry
    write_json(D/'PHASE_VII_REPOSITORY_REGISTRY.json',registry)
    audit=['# Phase VII30-reference source audit','',
           'Status: IN_PROGRESS. Source readiness and exact Git commit inventory do not constitute a completed semantic audit. '
           'New sparse snapshots omit weights/media and live outside IR-MFT. Failed checkout/network attempts are preserved. No external project code has been executed.','',
           '| Project | Exact available commit | Source acquisition | Semantic audit |','|---|---|---|---|']
    for name,entry in registry['repositories'].items():
        audit.append(f"| [{name}]({entry['github']}) | {entry.get('commit','UNAVAILABLE')} | {entry.get('status','PENDING')} | NOT COMPLETE |")
    audit += ['', 'Initial inspected code findings (partial audits)','',
              'MOTIP atffc0e905ac196a603027eca8d18fb0dff48c8bcc: training train.py (train_engine/train_one_epoch/prepare_for_motip), inference submit_and_evaluate.py and models/runtime_tracker.py. '
              'models/motip/id_decoder.py concatenates feature and local-ID embeddings, uses per-frame detection self-attention after the first layer, then cross-attends all past track/time tokens. '
              'Its causal mask rejects trajectory time>=current query time. models/motip/id_criterion.py uses masked CE or focal ID loss; official training also includes detector loss. '
              'configs/r50_deformable_detr_motip_dancetrack.yaml has feature256+ID256,6 layers,50 IDs+newborn,30-frame clips and interval<=4. '
              'data/naive_sampler.py enumerates legal clips and data/transforms.py randomly binds physical objects to local dictionary IDs and simulates occlusion/switches/newborns. '
              'RuntimeTracker.update processes current image then updates historical memory; frame detector input is replaced if adapting to frozen V3 proposals. '
              'Top-level LICENSE identifies Apache2.0; dependencies/individual subcomponents and pretrain license/access still require completion. '
              'The full official system and a fixed-V3 head adaptation must be reported separately; full-set attention/local dictionary competition overlap our proposed C contribution.','',
              'DINOv3 at6876159a11b4df116f30f667f8c9888617df0751: dinov3/hub/backbones.py defines vits16 (patch16,dim384,12 layers,6 heads,4 storage tokens); '
              'dinov3/models/vision_transformer.py::forward_features returns normalized CLS/storage/patch tokens. At128x256 this implies128 spatial patch tokens before additional CLS/storage tokens, not anatomical parts. '
              'dinov3/train/train.py and dinov3/train/ssl_meta_arch.py expose multi-crop/masking/EMA and DINO/iBOT/Gram-related objectives. '
              'LICENSE.md is the custom DINOv3 agreement dated2025-08-19; code and weight availability must not be treated as unrestricted Apache licensing. '
              'No weights downloaded or access terms accepted on behalf of the user. Full sampling/loss/dependency audit and lawful weight-access verification remain pending.','',
              'SU-T at479be89a27129f6879c62710a83abac6a8b261ee: README identifies a2025 preprint/AAAI2026 paper; tools/train.py and tools/su_tracker.py are entry candidates. '
              'trackers/SUT_UKF.py uses custom9-state/5-measurement filtering by default and includes a separate FilterPy UKF branch; trackers/UKF.py actually computes sigma points, but propagation falls back to linear F when fx is unset. '
              'The named fx in the tracker is a constant-velocity matrix, so a UKF label alone does not establish learned/nonlinear swimming dynamics. '
              'trackers/association.py::fish_iou_batch and association routes need active entry/default configuration tracing before porting. MIT top-level license verified; external FilterPy attribution remains separate. '
              'Isolate filter choice, geometry metric and lifecycle/output; copying the whole tracker would mix them. Training/loss/pretrained-data/feature audit is still incomplete.','',
              'TIDE atfba27aefd1da11bfc7f946240dd66d85ea02a39c: README points to2026 arXiv and MFT-Edge; src/track.py and src/lib/tracker/mc_bot_sort.py form a current-frame tracker route. '
              'src/lib/tracker/matching.py defines AGC variants, including AGCv1 adding inverse center-distance to IoU. Zero-distance numerical behavior and which AGC variant is actually called must be checked before a faithful geometry control. '
              'README training/reid64 and dual-branch network are not equivalent to our frozen V3/RGB pipeline. MIT top-level license verified; full network/active metric/loss/dataset audit remains pending.','',
              'All other projects, and unresolved fields above, require actual source reading and primary paper/weight references. No row is marked AUDIT_COMPLETE.','']
    (D/'PHASE_VII_30_PAPERS_CODE_AUDIT.md').write_text('\n'.join(audit))
    state=dict(status='IN_PROGRESS',stage='SOURCE_AUDIT_AND_A0_DIAGNOSTICS',branch='research/evidence-fishmot-v1',
               baseline_commit='042608779d936bcd442d8c266734a6d4d47636c1',
               completed=['Clean local/remote PhaseVI ref checked; newPhaseVII branch created','Complete user1723-line task book preserved',
                          'All6videos inherited raw-rank and same-support cue measurements exactly reproduced',
                          'All6videos actual-state native geometry/blending decomposition complete','Offline known-only retrieval/AUC/EER/physical-video identity guards added'],
               pending=json.loads((D/'PHASE_VII_EXPERIMENT_REGISTRY.json').read_text())['pending'],
               reference_inventory_sha256=sha256(args.inventory),
               measured_a0_sha256=sha256(D/'PHASE_VII_A0_STARTUP_MEASURED.json'),
               storage_free_gib=shutil.disk_usage(A).free/2**30,source_audit_complete=0)
    if support:
        state['completed'].append('Six-video all/some/no verified actual-history and native appearance-region support diagnosis complete')
        state['history_support_output_sha256']=sha256(support_path)
    state['next_actions']=['Complete remaining14 source acquisitions and all30 semantic audits',
                           'Diagnose original temporal-training producer/candidate support shift',
                           'Build controlled GT/detector crop identity pairs and encoder-level heldout visual/local comparisons',
                           'Implement full-set joint ID competition and supervised NULL controls without waiting for final A retention']
    write_json(D/'PHASE_VII_STATE.json',state)
    experiment_path=D/'PHASE_VII_EXPERIMENT_REGISTRY.json'
    experiments=json.loads(experiment_path.read_text())
    for entry in experiments['experiments']:
        entry['source_commit_at_stage_collection']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if entry['experiment_id']=='VII_SOURCE_INVENTORY_V1':
            entry.update(status='INVENTORY_COMPLETE_AUDIT_PENDING',
                         inventory_path=str(args.inventory),
                         inventory_sha256=sha256(args.inventory),
                         source_ready=sum(e['status']=='SOURCE_READY' for e in inventory['repositories'].values()),
                         acquisition_failed=sum(e['status']=='ACQUISITION_FAILED' for e in inventory['repositories'].values()),
                         failure_notes=['v1 aborted at preserved external-reference symlink validation; summary was left RUNNING but process is terminal.',
                                        'v2 preserved15 acquisition/checkout failures, predominantly intermittent TLS; incomplete new sparse clones not discarded.',
                                        'Some exact Git heads are available without materialized source; those are not audited.'])
        elif entry['experiment_id']=='VII_A0_INHERITED_RANK_REPRODUCTION_V1':
            entry.update(status='COMPLETE',all_six_videos_exact=True,
                         raw_output_sha256=sha256(A/'inherited_raw_rank_reproduction_v1.json'),
                         cue_output_sha256=sha256(A/'inherited_cue_rank_reproduction_v1.json'),
                         checkpoint_sha256=raw['checkpoint_sha256'],query_manifest_sha256=raw['query_manifest_sha256'])
        elif entry['experiment_id']=='VII_A0_GEOMETRY_RETRIEVAL_DECOMPOSITION_V1':
            entry.update(status='COMPLETE',output_sha256=sha256(A/'identity_geometry_decomposition_v1.json'),
                         checkpoint_sha256=extended['checkpoint_sha256'],query_manifest_sha256=extended['query_manifest_sha256'],
                         conclusion='Native blended row scores reproduce actual costs; geometry largely explains weak-GRU/strong-row discrepancy inSN015. PF001 reverses pure-rank contrast. No blanket GRU information-loss or full MOT causation claim.',
                         next_experiment='Shared candidate-label support and clean/mixed/unknown actual history stratification; inspect original query producer shift, then two-crop/local evidence.')
        elif entry['experiment_id']=='VII_SOURCE_ACQUISITION_REPAIR_V3':
            entry.update(status='COMPLETE_WITH_PRESERVED_ACQUISITION_FAILURES',
                         inventory_sha256=sha256(args.inventory),
                         source_ready=sum(e['status']=='SOURCE_READY' for e in inventory['repositories'].values()))
        elif entry['experiment_id']=='VII_A0_HISTORY_SUPPORT_AND_ELIGIBLE_POOL_V2' and support:
            entry.update(status='COMPLETE',output_sha256=sha256(support_path),
                         checkpoint_sha256=support['checkpoint_sha256'],source_snapshots=support['source_snapshots'],
                         conclusion='On SN015 all-verified actual histories FIFO99.595/GRU52.651. Contamination alone insufficient. Restricted native region FIFO94.111/GRU84.508 on7081 supported queries; smaller denominator is explicit.',
                         next_experiment='Inspect training/trace producer shift and train same-video hard-negative local/joint controls under true encoder-level outer exclusion; no large JEPA before source audit/target discrimination.')
    tests=A/'project_tests_startup_v1.log'
    if tests.exists() and '191 passed' in tests.read_text():
        experiments['validation']=dict(project_tests=191,warnings=18,log=str(tests),log_sha256=sha256(tests))
        state['validation']=experiments['validation']
        write_json(D/'PHASE_VII_STATE.json',state)
    experiments['execution_failures']=[dict(stage='startup_collector',error='Exact floating-point percent comparison differed at roundoff despite equal query counts; fixed tolerance1e-10 with explicit denominator check.',
                                            repaired=True,scientific_metric_vectors_changed=False)]
    write_json(experiment_path,experiments)
    print(json.dumps(dict(status=state['status'],exact_reproduction=True,reference_entries=len(registry['repositories']),
                          source_ready=sum(e.get('status')=='SOURCE_READY' for e in registry['repositories'].values())),indent=2))


if __name__=='__main__':main()
