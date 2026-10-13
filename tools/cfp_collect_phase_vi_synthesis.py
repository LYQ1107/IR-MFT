#!/usr/bin/env python3
"""Verify completed scientific evidence and write a scope-aware living report."""
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json


A = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase6')
D = Path('docs/research')
FIELDS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW')


def verified_official(name):
    path = A / name
    result = json.loads(path.read_text())
    if result['status'] != 'COMPLETE':
        raise ValueError('Complete official evidence required: ' + name)
    for entry in result['families'].values():
        command = entry['evaluation_command']
        evaluation = Path(command[command.index('--output') + 1]) / 'metrics.json'
        if sha256(evaluation) != entry['evaluation_sha256']:
            raise ValueError('Evaluation drift')
        record = json.loads(evaluation.read_text())
        if record['metrics'] != entry['metrics']:
            raise ValueError('Metric payload drift')
        predictions = Path(entry['aggregate_command'][-1])
        if sha256(predictions / 'manifest.json') != entry['aggregate_manifest_sha256']:
            raise ValueError('Aggregate drift')
        for seq, artifact in record['artifacts'].items():
            if sha256(predictions / (seq + '.txt')) != artifact['prediction_sha256']:
                raise ValueError('Prediction drift')
    return dict(path=str(path), sha256=sha256(path), result=result)


def verified_identity_seed_evidence():
    """Verify the seed1/2 replay families and their matching disabled baselines."""
    path = A / 'identity_extended_official_v1.json'
    result = json.loads(path.read_text())
    if result.get('status') != 'COMPLETE':
        raise ValueError('Complete identity-coordinate evidence required')
    for family, entry in result['families'].items():
        evaluation = Path(entry['evaluation_command'][entry['evaluation_command'].index('--output') + 1]) / 'metrics.json'
        if sha256(evaluation) != entry['evaluation_sha256']:
            raise ValueError('Identity evaluation drift: ' + family)
        record = json.loads(evaluation.read_text())
        if record['metrics'] != entry['metrics']:
            raise ValueError('Identity metric payload drift: ' + family)
        if family.startswith('identity_baseline_'):
            aggregate = Path(entry['evaluation_command'][entry['evaluation_command'].index('--predictions') + 1])
        else:
            aggregate = Path(entry['aggregate_command'][entry['aggregate_command'].index('--output') + 1])
        if sha256(aggregate / 'manifest.json') != entry['aggregate_manifest_sha256']:
            raise ValueError('Identity aggregate drift: ' + family)
        for sequence, artifact in record['artifacts'].items():
            if sha256(aggregate / (sequence + '.txt')) != artifact['prediction_sha256']:
                raise ValueError(f'Identity prediction drift: {family}/{sequence}')
    return dict(path=str(path), sha256=sha256(path), result=result)


def statistics(vectors):
    return {key: dict(mean=float(np.mean([m[key] for m in vectors])),
                      std_population=float(np.std([m[key] for m in vectors])),
                      std_sample=float(np.std([m[key] for m in vectors], ddof=1))) for key in FIELDS}


def table(groups):
    rows = ['| Method / scope | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, metrics in groups:
        rows.append('| ' + name + ' | ' + ' | '.join(f'{metrics[k]:.4f}' for k in FIELDS) + ' |')
    return rows


def main():
    report = dict(status='FINAL_SCOPED_CONCLUSION', scientific_success='SCOPED_NEGATIVE', official={}, model_seed_statistics={},
                  scope='Association-module heldout videos with frozen detector/ReID exposure. '
                        'Identity-coordinate seed expansion changes RGB/ordinary-producer/module seeds together but retains detector exposure. '
                        'No blind whole-system generalization claim.')
    names = {
        'quota': 'oof_quota_official_v1.json', 'risk_repeats': 'oof_quota_risk_init_official_v1.json',
        'relation': 'oof_relation_official_v1.json', 'relation_extra': 'oof_relation_extra_official_v1.json',
        'residual0': 'oof_residual_official_v1.json', 'residual_repeats': 'residual_discriminating_official_v1.json',
        'combinations': 'oof_combination_official_v1.json',
    }
    for key, name in names.items():
        report['official'][key] = verified_official(name)
    extension = A / 'oof_AB_extension_official_v1.json'
    if extension.exists() and json.loads(extension.read_text())['status'] == 'COMPLETE':
        report['official']['AB_extension'] = verified_official(extension.name)
    simple = A / 'predictive_margin_seed0_official_v1.json'
    if simple.exists() and json.loads(simple.read_text())['status'] == 'COMPLETE':
        report['official']['simple_seed0'] = verified_official(simple.name)
    identity_path = A / 'identity_extended_official_v1.json'
    if identity_path.exists():
        report['official']['identity'] = verified_identity_seed_evidence()
    report['no_calibration'] = json.loads((D / 'PHASE_VI_AB_NO_CALIBRATION_PROOF.json').read_text())
    base = report['official']['residual0']['result']['families']
    repeated = report['official']['residual_repeats']['result']['families']
    sequences = list(next(iter(base.values()))['metrics'])
    for kind in ('ordinary', 'noema', 'ema'):
        records = [base[kind + '_residual_utility']['metrics'],
                   *[repeated[kind + '_residual_utility_multiseed_init' + str(i)]['metrics'] for i in (1, 2)]]
        report['model_seed_statistics'][kind] = {seq: statistics([m[seq] for m in records]) for seq in sequences}
    if 'AB_extension' in report['official']:
        ext = report['official']['AB_extension']['result']['families']
        records = [report['official']['combinations']['result']['families']['AB']['metrics'],
                   ext['AB_init1']['metrics'], ext['AB_init2']['metrics']]
        report['model_seed_statistics']['AB'] = {seq: statistics([m[seq] for m in records]) for seq in sequences}
    if 'identity' in report['official'] and 'simple_seed0' in report['official']:
        identity_result = report['official']['identity']['result']
        seed_rows = [dict(seed=0,
                          baseline=report['official']['combinations']['result']['families']['B0']['metrics']['COMBINED_SEQ'],
                          candidate=report['official']['simple_seed0']['result']['families']['predictive_margin_identity_seed0']['metrics']['COMBINED_SEQ'])]
        for seed in (1, 2):
            seed_rows.append(dict(seed=seed,
                                  baseline=identity_result['families'][f'identity_baseline_seed{seed}']['metrics']['COMBINED_SEQ'],
                                  candidate=identity_result['families'][f'predictive_margin_identity_seed{seed}']['metrics']['COMBINED_SEQ']))
        deltas = [{field: row['candidate'][field] - row['baseline'][field] for field in FIELDS} for row in seed_rows]
        report['identity_conclusion'] = dict(
            candidate_family='predictive_margin_identity', seeds=seed_rows,
            delta_statistics=statistics(deltas), decision='DO_NOT_RETAIN_FOR_DEPLOYMENT',
            reason='Seed1/2 gains are small (+0.0258/+0.0613 HOTA) and each adds 2 IDSW; seed0 has a larger gain but +16 IDSW. '
                   'The candidate therefore does not establish reliable identity recovery beyond the ordinary baseline in this scope.')
    events_path = A / 'combination_event_evidence_v1/manifest.json'
    events = json.loads(events_path.read_text())
    report['events'] = dict(path=str(events_path), sha256=sha256(events_path), status=events['status'], paired=events['paired'])
    if events['status'] == 'COMPLETE':
        report['events']['official_comparison'] = {}
        baseline = events['official_events']['B0']['result']['videos']
        for kind in ('C', 'BC', 'ABC'):
            changed = events['official_events'][kind]['result']['videos']
            rows = {}
            for seq, original in baseline.items():
                old = {(e['frame'], e['gt_id']) for e in original['events']}
                new = {(e['frame'], e['gt_id']) for e in changed[seq]['events']}
                if len(old) != original['official_idswitch_count'] or len(new) != changed[seq]['official_idswitch_count']:
                    raise ValueError('Duplicate or missing official switch observations')
                rows[seq] = dict(baseline=len(old), variant=len(new), common=len(old & new),
                                 baseline_events_absent=len(old-new), variant_new_events=len(new-old))
            report['events']['official_comparison'][kind] = rows
    report['online'] = json.loads((D / 'PHASE_VI_RISK_ONLINE_EVIDENCE.json').read_text())
    ops = A / 'canonical_affine_ops_v2.json'
    report['affine_ops'] = dict(path=str(ops), sha256=sha256(ops), result=json.loads(ops.read_text()))
    if report['affine_ops']['result']['current_padding']['rgb_items_per_encoding_chunk'] != 128:
        raise ValueError('Actual canonical RGB padding must be128')
    write_json(D / 'PHASE_VI_SYNTHESIS_MEASURED.json', report)

    combo = report['official']['combinations']['result']['families']
    identity = report['official'].get('identity')
    identity_result = identity['result'] if identity else None
    identity_complete = identity_result is not None
    lines = ['# IR-MFT Phase VI research report', '',
             '**Status: FINAL_SCOPED_CONCLUSION.** The measured evidence supports retaining the strong ordinary GRU/current-measurement '
             'baseline and not deploying the candidate modules. The conclusion is scoped to the frozen detector/ReID exposure and six-video '
             'association protocol; it is not a claim that all recoverability mechanisms are impossible.', '',
             '## 1. Research question and exploration route', '',
             'Can strictly past accepted observations and the current candidate set identify recoverable ambiguity, '
             'separate plausible identities and protect stable memory? The route was baseline reproduction → actual-error taxonomy '
             '→ risk/ordinary controls → independent K2 → candidate attention/EMA controls → video-excluded state regeneration '
             '→ write-budget controls → raw relational prediction → bounded corrections → repeated seeds and full ABC interactions. '
             'Failed models were preserved and used to design discriminating experiments; no fixed trial count or HOTA threshold ended research.', '',
             '## 2. Strong references and data isolation', '',
             'Three-seed historical dev GRU/current measurements: HOTA53.591386,AssA39.138938,DetA73.456991,IDF162.700856,MOTA93.672112,IDSW87.333333. '
             'Preserved native BoT/RGB: HOTA52.481818,AssA37.498032,DetA73.517605,IDF157.476822,MOTA93.552298,IDSW113.666667. '
             'All three GRU outputs were exactly reproduced; six official reevaluations and88 inherited hashes verified. '
             'Native BoT output policy differs and is an additional strong reference, not a same-output-policy factor ablation. '
             'The old smoothed GRU51.885 is not used to inflate improvement.', '',
             'Each new fold excludes one complete outer video and a different inner calibration video from supervised producer/module fitting. '
             'Actual accepted states are regenerated by the excluded-video ordinary predictor, not corrected with GT. '
             'Frozen ReID has seen all six train videos and V3 has seen train/dev; OOF is conditional association evidence. '
             'BT005/SN013 are exposed historical references and the seven historically tuned test videos were excluded from search. '
             'The train-fold HOTA33.x and two-dev HOTA53.x concern different video sets and must not be directly subtracted.', '',
             '## 3. A: risk decisions and memory protection', '',
             'A uses21 observable edge features: prototype/history similarities, margins/entropy, IoU/motion, confidence, age/gap, stage, crowding, '
             'ordinary cost and cross-track margin. Same-input MLP, candidate pooling, learned cost threshold, history removal and calibrated '
             'untrained controls separate learning from simple gating. Unknown physical labels remain excluded from known-wrong supervision. '
             'Direct association replacement/rejection harms FN/MOTA; preserving current assignment/output while deferring suspect writes is safer. '
             'Absolute calibrated gates fail to transfer to regenerated producer states. Fixed per-stage floor(10% of actually matched scored edges) '
             'makes the action budget explicit; compare confidence, margin, entropy and seeded random ordering. '
             'Small pools cannot reject an edge and use ordinary writes. This is not a calibrated posterior of recoverability.', '',
             '## 4. B: candidate-conditioned prediction and teacher/student mechanism', '',
             'The initial109,186-parameter candidate attention/EMA family uses real accepted histories, current candidates, explicit NULL, masks and '
             'known-only contamination supervision. Ordinary candidate MLP/GRU/attention, two-layer Transformer105,153 and an ID-dictionary '
             'decoder108,416 are local controls. The ID decoder is MOTIP-inspired rather than a faithful official MOTIP implementation; '
             'whole-frame grouping/loss information differ, so no claim of defeating MOTIP is supported.', '',
             'The subsequent relation family operates on raw RGB cosine/candidate margins, actual first3+recent5 tokens, time and current geometry. '
             'Training-only EMA encodes clean current/future identity-to-candidate relations; future targets have no current fallback. '
             'Student forward/export has no GT, future crop or teacher. Cached full-history scalar channels are excluded from masked relation inputs. '
             'A real512-update EMA and exact model/optimizer/RNG epoch3-resume were verified. Known-negative missing-candidate augmentation '
             'increases NULL rejection but also wrongly rejects positives.', '',
             'Unrestricted learned cost replacement creates large IDSW. The residual variant preserves the unmasked ordinary prior and '
             'limits supplemental changes to±0.05 inside native appearance eligibility. Masking applies only to the supplemental branch. '
             'It has4,389 parameters, zero-initialized correction, candidate-centered latent prediction and optional known-cue utility. '
             'Its own epoch3 stop/resume matches continuous model/EMA/optimizer/all RNG/order/loss/guard exactly, excluding only Git provenance.', '',
             '## 5. C: bounded causal alternative explanations', '',
             'K2 branches on a legal near-tied pair swap with ambiguity0.05, discounted current matching cost0.9 and optional fixed0.02 hysteresis. '
             'Each explanation owns motion, accepted/early history, writer evidence, counters and RNG. Current frames use the current winner; '
             'future observations can change future decisions but cannot revise emitted history. Branch score is a heuristic, not a posterior. '
             'Duplicate-output, bounded-width, birth reuse, isolation and nested quota/alternate assignment guards are tested. '
             'K1/all-disabled complete outputs reproduce every six-video outer baseline byte-for-byte.', '',
             '## 6. Difference from RIS, PLIM and Fish-JEPA', '',
             'RIS weights historical reliability into a prototype; PLIM/Fish-JEPA predict a single identity representation before comparing candidates. '
             'Phase VI conditions on current candidate relations, separates memory action from output acceptance, and tests explicit alternative '
             'trajectory states. Added complexity alone is not innovation: teacher, learned risk and branching need independent evidence.', '',
             '## 7. Recoverability diagnosis and scoped oracle', '',
             'Known wrong rows often retain a current-stage positive but require score/acceptance repair. Geometry eligibility is appearance blending, '
             'not a hard Hungarian mask. Detection absence, stage absence, unknown labels, weak past cues and one-to-one competition are separated. '
             'An unknown past token never proves irrecoverability. Joint GT interventions on fixed states reduce naive independent reranking '
             'opportunity; they are analysis-only and not a tracking ceiling or inference mechanism. '
             'PHASE_VI_RECOVERABILITY_ANALYSIS.md and PHASE_VI_JOINT_ORACLE_MEASURED.json retain full two-axis/category/episode evidence.', '',
             '## 8. Full-video same-fold ABC results', '']
    lines += table([(kind, entry['metrics']['COMBINED_SEQ']) for kind, entry in sorted(combo.items())])
    lines += ['', 'AB has a seed0 positive signal; adding C lowers HOTA. This is not a final architecture selection. '
              'All metrics use the same frozen detection/output/evaluator rules and full videos.', '',
              '## 9. Repeated candidate-prediction initialization/query seeds', '',
              '| Family, encoder/producer0 fixed | HOTA mean±population SD | AssA mean | IDF1 mean | IDSW mean |',
              '|---|---:|---:|---:|---:|']
    for kind, group in report['model_seed_statistics'].items():
        s = group['COMBINED_SEQ']
        lines.append(f"| {kind} | {s['HOTA']['mean']:.4f}±{s['HOTA']['std_population']:.4f} | {s['AssA']['mean']:.4f} | {s['IDF1']['mean']:.4f} | {s['IDSW']['mean']:.4f} |")
    lines += ['', 'These are three initialization/query/mask seed repetitions, not three encoder/producer pipelines or independent scene confidence intervals. '
              'EMA and noEMA are nearly equal; no independent EMA advantage is established. Ordinary residual-only CE is worse. '
              'Utility-off controls do not remove the small predictive signal. Exact per-video vectors and both SD conventions are in the measured JSON.', '',
              '## 9b. Identity-coordinate and ordinary-producer seed replication', '']
    if identity_complete:
        identity_families = identity_result['families']
        lines += ['The following comparison changes the frozen RGB coordinate, excluded-video ordinary producer and residual-module seed together. '
                  'Each seed has a matching disabled baseline assembled from its own actual causal state replay; seed0 metrics are not used as its baseline. '
                  'The detector and its exposure remain shared, and this is conditional association evidence rather than whole-system unseen-scene validation.', '',
                  '| Identity-coordinate family | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for family in ('identity_baseline_seed1', 'ordinary_B_identity_seed1', 'predictive_B_identity_seed1',
                       'margin_only_identity_seed1', 'predictive_margin_identity_seed1',
                       'identity_baseline_seed2', 'ordinary_B_identity_seed2', 'predictive_B_identity_seed2',
                       'margin_only_identity_seed2', 'predictive_margin_identity_seed2'):
            metrics = identity_families[family]['metrics']['COMBINED_SEQ']
            lines.append('| ' + family + ' | ' + ' | '.join(f'{metrics[k]:.4f}' for k in FIELDS) + ' |')
        lines += ['', '| Seed/family | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔMOTA | ΔIDSW |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for key, comparison in sorted(identity_result['comparisons'].items()):
            delta = comparison['delta_combined']
            lines.append('| ' + key + ' | ' + ' | '.join(f'{delta[k]:+.4f}' for k in FIELDS) + ' |')
        lines += ['', 'Both seeds are retained in the measured evidence; no family is selected by a heldout result. '
                  'The ordinary residual control separates the predictive loss from the bounded residual mechanism, while margin-only separates '
                  'the write-order mechanism from learned risk. Exact per-video vectors, state hashes and replay/evaluation commands are in '
                  'PHASE_VI_SYNTHESIS_MEASURED.json and identity_extended_official_v1.json. Across the three available identity seeds, '
                  'predictive+fixed-margin changes HOTA by +0.9314, +0.0258 and +0.0613 while changing IDSW by +16, +2 and +2; '
                  'the seed1/2 effect is too small and the identity-switch direction is unfavorable for deployment.', '']
    else:
        lines += ['Identity-coordinate seed replication is still running; its results are not included in this snapshot.', '']
    lines += [
              '## 10. Earlier negative results and ordinary explanations', '',
              'Warm context-write HOTA33.2667 is matched by a2-parameter cost threshold33.3015. Regenerated state gates lose '
              '(context32.4190/MLP32.4971/threshold32.3156 vs33.4884). Quota risk context three-init mean33.7378 and MLP33.7456 '
              'do not exceed fixed margin33.8863. Raw-relation cost replacement gives HOTA26.8804–27.6981 and13,777–15,796IDSW; '
              'native geometry/eligibility restores stability. These negative findings remain preserved, not omitted in favor of AB.', '',
              'AB initialization repeats yield HOTA34.1118±0.1784. Fixed margin+B reaches34.4198 and noEMA+B with learned A reaches34.3855, '
              'so the learned risk network and EMA target are not required by the present positive signal. '
              'Temperature/bias removal reproduces all six complete AB predictions and memory/quota counts byte-for-byte; '
              'this quota method does not require risk probability calibration. This is not a claim that absolute calibrated gates are unnecessary. '
              'The simpler noEMA prediction+fixed-margin budget completed independently frozen RGB1/2, excluded-video producer1/2 and '
              'matched ordinary/noEMA module1/2 replication. Its seed1/2 gains are small and its IDSW rises; the detector remains fixed '
              'and backbone exposure remains. The deployment recommendation therefore stays with ordinary GRU/current measurements.', '',
              '## 11. Paired error corrections and official events', '',
              'N01/N10 pair only common observed-birth references with known birth/current labels. Newborn/disappeared support and unknowns '
              'are separate; persistent decisions are counted separately from consecutive episodes. These do not equal official IDSW.', '',
              '| Variant vs B0 | N01 | N10 | Net | N01 episodes | N10 episodes | Baseline-only | Variant-only | Unknown |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('C', 'AB', 'AC', 'BC', 'ABC'):
        vectors = [value for key, value in events['paired'].items() if key.startswith(kind + '_')]
        sums = {key: sum(v[key] for v in vectors) for key in ('N01', 'N10', 'N01_episodes', 'N10_episodes', 'baseline_only', 'variant_only', 'unknown')}
        lines.append(f"| {kind} | {sums['N01']} | {sums['N10']} | {sums['N01']-sums['N10']} | {sums['N01_episodes']} | {sums['N10_episodes']} | {sums['baseline_only']} | {sums['variant_only']} | {sums['unknown']} |")
    lines += ['', 'A higher HOTA with negative common-birth net counts does not imply fabricated metrics: changed births and unknown support '
              'are excluded from that local reference. It does mean those local counts cannot explain the global improvement as verified physical recovery. '
              f"Exact pinned CLEAR observation campaign status: {events['status']}. Removed baseline switch events and new switch events are "
              'reported separately; a shifted switch frame is not necessarily a physical identity correction.', '',
              '## 12. Complete online result and numerical parity', '',
              'The representative canonical riskMLP deployment completed4,834 raw frames: HOTA53.1035,AssA38.3609,DetA73.5676, '
              'IDF162.1078,MOTA93.6700,IDSW87. BT005 HOTA59.9983/IDF174.5622/IDSW21; SN013 HOTA48.1678/IDF153.9370/IDSW66. '
              'It preserves ordinary cached current frame/ID outputs, so no deployment identity gain is demonstrated. '
              'BT005 is fully byte-equal to cache. SN013 has7 numeric rows only at2999/3000, at most0.338px/score0.000952, '
              'with equal IDs and memory/quota counters. Complete byte equality must not be claimed. The identity-coordinate candidate '
              'was not retained for raw online deployment because its matched seed1/2 improvements were small and IDSW increased; a raw '
              'candidate deployment remains an explicit unperformed follow-up rather than a claimed result.', '',
              '## 13. Parameters, latency and countable operations', '',
              'Complete decode/causal IR/detector/NMS/RGB crop+encoding/GRU/risk/lifecycle/output costs3907.5533seconds, '
              '1.2371FPS,p50739.5ms,p951116.8ms. Parent/worker peaks are652,593,664/1,947,126,784bytes (separate process peaks). '
              'Parameters: detector186,199,381;RGB24,607,808;GRU231,040;riskMLP1,793;residual4,389. Shared-load timing is not a controlled speedup claim.', '']
    padding = report['affine_ops']['result']['current_padding']
    lines += [f"Exact dense affine count: detector single item {report['affine_ops']['result']['detector_one_item']['affine_flops_without_bias']:,} FLOPs; "
              f"actual eight-item frame padding {padding['detector_affine_flops_per_frame']:,}; "
              f"RGB single crop {report['affine_ops']['result']['rgb_one_crop']['affine_flops_without_bias']:,}; "
              f"actual128-crop padded chunk {padding['rgb_affine_flops_per_padded_chunk']:,}. "
              'Two FLOPs/MAC; excludes BN/activation/pooling/fusion, codec/crops/transfer/NMS, GRU/risk features and matching. '
              'Risk affine cost is3,456FLOPs/scored edge. This is component accounting, not a complete system FLOPs value.', '',
              '## 14. Teacher, risk and branching novelty audit', '',
              'EMA necessity: unproven; same-input noEMA performs nearly identically. Candidate risk necessity: unproven; margin and '
              'ordinary thresholds explain much of the benefit. K2 necessity: unproven; C is marginal and lowers AB. '
              'AB complementarity was tested with repeated initialization, noEMA and fixed-margin controls; the identity-coordinate expansion '
              'does not establish a deployment-level candidate gain. Three-seed baseline variance does not establish scene generalization. '
              'No CVPR-ready novelty or guaranteed recovery claim follows from these results.', '',
              '## 15. Open scientific questions and unperformed experiments', '',
              'Completed: AB coupled initialization1/2, noEMA and fixed-margin controls, full-trajectory no-calibration equivalence, '
              'and identity-coordinate/producer/module seed expansion with matching baselines. '
              'Unperformed follow-ups include raw online deployment of the unretained candidate, independent external data, faithful full MOTIP/GOT reproduction, larger local hypothesis groups, learned branch selection, '
              'joint competing-track training and detector/encoder retraining remain unperformed. '
              'No claim of general impossibility, information-theoretic bound or exhausted possible architectures is made.', '',
              '## 16. External data and primary research references', '',
              'MOTIP, GOT-JEPA, V-JEPA2, HybridSORT, BoT-SORT and uncertainty/multiple-trajectory prior art were reviewed through primary sources '
              'in PHASE_VI_RELATED_WORK_AND_EXTERNAL_DATA.md. FishTrack23, Fraunhofer, FISHTRAC and SFISHTRACK have unresolved '
              'license/access/cadence/overlap checks; none is claimed as a new blind test. Synthetic IR is computed from RGB and is not an independent sensor.', '',
              '## 17. Reproducibility and recommended current practice', '',
              'Preserve the strong ordinary GRU/current-measurement baseline for deployment. The measured candidate families remain research '
              'diagnostics and should not replace the baseline; the representative risk method also preserves ordinary current frame/ID outputs. '
              'Replay frozen plans with cfp_run_residual_relation_campaign.py, cfp_run_oof_candidate_relation_campaign.py, '
              'cfp_run_oof_combination_campaign.py and cfp_run_oof_combination_extension.py; aggregate/evaluate with '
              'cfp_evaluate_oof_families.py. The event runner verifies original CLEAR counts and known-only paired labels. '
              'Actual-resume tools and full online collector have independent source/checkpoint/output hashes.', '',
              '## 18. Engineering invariants and delivery', '',
              'Independent switches, empty/missing/unknown/nonfinite/future/history/gap and branch-isolation guards are retained. '
              'Current input must precede the write; only matched current measurements are emitted. Historical assets are never overwritten, '
              'new outputs are immutable,>=40GiB reserve is maintained and only reserved idleGPU0/1 are used. '
              'The project suite last passed187tests; the additional paired-support and identity-evidence checks are recorded with the stage artifacts. '
              'The final live remote hash and branch are recorded in PHASE_VI_STATE.json after delivery verification. '
              'Weights, datasets, proposal features and large traces remain local.', '',
              '## 19. Hypothesis status and discriminating evidence', '',
              '| Hypothesis | Current evidence | Status |',
              '|---|---|---|',
              '| H0 reproducibility/causality |88 hashes, exact disabled outputs, actual resume, full online numerical audit, isolation guards |Supported in tested scope |',
              '| H1 recoverability taxonomy |Current candidate availability, accepted history, unknown labels and joint competition are distinct |Supported diagnostically; no universal recovery ceiling |',
              '| H2 candidate-conditioned risk |Ordinary margin/threshold controls explain benefits; absolute gates fail transfer |Learned context necessity unproven |',
              '| H3 predictive candidate relations |Repeated predictive residual improves vs CE-only; EMA adds no independent benefit; identity seed1/2 matched gains are small and IDSW increases |Scoped signal, not retained for deployment |',
              '| H4 alternative trajectory explanations |C is marginal and lowers AB; correct retained challengers rare in some scenes |Necessary contribution unproven |',
              '| H5 coverage/temporal alternatives |Geometry prior restoration helps; full graph/retained-cue oracles are scoped |Learned coverage/trajectory conditioning remains open |', '',
              '## 20. Per-video matrix', '']
    for seq in (s for s in sequences if s != 'COMBINED_SEQ'):
        lines += ['', '**' + seq + '**', '']
        lines += table([(kind, entry['metrics'][seq]) for kind, entry in sorted(combo.items())])
        if identity_complete:
            lines += ['', '*Identity-coordinate seed families (matching seed baseline):*', '']
            names = ('identity_baseline_seed1', 'ordinary_B_identity_seed1', 'predictive_B_identity_seed1',
                     'margin_only_identity_seed1', 'predictive_margin_identity_seed1',
                     'identity_baseline_seed2', 'ordinary_B_identity_seed2', 'predictive_B_identity_seed2',
                     'margin_only_identity_seed2', 'predictive_margin_identity_seed2')
            lines += table([(name, identity_result['families'][name]['metrics'][seq]) for name in names])
    lines += ['', '## 21. Exact reproduction entry points', '',
              'Use the preserved runtime and local frozen artifacts under /data3/liuyeqiang/IR-MFT_assets/cfp. '
              'The repository alone does not contain dataset/weights/features. The pinned TrackEval is '
              '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval.', '',
              '```bash',
              'MFT26IR_ROOT=/data3/liuyeqiang/datasets/mft26-ir-causal-v1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python -m pytest -q tests',
              'OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_collect_phase_vi_synthesis.py',
              'OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /data3/liuyeqiang/.venvs/cfp-fishmot/bin/python tools/cfp_plot_phase_vi_synthesis.py',
              '```', '',
              'Campaign commands and immutable outputs are recorded in each measured official JSON. '
              'Frozen plans include phase_vi_oof_combination_matrix_v1.json, phase_vi_residual_discriminating_training_v1.json, '
              'phase_vi_oof_AB_discriminating_extension_v1.json and phase_vi_identity_seed_bootstrap_training_v1.json. '
              'Use new output paths for repetitions; do not overwrite historical predictions/checkpoints.', '',
              '## 22. Delivery and remaining conclusion boundary', '',
              'Latest verified delivery is recorded in PHASE_VI_STATE.json and local phase6/github_delivery evidence. '
              'Current new evidence is only claimed published after a live remote hash check. '
              '[Scientific summary figure](figures/phase_vi_synthesis.svg) depicts completed matrix/model-initialization controls, '
              'not final identity-pipeline generalization.', '',
              'This is a scoped final scientific conclusion: the candidate mechanisms remain valuable diagnostics, but the available three-seed evidence does not justify replacing the strong ordinary baseline. External and raw candidate-deployment follow-ups remain explicit.', '']
    (D / 'PHASE_VI_FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k: v['COMBINED_SEQ']['HOTA'] for k, v in report['model_seed_statistics'].items()}))


if __name__ == '__main__':
    main()
