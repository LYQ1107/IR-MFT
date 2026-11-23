#!/usr/bin/env python3
"""Summarize the complete registered family, never partial or best-seed results."""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'
FIELDS = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']


def aggregate(rows):
    assert len(rows) == 3
    return dict(ids=[row['id'] for row in rows],
        mean={key: mean(row['metrics'][key] for row in rows) for key in FIELDS},
        sample_std={key: stdev(row['metrics'][key] for row in rows) for key in FIELDS},
        per_seed=[dict(id=row['id'], metrics=row['metrics']) for row in rows],
        per_sequence={seq: {key: mean(row['per_sequence'][seq][key] for row in rows)
                          for key in FIELDS} for seq in ['BT-005', 'SN-013']})


def main():
    work = Path(json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    state = json.loads((work / 'correlation_geometry_dev_state.json').read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 84
    assert all(row['status'] == 'COMPLETE' for row in state['jobs'].values())
    assert len(state['full_anchor_and_unavailable_proofs']) == 2
    names = ['CORRELATION_PILOT_DEV', 'TEMPORAL_GEOMETRY_DEV', 'CORRELATION_GEOMETRY_DEV']
    reports = {name: json.loads((DOCS / f'PERCEPTUAL_MFT_20261123_{name}.json').read_text()) for name in names}
    previous = {row['id']: row for row in reports[names[0]]['rows']}
    temporal = {row['id']: row for row in reports[names[1]]['rows']}
    current = {row['id']: row for row in reports[names[2]]['rows']}
    assert len(previous) == 25 and len(temporal) == 36 and len(current) == 84
    assert set(current) == set(state['jobs'])
    epochs = dict(rgb=18, gray=18, ir=24)

    def curve(cue, seed, epoch, policy):
        key = f'{"CGA" if seed == 0 else "CG3"}_{cue}_full_s{seed}_e{epoch:02d}'
        key += '_raw' if policy == 'raw' else ''
        return (previous if seed == 0 else current)[key]

    def role_row(cue, seed, role, policy):
        if seed == 0:
            key = f'TC_{cue}_{"none" if role == "full" else role}' + ('_raw' if policy == 'raw' else '')
            return temporal[key]
        key = f'CG3_{cue}_full_s{seed}_e{epochs[cue]:02d}'
        key += '' if role == 'full' else '_'+role
        key += '_raw' if policy == 'raw' else ''
        return current[key]

    curves = {}
    common = {}
    descriptive_best = {}
    for cue in ['rgb', 'gray', 'ir']:
        entries = []
        for epoch in [6, 12, 18, 24]:
            for policy in ['priority', 'raw']:
                entries.append(dict(epoch=epoch, policy=policy,
                    **aggregate([curve(cue, seed, epoch, policy) for seed in range(3)])))
        curves[cue] = entries
        common[cue] = max(entries, key=lambda row: row['mean']['HOTA'])
        descriptive_best[cue] = [max([curve(cue, seed, epoch, policy)
            for epoch in [6, 12, 18, 24] for policy in ['priority', 'raw']],
            key=lambda row: row['metrics']['HOTA'])['id'] for seed in range(3)]
    roles = {}
    for role in ['full', 'rgb_boxes', 'rgb_scores']:
        for policy in ['priority', 'raw']:
            key = role+'_'+policy
            groups = {cue: aggregate([role_row(cue, seed, role, policy) for seed in range(3)])
                      for cue in ['rgb', 'gray', 'ir']}
            deltas = {cue: [role_row('ir', seed, role, policy)['metrics']['HOTA']
                           - role_row(cue, seed, role, policy)['metrics']['HOTA'] for seed in range(3)]
                      for cue in ['rgb', 'gray']}
            roles[key] = dict(groups=groups, paired_IR_minus_control_HOTA=deltas,
                paired_delta_mean={cue: mean(values) for cue, values in deltas.items()})
    interventions = []
    for intervention in ['unavailable', 'spatial_roll']:
        for role in ['full', 'rgb_boxes', 'rgb_scores']:
            for policy in ['priority', 'raw']:
                key = f'CG3_IR_{intervention}' + ('' if role == 'full' else '_'+role)
                key += '_raw' if policy == 'raw' else ''
                row = current[key]; parent = role_row('ir', 0, role, policy)
                interventions.append(dict(id=key, intervention=intervention, head_role=role,
                    policy=policy, parent=parent['id'], metrics=row['metrics'],
                    per_sequence=row['per_sequence'],
                    HOTA_change_from_correct_IR=row['metrics']['HOTA']-parent['metrics']['HOTA']))
    report = dict(status='COMPLETE_3_SEED_DEV_REPLICATION_NOT_TEST',
        scope='All4834 Dev frames and all registered84 new cases. Seed means average three '
              'official combined-video scores, not the two per-sequence scores. '
              'Sample standard deviation over3 seeds is descriptive, not a significance test.',
        source_report_sha256={name: hashlib.sha256((DOCS / f'PERCEPTUAL_MFT_20261123_{name}.json').read_bytes()).hexdigest()
                              for name in names}, fullcue_all_common_curves=curves,
        fullcue_common_epoch_policy_by_mean_HOTA=common,
        per_seed_best_grid_ids_descriptive_only=descriptive_best,
        fixed_source_seed0_epochs=epochs, fixed_epoch_head_roles=roles,
        seed0_IR_feature_interventions=interventions, supported_multimodal_innovations=0,
        goal_met=False, Test_read=False,
        selection='Common fullcue epoch/policy chosen by3-seed mean Dev. Head-role evaluation '
                  'uses the preregistered prior seed0 epochs across every seed, with no new '
                  'per-seed role epoch selection. Retain every failed control. '
                  'NMS/emission factorial and original-parent removals are still separate pending evidence.')
    stem = DOCS / 'PERCEPTUAL_MFT_20261123_CORRELATION_REPLICATION_ANALYSIS'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Complete three-seed correlation replication', '', report['scope'], '', report['selection'], '',
        '| Common full-cue selection | Epoch | Policy | HOTA mean ± sample SD | DetA | AssA | IDF1 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for cue, row in common.items():
        m = row['mean']; sd = row['sample_std']['HOTA']
        lines.append(f"| {cue} | {row['epoch']} | {row['policy']} | {m['HOTA']:.3f} ± {sd:.3f} | "
                     f"{m['DetA']:.3f} | {m['AssA']:.3f} | {m['IDF1']:.3f} |")
    lines += ['', '| Fixed-epoch role/policy | RGB mean HOTA | Gray | Actual IR | IR − RGB | IR − gray |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, row in roles.items():
        means = [row['groups'][cue]['mean']['HOTA'] for cue in ['rgb', 'gray', 'ir']]
        deltas = row['paired_delta_mean']
        lines.append('| '+key+' | '+' | '.join(f'{x:.3f}' for x in means)
                     +f" | {deltas['rgb']:+.3f} | {deltas['gray']:+.3f} |")
    lines += ['', '| Seed0 intervention | Role | Policy | HOTA | Change from correct IR |',
              '| --- | --- | --- | ---: | ---: |']
    for row in interventions:
        lines.append(f"| {row['intervention']} | {row['head_role']} | {row['policy']} | "
                     f"{row['metrics']['HOTA']:.3f} | {row['HOTA_change_from_correct_IR']:+.3f} |")
    lines += ['', 'These Dev values do not establish the >=+5 Test HOTA target or2–3 independent '
              'contributions. Full per-seed metrics, all common curves and sequence-level means '
              'are retained in the adjacent JSON. Historical Test exposure and previous RGB '
              'pretraining on present Dev scenes remain limitations.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(stem.with_suffix('.json')), common_HOTA={cue:row['mean']['HOTA']
        for cue,row in common.items()}, fixed_raw_IR_HOTA=roles['rgb_boxes_raw']['groups']['ir']['mean']['HOTA'],
        paired_raw_IR_minus_gray=roles['rgb_boxes_raw']['paired_IR_minus_control_HOTA']['gray'],
        Test_read=False)))


if __name__ == '__main__':
    main()
