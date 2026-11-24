#!/usr/bin/env python3
"""Read all50 finished Dev cases together; freeze full-Train choices without Test."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def select_curve(rows, cue, components):
    eligible = [r for r in rows if not r['config'].get('baseline') and
                r['config']['cue'] == cue and r['config']['components'] == components and
                not r['config'].get('intervention')]
    if not eligible:
        raise ValueError('No complete declared curve')
    return min(eligible, key=lambda row: (-row['metrics']['HOTA'], row['config']['epoch'],
                                        row['config']['policy'] != 'raw'))


def paired_row(rows, bits, primary):
    candidates = [r for r in rows if not r['config'].get('baseline') and
        r['config']['cue'] == 'ir' and r['config']['components'] == bits and
        not r['config'].get('intervention') and
        r['config']['epoch'] == primary['config']['epoch'] and r['config']['policy'] == primary['config']['policy']]
    if len(candidates) != 1:
        raise ValueError('Require exact same-checkpoint-budget and same-policy removal')
    return candidates[0]


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    state_path = work / 'structural_full_train_dev_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 50
    assert all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    spec_path = ROOT / 'configs/perceptual_structural_fulltrain_dev_20261123.json'
    assert sha256(spec_path) == state['specification_sha256']
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    rows = []
    for name, job in state['jobs'].items():
        folder = work / 'structural_full_train_dev' / name
        path = folder / 'metrics.json'
        data = json.loads(path.read_text())
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['frames'] == 4834
        assert manifest['split'] == 'dev' and manifest['sequences'] == protocol['dev']
        assert manifest['config_sha256'] == job['sha256'] == sha256(job['config'])
        assert manifest['config']['full_train_study_sha256'] == sha256(study_path)
        csv_path = folder / 'trackeval_output/pedestrian_detailed.csv'
        per = detailed_metrics(csv_path, protocol['dev'], data['trackeval'])
        assert sum(r['CLR_Frames'] for r in per.values()) == 4834
        # Full training input histories remain in the separate training ledger,
        # not duplicated for every sequence and case in this public metric table.
        weights = {s: info.get('checkpoint_sha256') for s, info in manifest['model_sources'].items()}
        assert not weights or len(set(weights.values())) == 1
        rows.append(dict(id=name, config=manifest['config'], metrics=data['trackeval'], per_sequence=per,
            coverage=manifest['coverage'], proposal_sources=manifest['sources'],
            checkpoint_sha256=next(iter(weights.values())) if weights else None,
            metrics_sha256=sha256(path), official_detailed_csv_sha256=sha256(csv_path),
            source_versions=manifest['source_versions']))
    by_id = {r['id']: r for r in rows}
    primary = select_curve(rows, 'ir', '111')
    controls = {cue: select_curve(rows, cue, '111') for cue in ['rgb', 'gray']}
    canonical, fresh = by_id['FT_source_RGB_canonical'], by_id['FT_source_RGB_fresh']
    removals = []
    for mechanism, bits in [('S', '011'), ('C', '101'), ('R', '110')]:
        paired, best = paired_row(rows, bits, primary), select_curve(rows, 'ir', bits)
        removals.append(dict(mechanism=mechanism, components=bits, paired=paired['id'], Dev_best=best['id'],
            primary_minus_paired_HOTA=primary['metrics']['HOTA']-paired['metrics']['HOTA'],
            primary_minus_Dev_best_HOTA=primary['metrics']['HOTA']-best['metrics']['HOTA']))
    interventions = {}
    for name in ['IR_image_spatial_roll', 'unavailable']:
        selected = [r for r in rows if not r['config'].get('baseline') and r['config'].get('intervention') == name and
            r['config']['epoch'] == primary['config']['epoch'] and r['config']['policy'] == primary['config']['policy']]
        assert len(selected) == 1
        interventions[name] = selected[0]['id']
    off = by_id[interventions['unavailable']]
    expected_RGB = fresh if primary['config']['policy'] == 'raw' else canonical
    assert off['metrics'] == expected_RGB['metrics']
    assert off['official_detailed_csv_sha256'] == expected_RGB['official_detailed_csv_sha256']
    assert all(off['coverage'][seq]['output_sha256'] == expected_RGB['coverage'][seq]['output_sha256']
               for seq in protocol['dev'])
    strongest = max(controls.values(), key=lambda r: r['metrics']['HOTA'])
    report = dict(status='COMPLETE_ALL50_FULL_TRAIN_DEV_AND_FROZEN_CHOICES', rows=rows,
        primary=primary['id'], primary_HOTA=primary['metrics']['HOTA'], primary_config=primary['config'],
        controls={cue: row['id'] for cue, row in controls.items()},
        primary_minus_canonical_RGB_HOTA=primary['metrics']['HOTA']-canonical['metrics']['HOTA'],
        primary_minus_strongest_matched_RGBgray_HOTA=primary['metrics']['HOTA']-strongest['metrics']['HOTA'],
        removals=removals, interventions=interventions,
        image_roll_change_HOTA=by_id[interventions['IR_image_spatial_roll']]['metrics']['HOTA']-primary['metrics']['HOTA'],
        off_tracks_metrics_CSV_exact_to_policy_RGB=True,
        selection_rule=study['selection'], full_train_study_sha256=sha256(study_path),
        completed_Dev_state_sha256=sha256(state_path), specification_sha256=sha256(spec_path),
        source_proofs=state['full_proofs'], fixed_seed=0, additional_seed_work=False,
        mandatory_Test=True, Test_accuracy_used_for_selection=False,
        scope='All six configurations trained three complete Train traversals/20736updates each. '
              'All50complete4834Dev cases retained. New fullTrain choices use Dev only; '
              'previous historical Test exposure and pretraining Dev exposure remain disclosed. '
              'Training coverage, sampling weights and budget change together, not separate innovation claims.',
        goal_met=False, supported_innovations=0)
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS'
    assert not stem.with_suffix('.json').exists(), 'Preserve the choice report used for Test freezing'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Complete full-Train Dev family and frozen choices', '', report['scope'], '',
        '| Case | HOTA | DetA | AssA | IDF1 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + row['id'] + ' | ' + ' | '.join(f'{row["metrics"][k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1']) + ' |')
    lines += ['', f"Frozen primary: {primary['id']}; HOTA {primary['metrics']['HOTA']:.3f}.", '',
        f"Versus original RGB: {report['primary_minus_canonical_RGB_HOTA']:+.3f}; versus strongest matched RGB/gray: "
        f"{report['primary_minus_strongest_matched_RGBgray_HOTA']:+.3f}.", '',
        '| Removal | Same epoch/policy | Dev-best removal |', '| --- | ---: | ---: |']
    for row in removals:
        lines.append(f"| {row['mechanism']} | {row['primary_minus_paired_HOTA']:+.3f} | {row['primary_minus_Dev_best_HOTA']:+.3f} |")
    lines += ['', study['selection'], '', 'All formal configurations require full7Test even if Dev is negative. No goal completion or innovation is established here.']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], primary=report['primary'], HOTA=report['primary_HOTA'])))


if __name__ == '__main__':
    main()
