#!/usr/bin/env python3
"""Read-only inventory of completed seed0 fits and direct replay dependencies.

Indirect encoder/feature dependencies are deliberately not treated as a proof
of complete Test coverage. Nothing is relabelled after a negative Dev result.
"""
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


FAMILIES = [
    'metric', 'association_metric', 'native_cue', 'native_neck_metric',
    'native_query', 'native_robust_query', 'native_neck_robust_query', 'native_neck_clean_query',
]


def paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from paths(item)
    elif isinstance(value, str) and value.startswith('/'):
        yield value


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    replays = []
    for path in sorted(work.glob('*/*/metrics.json')):
        data = json.loads(path.read_text())
        manifest = data.get('manifest', {})
        if data.get('status') != 'COMPLETE' or manifest.get('split') not in ['dev', 'test']:
            continue
        split = manifest['split']
        expected = protocol['dev' if split == 'dev' else 'historical_test']
        assert manifest['sequences'] == expected and manifest['frames'] == {'dev': 4834, 'test': 15985}[split]
        replays.append(dict(path=path, split=split, config=manifest['config'],
                            HOTA=data['trackeval']['HOTA']))
    rows, absent = [], []
    for family in FAMILIES:
        folder = work / family
        if not folder.exists():
            absent.append(family)
            continue
        for complete_path in sorted(folder.glob('*/complete.json')):
            complete = json.loads(complete_path.read_text())
            if complete.get('seed') != 0:
                continue
            assert complete['status'].startswith('COMPLETE')
            # Labels remain offline. Keep both historical spelling conventions.
            dev_rows = complete.get('Dev_fit_rows', complete.get('Dev_fitting_rows'))
            test_rows = complete.get('Test_fit_rows', complete.get('Test_fitting_rows'))
            assert dev_rows == test_rows == 0
            fit = complete_path.parent
            references = []
            for replay in replays:
                references_here = [p for p in paths(replay['config']) if p.startswith(str(fit) + '/')]
                if references_here:
                    references.append(dict(split=replay['split'], metrics_path=str(replay['path']),
                        HOTA=replay['HOTA'], checkpoint_paths=sorted(set(references_here)),
                        metrics_sha256=sha256(replay['path'])))
            rows.append(dict(family=family, fit=fit.name, complete_status=complete['status'],
                completed_epochs=complete['epochs'], fixed_seed=0, cue=complete.get('cue'),
                frontend=complete.get('frontend'), complete_path=str(complete_path),
                complete_sha256=sha256(complete_path), Dev_fit_rows=dev_rows, Test_fit_rows=test_rows,
                direct_complete_Dev_references=[r for r in references if r['split'] == 'dev'],
                direct_complete_Test_references=[r for r in references if r['split'] == 'test'],
                direct_Test_dependency_found=any(r['split'] == 'test' for r in references),
                complete_formal_Test_coverage_verified=False,
                caveat='Direct path inventory only. Encoders can be indirect dependencies of feature caches; '
                       'existence of any direct reference does not prove every required formal role is covered.'))
    report = dict(status='COMPLETED_SEED0_NONSTRUCTURAL_FIT_DIRECT_DEPENDENCY_INVENTORY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
        completed_seed0_fits=len(rows), family_folders_absent=absent,
        source_sha256=sha256(__file__), replays_inspected=len(replays),
        context='Fixed NovemberDiffV2IR campaign. Read only completed artifacts; no fitting, GPU work, '
                'new evaluation, threshold choice, Dev/Test primary selection or seed replication.',
        interpretation='These are existing trained candidates or representation controls. Negative Dev does '
            'not retroactively make a formal candidate diagnostic. This inventory is incomplete for indirect '
            'dependencies and other registered families; it is not a global Test-coverage pass.',
        required_next='Trace feature/cache provenance and each preregistered formal role, freeze original '
            'complete-Dev choices, then backfill engineering-passed seed0 roles irrespective of Dev sign. '
            'Do not add seed1/2 replications. Do not count generic association calibration as visual fusion novelty.',
        current_fullTrain_unmodified=True, goal_met=False, supported_innovations=0)
    destination = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NONSTRUCTURAL_FIT_INVENTORY.json'
    assert not destination.exists()
    dump(destination, report)
    lines = ['# Existing nonstructural seed0 fits: direct dependency inventory', '', report['interpretation'], '',
        '| Family | Fit | Executed epochs | Complete Dev references | Complete Test references |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['family']} | {row['fit']} | {row['completed_epochs']} | "
                     f"{len(row['direct_complete_Dev_references'])} | {len(row['direct_complete_Test_references'])} |")
    lines += ['', report['context'], '', report['required_next'], '',
              'Counts describe direct checkpoint paths in complete replay configurations. A zero is not proof '
              'that an encoder was never used indirectly, and a positive count is not a complete formal-role audit.']
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], fits=len(rows), replays=len(replays),
        fits_with_direct_Test_reference=sum(r['direct_Test_dependency_found'] for r in rows),
        indirect_provenance_and_formal_role_audit_pending=True)))


if __name__ == '__main__':
    main()
