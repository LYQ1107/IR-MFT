#!/usr/bin/env python3
"""Summarize all completed registered recovery removals, never partial curves."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root']); docs = ROOT/'docs/fusion_research'
    state_path = work/'correlation_recoverability_staged_dev_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 24
    assert all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    epochs = [6, 12, 18, 24]; variants = ['full', 'aligned', 'no_curriculum', 'blanket_counterfactual']
    histories = {}
    for variant in variants:
        path = work/'correlation_recoverability_fit'/f'ir_{variant}_s0/complete.json'
        fit = json.loads(path.read_text())
        assert fit['epochs'] == 24 and sum(r['steps'] for r in fit['history']) == 6912
        histories[variant] = [r['sample_digest'] for r in fit['history']]
    assert all(x == histories['full'] for x in histories.values())
    rows = []; curves = {}
    for variant in variants:
        stage = 'correlation_recoverability_pilot_dev' if variant == 'full' else 'correlation_recoverability_staged_dev'
        for policy in ['raw', 'priority']:
            curve = []
            for epoch in epochs:
                name = f'CGR_ir_{variant}_s0_e{epoch:02d}' + ('_raw' if policy == 'raw' else '')
                path = work/stage/name/'metrics.json'; value = json.loads(path.read_text())
                assert value['status'] == 'COMPLETE' and value['manifest']['frames'] == 4834 and value['manifest']['split'] == 'dev'
                row = dict(id=name, variant=variant, policy=policy, epoch=epoch, metrics=value['trackeval'], metrics_sha256=sha256(path))
                rows.append(row); curve.append(row)
            curves[(variant, policy)] = curve
    full_best = max(r['metrics']['HOTA'] for r in rows if r['variant'] == 'full')
    comparisons = []
    for variant in variants[1:]:
        best = max((r for r in rows if r['variant'] == variant), key=lambda r: r['metrics']['HOTA'])
        comparisons.append(dict(removal=variant, best_id=best['id'], best_HOTA=best['metrics']['HOTA'],
                                full_minus_removal_HOTA=full_best-best['metrics']['HOTA']))
    report = dict(status='COMPLETE_ALL24_EPOCH_REMOVALS_AND32_FULL_DEV_POLICIES', rows=rows,
                  comparisons=comparisons, all24_Train_sample_digests_match=True, state_sha256=sha256(state_path),
                  scope='Seed0 only. All4 checkpoints and raw/priority policies. Dev-only best-grid removal comparisons, not three-seed contributions or Test accuracy.',
                  supported_multimodal_innovations=0, Test_read=False, goal_met=False)
    dump(docs/'PERCEPTUAL_MFT_20261123_RECOVERABILITY_REMOVALS.json', report)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    labels = ['Complete recovery repair', 'No local offset search', 'No regional weakening', 'No validity exclusion']
    for i, policy in enumerate(['raw', 'priority']):
        for j, metric in enumerate(['HOTA', 'AssA']):
            ax = axes[i, j]
            for variant, label in zip(variants, labels):
                ax.plot(epochs, [r['metrics'][metric] for r in curves[(variant, policy)]], 'o-', label=label, markersize=4)
            if metric == 'HOTA':
                ax.axhline(49.965, color='0.4', linestyle='--', label='Earlier strong gray HOTA')
            ax.set_title(f'{policy}: {metric}'); ax.set_xticks(epochs); ax.set_xlabel('Train cache epoch'); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('All full-Dev recovery removals, seed 0; no Test selection')
    path = docs/'PERCEPTUAL_MFT_20261123_RECOVERABILITY_REMOVALS.png'
    fig.savefig(path, dpi=125); plt.close(fig); assert path.stat().st_size <= 512*1024
    lines = ['# Recovery removals: all32 registered policies', '', report['scope'], '',
             '![Complete curves](PERCEPTUAL_MFT_20261123_RECOVERABILITY_REMOVALS.png)', '',
             '| Removed mechanism | Best removal HOTA | Full minus removal |', '| --- | ---: | ---: |']
    for row in comparisons:
        lines.append(f"| {row['removal']} | {row['best_HOTA']:.3f} | {row['full_minus_removal_HOTA']:+.3f} |")
    lines += ['', 'Full recovery best HOTA is'+f'{full_best:.3f}'+'. Earlier parent IR49.771 and strongest gray49.965 remain required controls.',
              'Negative or neutral removal deltas do not support a contribution. No new Test result or established innovation is asserted.']
    (docs/'PERCEPTUAL_MFT_20261123_RECOVERABILITY_REMOVALS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(comparisons=comparisons, Test_read=False)))


if __name__ == '__main__':
    main()
