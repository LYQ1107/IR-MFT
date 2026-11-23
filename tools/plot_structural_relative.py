#!/usr/bin/env python3
"""Complete Dev curves for training adequacy, removals and IR intervention."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/fusion_research'


def main():
    stem=DOCS/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS'
    report=json.loads(stem.with_suffix('.json').read_text())
    assert report['status']in ['COMPLETE_CURRENT_STRUCTURAL_RELATIVE_STAGES','COMPLETE_STRUCTURAL_RELATIVE_ANALYSIS']
    main_rows=report['all_full_curves'];epochs=[6,12,18,24]
    colors={'rgb':'#277da8','gray':'#a17e00','ir':'#aa4239'}
    names={'rgb':'RGB repeat','gray':'Grayscale RGB','ir':'Actual new pseudo-IR'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    for ax,policy in zip(axes[0],['raw','priority']):
        for cue in ['rgb','gray','ir']:
            rows={r['config']['epoch']:r for r in main_rows
                  if r['config'].get('cue')==cue and not r['config'].get('baseline')
                  and r['config']['output_policy']==policy}
            assert set(rows)==set(epochs)
            ax.plot(epochs,[rows[e]['metrics']['HOTA']for e in epochs],'o-',color=colors[cue],label=names[cue])
        ax.axhline(report['baseline']['metrics']['HOTA'],color='.5',ls='--',lw=1,label='Original RGB')
        ax.set_title('Full controls: '+policy);ax.legend(fontsize=8)
    policy=report['selected_IR_policy'];ax=axes[1,0]
    full={r['config']['epoch']:r for r in main_rows if r['id'].startswith('SR_ir_full_s0_')
          and r['config']['output_policy']==policy}
    if report['pending']['module_removals']:
        ax.text(.5,.5,'Module-removal stage pending',ha='center',va='center',transform=ax.transAxes)
    else:
        removed=json.loads((DOCS/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_REMOVALS_DEV.json').read_text())['rows']
        assert len(removed)==24
        ax.plot(epochs,[full[e]['metrics']['HOTA']for e in epochs],'o-',color=colors['ir'],label='Full IR')
        for variant,label,color in [('no_structure','All structure removed','#577590'),
                                    ('no_cycle','Cycle check removed','#9b60a8'),
                                    ('no_centering','Centering removed','#809131')]:
            rows={r['config']['epoch']:r for r in removed if r['config']['variant']==variant
                  and r['config']['output_policy']==policy}
            assert set(rows)==set(epochs)
            ax.plot(epochs,[rows[e]['metrics']['HOTA']for e in epochs],'o--',label=label,color=color)
        ax.legend(fontsize=8)
    ax.set_title('IR removals at full-model selected policy: '+policy)
    ax=axes[1,1]
    if report['pending']['IR_interventions']:
        ax.text(.5,.5,'IR-intervention stage pending',ha='center',va='center',transform=ax.transAxes)
    else:
        selected=[r for r in report['IR_interventions']if r['output_policy']==policy]
        roll={r['epoch']:r for r in selected if r['intervention']=='IR_image_spatial_roll'}
        off=next(r for r in selected if r['intervention']=='unavailable')
        assert set(roll)==set(epochs)
        ax.plot(epochs,[full[e]['metrics']['HOTA']for e in epochs],'o-',color=colors['ir'],label='Correct current IR')
        ax.plot(epochs,[roll[e]['row']['metrics']['HOTA']for e in epochs],'o--',color='#9b60a8',label='Quarter-image IR roll')
        ax.axhline(off['row']['metrics']['HOTA'],color='.5',ls='--',label='IR unavailable at epoch24')
        ax.legend(fontsize=8)
    ax.set_title('IR interventions at selected policy: '+policy)
    for ax in axes.flat:
        ax.set_xticks(epochs);ax.set_xlabel('Passes over fixed1152-frame Train cache')
        ax.set_ylabel('Official combined Dev HOTA');ax.grid(alpha=.2)
    fig.suptitle('Fixed seed0; full4834-frame Dev, not Test\n'
                 'All6fits:145600 adapter parameters, equal24E/6912-update budget; no established contribution claim',fontsize=11)
    path=stem.with_suffix('.png');fig.savefig(path,dpi=125);plt.close(fig)
    assert path.stat().st_size<=512*1024
    print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,Test_read=False)))


if __name__=='__main__':
    main()
