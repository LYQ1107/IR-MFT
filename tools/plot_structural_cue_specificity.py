#!/usr/bin/env python3
"""Plot all registered Train domains and cues; no tracking accuracy or selection."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.diagnose_structural_train_mechanism import read,sha


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    docs=ROOT/'docs/fusion_research';stem=docs/'PERCEPTUAL_MFT_20261125_TRAIN_CUE_SPECIFICITY'
    value=read(stem.with_suffix('.json'));execution=read(Path(str(stem)+'_EXECUTION.json'))
    assert sha(stem.with_suffix('.json'))==execution['full_result_sha256']
    assert value['status']=='COMPLETE_TRAIN_CURRENT_PAST_CUE_SPECIFICITY_NOT_TRACKING_EFFICACY'
    assert not value['Dev_read'] and not value['Test_read'] and not value['new_tracking_metrics']
    spec=read(ROOT/'configs/perceptual_train_cue_specificity_20261125.json')
    sequences=[s['sequence'] for s in spec['samples']];cases=spec['cases'];assert len(sequences)==6 and len(cases)==3
    outputs=[stem.with_suffix(ext) for ext in ['.png','.pdf']]
    receipt=Path(str(stem)+'_FIGURE.json');assert not receipt.exists() and not any(p.exists() for p in outputs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(9,6.5),sharex=True,constrained_layout=True)
    colors={'rgb':'#0072B2','gray':'#777777','ir':'#D55E00'};locations=np.arange(6);width=.24
    for index,case in enumerate(cases):
        supports=[next(r for r in value['aggregated_support'] if r['case']==case['id'] and r['scope']==s and r['threshold']=='0.6') for s in sequences]
        contrast=[next(r['current_minus_past_foreground_ranking_AUC'] for r in value['contrast_AUC'] if r['case']==case['id'] and r['scope']==s) for s in sequences]
        assert all(v is not None for v in contrast)
        position=locations+(index-1)*width
        axes[0].bar(position,[s['current']-s['past'] for s in supports],width,color=colors[case['cue']],label=case['cue'].upper())
        axes[1].bar(position,contrast,width,color=colors[case['cue']])
    axes[0].axhline(0,color='black',linewidth=.8);axes[0].set_ylabel('Dense GT support change\n(current minus past cue; count)')
    axes[0].legend(frameon=False,ncol=3);axes[0].set_title('Train-only cue specificity: all six videos and three matched adapters')
    axes[1].axhline(.5,color='black',linewidth=.8,linestyle='--');axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Foreground versus background\ncontrast ranking AUC')
    axes[1].set_xticks(locations,sequences);axes[1].set_xlabel('Fixed Train samples; not held-out tracking efficacy')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for path in outputs:
        fig.savefig(path,dpi=180,bbox_inches='tight');assert path.stat().st_size<2**20
    plt.close(fig)
    receipt.write_text(json.dumps(dict(source_result_sha256=sha(stem.with_suffix('.json')),plotter_sha256=sha(__file__),
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in outputs},scope='All18cue/domainrows;denseTrainanchorsupportandcontrastAUC,notHOTA/AP/heldoutefficacy.',
        supported_innovations=0,goal_met=False),indent=2)+'\n')
    print(json.dumps(dict(outputs=[str(p) for p in outputs],new_tracking_metrics=False)),flush=True)


if __name__=='__main__':main()
