#!/usr/bin/env python3
"""Export every calibration/metric Dev control, including negative results."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);rows=[]
    for stage,cfgdir in [('repair2','perceptual_safe_20261122'),('metric_dev','perceptual_metric_dev_20261122'),('metric24_dev','perceptual_metric24_dev_20261122')]:
        for config in sorted((ROOT/'configs'/cfgdir).glob('*.json')):
            folder=w/stage/config.stem;p=folder/'metrics.json';v=json.loads(p.read_text());m=v['manifest']
            assert v['status']=='COMPLETE' and m['frames']==4834 and m['sequences']==protocol['dev']
            assert m['config_sha256']==sha256(config)
            per=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['dev'],v['trackeval'])
            assert sum(x['CLR_Frames'] for x in per.values())==4834
            rows.append(dict(id=config.stem,stage=stage,config=m['config'],metrics=v['trackeval'],per_sequence=per,
                             metrics_path=str(p),metrics_sha256=sha256(p),config_sha256=sha256(config),
                             inference_statistics=m['statistics'],replay_FPS=m['replay_FPS'],
                             evidence_tracker_source_sha256=m['evidence_tracker_source_sha256'],
                             adapter=m.get('evidence_adapter')))
    assert len(rows)==56
    stop=json.loads((w/'metric_epoch24_decision.json').read_text())
    histories={}
    for cue in ['rgb','ir']:
        for seed in range(3):
            p=w/'metric'/f'{cue}_s{seed}'/'history.json';history=json.loads(p.read_text())
            assert len(history)==24
            assert history[:18]==json.loads((p.parent/'stage18_history.json').read_text())
            histories[f'{cue}_s{seed}']=dict(history=history,sha256=sha256(p))
    report=dict(status='COMPLETE_DEV_CONTROLS_GOAL_NOT_MET',scope='20 graph-preserving calibration controls plus 36 matched metric Dev snapshots. No Test selection. These are ordinary controls, not established innovations.',
                plan_sha256=sha256(plan_path),rows=rows,common_stopping=stop,fit_history=histories,
                executed_epochs_each=24,selected_RGB_epochs=[18,24,18],selected_IR_epochs=[6,6,6],
                supported_innovations=0,formal_Test_status='PENDING_FROZEN_SELECTION')
    docs=ROOT/'docs/fusion_research';dump(docs/'PERCEPTUAL_MFT_20261122_CALIBRATION_METRIC.json',report)
    text=['# Complete calibration and ordinary metric controls','',report['scope'],'',
          'All six metric heads executed 24 epochs, with identical schedules and seeds 0/1/2 on physical GPUs 2/3. The common stopping rule compares best Dev through24 with best through12. All six improvements are at most 0.003 HOTA points, below the registered 0.2 threshold. No Test score entered stopping or selection. These are feature-head epochs, not detector fine-tuning epochs.','',
          'RGB selected epochs are 18/24/18; IR selected epochs are 6/6/6 because all its Dev HOTA snapshots tie. Earliest checkpoint wins ties. A selected early epoch does not mean the other epochs were skipped. The same selected RGB head is retained on RGB and new-IR proposal streams for a controlled comparison.','',
          '| Calibration candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['stage']!='repair2':continue
        m=row['metrics'];text.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f" | {int(m['IDSW'])} |")
    text+=['','Original RGB Dev HOTA is 47.579; new-IR Dev is 46.236. The strongest ordinary RGB calibration reaches 47.584 (+0.005), which is not meaningful evidence for a new multimodal contribution. No new-input calibration exceeds the original new-input control.','',
           '| Metric cue / frontend | Seed | HOTA e06 | e12 | e18 | e24 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for prefix in ['M_rgb_rgb','M_new_rgb','M_new_ir']:
        for seed in range(3):
            values=[next(r['metrics']['HOTA'] for r in rows if r['id']==f'{prefix}_s{seed}_e{epoch:02d}') for epoch in [6,12,18,24]]
            text.append(f'| {prefix} | {seed} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    text+=['','Training loss falls, but that does not establish tracking improvement. The graph-preserving backend has few decisions changed by these learned descriptors, and IR-guided appearance does not distinguish identities better on the complete Dev set.','',
           '![Training and Dev controls](PERCEPTUAL_MFT_20261122_CALIBRATION_METRIC.png)','',
           'Full official headline/per-sequence fields, source hashes, all six loss curves and the stopping decision are in the adjacent JSON. The historical Test was exposed before this campaign and the original RGB pretraining included present Dev videos; results must remain exploratory.']
    (docs/'PERCEPTUAL_MFT_20261122_CALIBRATION_METRIC.md').write_text('\n'.join(text)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for cue,color in [('rgb','#E69F00'),('ir','#0072B2')]:
        losses=np.array([[x['losses']['total'] for x in histories[f'{cue}_s{s}']['history']] for s in range(3)])
        axes[0].plot(range(1,25),losses.mean(0),color=color,label=f'{cue.upper()} cue, 3 seeds')
        axes[0].fill_between(range(1,25),losses.min(0),losses.max(0),color=color,alpha=.15)
    axes[0].set(title='Train-only metric loss',xlabel='Executed feature-training epochs',ylabel='Loss');axes[0].legend(fontsize=8)
    for ax,prefix,title,color in [(axes[1],'M_rgb_rgb','RGB frontend + RGB metric','#E69F00'),(axes[2],'M_new_ir','New-IR frontend + IR metric','#0072B2')]:
        for seed,style in zip(range(3),['-','--',':']):
            yy=[next(r['metrics']['HOTA'] for r in rows if r['id']==f'{prefix}_s{seed}_e{e:02d}') for e in [6,12,18,24]]
            ax.plot([6,12,18,24],yy,style,marker='o',color=color,label=f'Seed {seed}')
        ax.axhline(47.579 if prefix=='M_rgb_rgb' else 46.236,color='#555',linestyle='-.',label='Unchanged ByteTrack')
        ax.set(title=title,xlabel='Executed feature-training epochs',ylabel='Complete Dev HOTA (%)',ylim=(45.5,48));ax.set_xticks([6,12,18,24]);ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(docs/'PERCEPTUAL_MFT_20261122_CALIBRATION_METRIC.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status=report['status'],Dev_rows=len(rows),epochs_each=24,supported_innovations=0)))


if __name__=='__main__':main()
