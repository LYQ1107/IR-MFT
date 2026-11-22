#!/usr/bin/env python3
"""Export every registered point-screen row and its Dev-only diagnostics."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text());rows=[]
    assert json.loads((w/'points_dev_state.json').read_text())['status']=='COMPLETE'
    for path in sorted((ROOT/'configs/perceptual_points_dev_20261122').glob('*.json')):
        folder=w/'points_dev'/path.stem;p=folder/'metrics.json';v=json.loads(p.read_text());m=v['manifest']
        assert v['status']=='COMPLETE' and m['frames']==4834 and m['sequences']==protocol['dev']
        assert m['config_sha256']==sha256(path)
        per=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['dev'],v['trackeval'])
        assert sum(x['CLR_Frames'] for x in per.values())==4834
        rows.append(dict(id=path.stem,config=m['config'],metrics=v['trackeval'],per_sequence=per,
                         statistics=m['statistics'],config_sha256=sha256(path),metrics_sha256=sha256(p),
                         foreground_tracker_source_sha256=m['foreground_tracker_source_sha256'],
                         byte_tracker_source_sha256=m['byte_tracker_source_sha256'],sources=m['sources'],
                         replay_FPS=m['replay_FPS']))
    assert len(rows)==30
    for arm,reference in [('rgb','C00_rgb_ByteTrack'),('new','C01_new_ByteTrack')]:
        prior=json.loads((w/'screen'/reference/'metrics.json').read_text())['trackeval']
        actual=next(r['metrics'] for r in rows if r['id']==f'P_{arm}_00_zero')
        assert actual==prior
    diagnostic_path=w/'foreground_correspondence_diagnosis.json'
    report=dict(status='COMPLETE_POINT_DEV_SCREEN_GOAL_NOT_MET',rows=rows,
                scope='Complete4,834-frame Dev screen, not Test. Single-lag geometric point transport and multi-lag/co-visible memory controls. Ordinary point tracking and generic memory are not by themselves new multimodal contributions.',
                baseline_parity_all_official_fields=True,supported_multimodal_innovations=0,
                diagnosis=json.loads(diagnostic_path.read_text()),diagnosis_sha256=sha256(diagnostic_path),
                next_action='Keep RGB source points, restrict generated-IR geometry to conditional ownership increments, validate across past observations and test supported candidate births against global threshold reduction.')
    docs=ROOT/'docs/fusion_research';dump(docs/'PERCEPTUAL_MFT_20261122_POINT_SCREEN.json',report)
    text=['# Complete foreground point Dev screen','',report['scope'],'',
          '| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Duplicate vetoes |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        m=row['metrics'];veto=sum(s['duplicate_vetoes'] for s in row['statistics'].values())
        text.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f" | {int(m['IDSW'])} | {veto} |")
    text+=['','The zero-point wrappers exactly reproduce every official field for both original frontends. The strongest RGB-only point/memory control is 48.486 HOTA (+0.907 over47.579). That gain is generic tracking evidence, not evidence for pseudo-IR fusion.','',
           'On new-IR V3 proposals, RGB body points reach47.608 versus46.236 for unchanged ByteTrack, while hard joint IR/RGB point selection falls to43.894. Source-only and target-only controls show that restricting the source points is especially harmful. Multi-lag averaging also does not consistently beat the latest observation. Previously co-visible memory prevents two duplicate deletions in each corresponding full variant; aggregate effects still require complete Test validation.','',
           'The offline diagnostic maps proposals to Dev GT only after the independent point producer finishes. Adjacent-frame confident correspondence precision is about98.5%; the confidence>=0.7 subset is about99.2–99.4%. These are proposal evidence diagnostics, not official identity switches or proof of tracking improvement. Even a small number of wrong links can fragment a long identity.','',
           'The next registered family keeps RGB source points, uses IR only for an ownership increment under RGB uncertainty, requires agreement across earlier observations, and compares supported low-score candidate births with an ordinary global0.6 birth threshold. It also tests whether clear original motion associations should be preserved. No new Test result selects these rules.','',
           '![Paired point screen](PERCEPTUAL_MFT_20261122_POINT_SCREEN.png)','',
           'All headline/per-sequence fields, extraction paths/hashes and raw evidence diagnosis are retained in the adjacent JSON. Caches and input images remain outside Git.']
    (docs/'PERCEPTUAL_MFT_20261122_POINT_SCREEN.md').write_text('\n'.join(text)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharex=True)
    for ax,arm,title in zip(axes,['new','rgb'],['New-IR V3 proposals','RGB fine-tune proposals']):
        selected=[r for r in rows if r['config']['detection_arm']==arm]
        colors=['#777' if r['id'].endswith('zero') else '#E69F00' if r['config']['source_mode']=='rgb' and r['config']['target_mode']=='rgb' else '#0072B2' for r in selected]
        ax.barh(range(15),[r['metrics']['HOTA'] for r in selected],color=colors)
        ax.set_yticks(range(15),[r['id'].split('_',3)[-1] for r in selected],fontsize=8);ax.invert_yaxis()
        ax.axvline(47.579,color='#E69F00',linestyle='--',label='Original RGB ByteTrack')
        ax.set(title=title,xlabel='Complete Dev HOTA (%)',xlim=(0,55));ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8,loc='lower right')
    fig.tight_layout();fig.savefig(docs/'PERCEPTUAL_MFT_20261122_POINT_SCREEN.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status=report['status'],Dev_rows=30,supported_multimodal_innovations=0)))


if __name__=='__main__':main()
