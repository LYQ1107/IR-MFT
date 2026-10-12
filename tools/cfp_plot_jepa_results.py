#!/usr/bin/env python3
"""Small standalone scientific figures from measured official JSON only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfp.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--measured',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinary-reference',type=Path,default=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5/ris_v1_control_ordinary_gru_s0_w0.3_logrepair_v2_evaluation/metrics.json'))
    args=p.parse_args()
    data=json.loads(args.measured.read_text());args.output.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(args.ordinary_reference.read_text())['metrics']['COMBINED_SEQ']['HOTA']
    chosen={1:['ordinary_transformer','J3','J6','no_ema','ordinary_gru_error'],2:['future_J3','future_J6','future_no_ema','future_frozen_encoder'],3:['gru_latent_identity','gru_no_ema_latent_identity','transformer_J6','transformer_latent_identity','transformer_ordinary_latent_identity']}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for round_no,ax in enumerate(axes,1):
        for variant in chosen[round_no]:
            values=sorted((g['weight'],g['seeds']['0']['metrics']['COMBINED_SEQ']['HOTA']) for g in data['groups'].values() if g['round']==round_no and g['variant']==variant and '0' in g['seeds'])
            if values:ax.plot(*zip(*values),marker='o',label=variant)
        ax.axhline(baseline,color='black',linestyle='--',linewidth=1,label='Original GRU seed0 current boxes')
        ax.set_title(f'JEPA round {round_no} (seed0)');ax.set_xlabel('Appearance weight');ax.set_xticks([.1,.3,.5]);ax.grid(alpha=.25)
        ax.legend(fontsize=7,loc='lower center');ax.set_ylim(44,58)
    axes[0].set_ylabel('Official combined HOTA (%)')
    fig.suptitle('Fixed V3 proposals / shared CFP lifecycle / current measurement boxes')
    fig.tight_layout();fig.savefig(args.output/'jepa_round_tracking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for round_no,color in [(1,'tab:blue'),(2,'tab:orange'),(3,'tab:green')]:
        models=[m for m in data['models'].values() if m['guard']['seed']==0 and any(g['config_sha256']==m['guard']['config_sha256'] and g['round']==round_no for g in data['groups'].values())]
        for ema,marker in [(True,'o'),(False,'x')]:
            points=[m for m in models if bool(m['spec'].get('ema',False))==ema]
            if points:ax.scatter([100*m['diagnostics']['clean']['rank1'] for m in points],[100*m['diagnostics']['actual']['rank1'] for m in points],color=color,marker=marker,label=f'Round{round_no}, '+('EMA' if ema else 'ordinary/noEMA'))
    reference=next(m for m in data['models'].values() if m['guard']['variant']=='J3' and m['guard']['seed']==0)
    ax.axhline(100*reference['diagnostics']['actual']['mean_rank1'],color='black',linestyle='--',label='Actual-history mean seed0')
    ax.axvline(100*reference['diagnostics']['clean']['mean_rank1'],color='gray',linestyle=':',label='Clean-history mean seed0')
    ax.set(xlabel='Clean-history rank1 (%)',ylabel='Actual accepted-history rank1 (%)',xlim=(0,100),ylim=(0,100),title='SN015 training calibration: fixed-query transfer gap')
    ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(args.output/'jepa_clean_actual_transfer.png',dpi=160);plt.close(fig)
    write_json(args.output/'manifest.json',dict(measured_sha256=sha256(args.measured),source_sha256=sha256(__file__),ordinary_reference_sha256=sha256(args.ordinary_reference),figures={p.name:sha256(p) for p in args.output.glob('*.png')},scope='Static measured seed0 plots. Cached tracking HOTA isnotfullonline evaluation; calibration queryrank isnottracking IDrecovery. Shared frozen protocols, no extrapolation ornew metrics.'))


if __name__=='__main__':main()
