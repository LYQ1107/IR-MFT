#!/usr/bin/env python3
"""Scientific three-seed matched clean-history identity plot, outside Git."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfp.provenance import sha256,write_json,snapshot_sources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v1',type=Path,required=True);ap.add_argument('--v2',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.with_suffix('.png').exists():raise ValueError('New immutable figure required')
    reports=[json.loads(p.read_text()) for p in [a.v1,a.v2]]
    for r in reports:assert r['completed_fits']==27 and r['exact_input_digests_and_frozen_support']
    modes=['global','global_gru','local_attention','local_temporal','local_contrastive','no_ema','no_prediction','no_id','full']
    names=['Global','Global GRU','Local attention','Local temporal','Local contrastive','No EMA','No prediction *','No ID loss','Full JEPA']
    fig,ax=plt.subplots(figsize=(10.5,6.7));positions=np.arange(len(modes));colors=['#7996b6','#a05838']
    for j,r in enumerate(reports):
        values=r['summary']['SN-015']['gt']['detector']['25']['models'];means=[values[m]['retrieval']['rank1_percent']['mean'] for m in modes];sds=[values[m]['retrieval']['rank1_percent']['sample_sd'] for m in modes]
        ax.errorbar(means,positions+(j-.5)*.24,xerr=sds,fmt='o',capsize=3,color=colors[j],label=['V1: predicted representation','V2: past context representation'][j],markersize=6,linewidth=1.4)
    reference=reports[0]['summary']['SN-015']['gt']['detector']['25']['frozen_cls_fifo']['retrieval']['rank1_percent'];ax.axvline(reference,color='#525252',ls='--',lw=1.1,label=f'Frozen CLS FIFO ({reference:.2f}%)')
    ax.set_yticks(positions,names);ax.invert_yaxis();ax.set_xlim(20,78);ax.set_xlabel('Rank1 (%) — mean ± sample SD over training seeds 0, 1, 2');ax.grid(axis='x',alpha=.2);ax.set_title('Held-out SN-015: matched clean physical history → current detector candidates\nMinimum past gap 25 frames; identical supported queries and unknown-containing pools',fontsize=12,pad=13)
    ax.legend(loc='lower left',fontsize=9);fig.subplots_adjust(left=.19,right=.97,top=.84,bottom=.15)
    fig.text(.19,.04,'Supported queries: 564 / 793; * V2 no-prediction and local-temporal are exact aliases.\nNo-ID removes explicit ID loss but retains annotated temporal grouping. This is offline retrieval, not native MOT.',fontsize=9,color='#444444')
    for extension in ['png','svg','pdf']:fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
    plt.close(fig);write_json(a.output.with_suffix('.json'),dict(input_sha256=[sha256(p) for p in [a.v1,a.v2]],source=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),artifacts={e:sha256(a.output.with_suffix('.'+e)) for e in ['png','svg','pdf']},scope='Matched firstfoldK4 cleanphysical offline diagnostic; no native tracking/novelty/gain claim. Three seedSD not biological/query independence uncertainty.'))

if __name__=='__main__':main()
