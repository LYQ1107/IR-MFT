#!/usr/bin/env python3
"""Static scientific figure of actual frozen-query predictor diagnoses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cfp.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Scientific diagnostic figure outputs are immutable')
    models={};sources={}
    for architecture in ['gru','transformer']:
        for objective in ['predictive','association']:
            name=f'predictor_{architecture}_{objective}_clean_seed0';path=args.phase_root/(name+'_dev_diagnosis')/'manifest.json';m=json.loads(path.read_text())
            if m['status']!='COMPLETE':raise ValueError('All four measured diagnoses required')
            models[architecture+'_'+objective]=m;sources[str(path)]=sha256(path)
    exemplar=models['gru_predictive'];videos=list(exemplar['videos']);labels=['Last','EMA (8)','FIFO (8)','GRU predict','GRU assoc','Transformer predict','Transformer assoc']
    figure,axes=plt.subplots(2,len(videos),figsize=(12,7),sharey=True);values={}
    for column,video in enumerate(videos):
        for row,history in enumerate(['clean','actual']):
            data=[exemplar['videos'][video][history][k]['same_frame_rank1_percent'] for k in ['last','ema','fifo']]+[models[k]['videos'][video][history]['model']['same_frame_rank1_percent'] for k in ['gru_predictive','gru_association','transformer_predictive','transformer_association']]
            values[video+'_'+history]=dict(zip(labels,data));axis=axes[row,column];bars=axis.bar(np.arange(7),data,color=['#7c8ba1']*3+['#2a9d8f','#e9a23b','#2a9d8f','#e9a23b']);axis.set_ylim(0,100);axis.set_xticks(np.arange(7),labels,rotation=28,ha='right',fontsize=8);axis.set_title(video+' / '+('physically clean history' if history=='clean' else 'actual accepted history'));axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
            for bar,value in zip(bars,data):axis.text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.1f}',ha='center',fontsize=8)
            if column==0:axis.set_ylabel('Same-frame candidate Rank@1 (%)')
    figure.suptitle('Frozen development-query diagnosis, seed 0; no fitting on dev',fontsize=13)
    figure.text(.5,.015,'Offline GT-labelled query diagnostic; not tracking IDF1. All models fixed epoch 8, identical frozen RGB latent targets and history budget.',ha='center',fontsize=8)
    figure.tight_layout(rect=[0,.04,1,.95]);args.output.mkdir(parents=True);path=args.output/'prediction_query_rank1.png';figure.savefig(path,dpi=170);plt.close(figure)
    write_json(args.output/'manifest.json',dict(status='COMPLETE',input_manifest_sha256=sources,source_sha256=sha256(__file__),figure_sha256=sha256(path),measured_values=values,scope='Four models, seed0 only, canonical-dev offline GT candidate diagnosis, no uncertainty invented/no tracking or causal-oracle gain claim.'))


if __name__=='__main__':main()
