#!/usr/bin/env python3
"""Scientific figures from all42 control metrics and retrospective motion support."""
import argparse,json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfp.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['occlusion','motion','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('New immutable measured figure bundle required')
    o=json.loads(a.occlusion.read_text());m=json.loads(a.motion.read_text());a.output.mkdir();fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');videos=['SN-001','SN-015','PF-001','BT-001','BT-003','MSK-002'];x=np.arange(6)
    k=[m['videos'][v]['all_supported']['KF_center_error_mean'] for v in videos];c=[m['videos'][v]['all_supported']['last_two_measurement_CV_center_error_mean'] for v in videos];axes[0].bar(x-.17,k,.34,label='Native KF',color='#326f9f');axes[0].bar(x+.17,c,.34,label='Last two measurements CV',color='#cf9136');axes[0].set_xticks(x,videos,rotation=25);axes[0].set_ylabel('Mean center error / current GT box diagonal');axes[0].set_title('Retrospective clean-recent-history support');axes[0].legend(fontsize=8)
    keys=['constant_half0','row_mean0','overlap0','permuted_overlap0','permuted_overlap1','permuted_overlap2'];labels=['Half','Row mean','Overlap','Perm 0','Perm 1','Perm 2'];base=o['macro_by_configuration']['native0'];h=[o['macro_by_configuration'][v]['HOTA']-base['HOTA'] for v in keys];i=[o['macro_by_configuration'][v]['IDF1']-base['IDF1'] for v in keys];x=np.arange(6);axes[1].bar(x-.17,h,.34,label='HOTA change',color='#326f9f');axes[1].bar(x+.17,i,.34,label='IDF1 change',color='#cf9136');axes[1].axhline(0,color='#333',linewidth=.8);axes[1].set_xticks(x,labels,rotation=25);axes[1].set_ylabel('Macro-video change (percentage points)');axes[1].set_title('Association-only overlap proxies, all six videos');axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    artifacts={}
    for ext in ['png','pdf','svg']:
        p=a.output/('observable_motion_controls.'+ext);fig.savefig(p,dpi=220);artifacts[ext]=dict(path=str(p),sha256=sha256(p))
    plt.close(fig);write_json(a.output/'manifest.json',dict(status='MEASURED_OBSERVABLE_MOTION_CONTROL_FIGURES',occlusion_sha256=sha256(a.occlusion),motion_sha256=sha256(a.motion),script_sha256=sha256(__file__),artifacts=artifacts,caption='Left: selected entirely-known consistent recent histories only, currentGT used after all6native predictionbyteparities, excludes unknown/mixed/short support, notnonlinear/occlusion causality proof. KF/CVhistorycapacity differs. Right: deterministicheuristics and3permutation placebo seeds, not3newtrainedmethods; fixedlegacyV3/512GRUs0 inputs, raw/uncachedtime excluded. No confidence/significance orpositiveindependentnovelty claim.'))

if __name__=='__main__':main()
