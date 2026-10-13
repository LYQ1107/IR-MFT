#!/usr/bin/env python3
"""Inner-only source-contract check: learned assignment vs native feasible graph."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfp.centered_joint_identity import CenteredJointIdentity
from cfp.joint_identity_decoder import decode_joint_assignment
from cfp.joint_identity_training_data import load_joint_training,joint_frame_batch
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['model','queries','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--video',required=True);a=ap.parse_args()
    m=json.loads((a.model/'manifest.json').read_text());c=m['config']
    if a.output.exists() or a.video not in c['calibration_videos']:raise ValueError('Registered inner-only graph diagnosis; reject outer before loading weights or querybanks')
    if m['status']!='COMPLETE' or sha256(a.model/'epoch8.pt')!=m['checkpoint_sha256']:raise ValueError('Exact completedmodel required')
    torch.set_num_threads(2);model=CenteredJointIdentity(c['dimension'],c['hidden'],c['heads'],c['capacity'],m['guard']['architecture']).eval();model.load_state_dict(torch.load(a.model/'epoch8.pt',map_location='cpu',weights_only=False)['model']);records,banks,manifest=load_joint_training(a.queries)
    if sha256(a.queries/'manifest.json')!=m['guard']['query_manifest_sha256']:raise ValueError('Sameimmutable querysupply required')
    counts=dict(frames=0,current_detections=0,selected_existing=0,selected_outside_native_feasible_graph=0,selected_null=0,known_existing=0,known_existing_outside_native_graph=0,known_null=0,unknown_or_nonunique=0);rng=np.random.default_rng(m['guard']['seed']+1000)
    with torch.inference_mode():
        for record in records:
            if record['video']!=a.video:continue
            inputs,s=joint_frame_batch(record,banks,'cpu',rng,c['capacity']);out=model(**inputs);chosen=decode_joint_assignment(out['logits'],inputs['detection_valid']);q,i=banks[a.video]['queries'],record['position'];lo,hi=q['edge_offsets'][i:i+2];cost=q['baseline_cost'][lo:hi].reshape(inputs['geometry'].shape);legal=cost<=c['unmatched_cost'];counts['frames']+=1;counts['current_detections']+=len(chosen)
            for j,t in enumerate(chosen.tolist()):
                if t<0:counts['selected_null']+=1
                else:
                    counts['selected_existing']+=1;counts['selected_outside_native_feasible_graph']+=int(not legal[t,j])
            n=len(inputs['history'])
            for j,t in enumerate(s['target'].tolist()):
                if t<0:counts['unknown_or_nonunique']+=1
                elif t==n:counts['known_null']+=1
                else:
                    counts['known_existing']+=1;counts['known_existing_outside_native_graph']+=int(not legal[t,j])
    write_json(a.output,dict(status='COMPLETE_INNER_ONLY_NATIVE_GRAPH_CONTRACT_DIAGNOSIS',video=a.video,counts=counts,native_unmatched_cost=c['unmatched_cost'],model_checkpoint_sha256=m['checkpoint_sha256'],query_manifest_sha256=sha256(a.queries/'manifest.json'),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),scope='InnerSN001only; fullroute mappingselectededges tozero must not authorize native-forbidden baselinecost>.97edges. Geometry/lifecycle contract issue, not outerthreshold tuning. Knownpositive outside feasiblegraph remains physicalexisting, not relabelled NULL. This diagnostic doesnot loadoutermodel scores.' ))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
