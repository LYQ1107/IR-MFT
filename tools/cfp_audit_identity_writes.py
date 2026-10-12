#!/usr/bin/env python3
"""Retrospective physical write audit; GT cannot influence saved online decisions."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256,write_json,snapshot_sources
from cfp.supervision import independent_proposal_labels

spec=importlib.util.spec_from_file_location('cfp_write_gt_helper',Path(__file__).with_name('cfp_build_utility_queries.py'))
gt_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(gt_helper)


def audit_records(records,labels,frame_of_proposal,ambiguous):
    anchors={};counts=defaultdict(int);by_identity=defaultdict(lambda:defaultdict(int));examples=[];previous_frame=0;seen=set();last_written={};ages=[]
    for record in records:
        frame=record['frame'];index=record['proposal_index'];track=record['track_id']
        if frame<previous_frame or (frame,track) in seen or index is None or index<0 or index>=len(labels) or frame_of_proposal[index]!=frame:raise ValueError('Write record is not a unique current-frame proposal')
        previous_frame=frame;seen.add((frame,track));written=record['written'];label=int(labels[index]);counts['records']+=1
        if track in last_written:ages.append(frame-last_written[track])
        if written:last_written[track]=frame
        if label in ambiguous.get(frame,set()):label=-1
        if written:counts['actually_written']+=1
        else:counts['refused_or_unavailable']+=1
        if track not in anchors:
            if written and label>=0:
                anchors[track]=label;counts['known_anchor_initializations']+=1
            else:counts['unknown_past_anchor_records']+=1
            continue
        if label<0:
            counts['unknown_current_physical_label']+=1;continue
        correct=label==anchors[track];key=('written_' if written else 'refused_')+('correct' if correct else 'wrong')
        counts[key]+=1;by_identity[str(anchors[track])][key]+=1
        if len(examples)<20 and ((written and not correct) or (not written and correct)):
            examples.append(dict(**record,original_past_physical_anchor=anchors[track],current_physical_id=label))
    counts=dict(counts);written_known=counts.get('written_correct',0)+counts.get('written_wrong',0)
    refused_known=counts.get('refused_correct',0)+counts.get('refused_wrong',0)
    return dict(counts=counts,existing_known_write_precision=counts.get('written_correct',0)/written_known if written_known else None,
                refused_known_wrong_fraction=counts.get('refused_wrong',0)/refused_known if refused_known else None,
                prior_memory_age_at_write_attempt=dict(attempts=len(ages),p50_p95_frames=np.quantile(ages,[.5,.95]).tolist() if ages else None,maximum_frames=max(ages) if ages else None,fraction_over_trained_gap30=float(np.mean(np.asarray(ages)>30)) if ages else None),
                by_original_physical_anchor={k:dict(v) for k,v in by_identity.items()},examples=examples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--detections',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Physical memory audits are immutable')
    protocol=json.loads(args.protocol.read_text());manifest=json.loads((args.predictions/'manifest.json').read_text())
    if protocol['state']!='FROZEN' or manifest['status']!='COMPLETE' or manifest['partition']!='dev' or manifest['protocol_sha256']!=sha256(args.protocol) or not manifest['save_write_audit'] or set(manifest['videos'])!=set(protocol['dev']):raise ValueError('Complete canonical-dev observable write logs required')
    annotation=args.data_root/'annotations/train.json'
    if sha256(annotation)!=protocol['train_annotation_sha256']:raise ValueError('Physical annotation changed')
    gt=gt_helper.gt_by_video(json.loads(annotation.read_text()),protocol)
    args.output.mkdir(parents=True);sources=snapshot_sources(args.output,[__file__,'tools/cfp_build_utility_queries.py','cfp/supervision.py','cfp/provenance.py',args.protocol])
    report=dict(status='RUNNING',prediction_manifest_sha256=sha256(args.predictions/'manifest.json'),protocol_sha256=sha256(args.protocol),source_snapshots=sources,scope='Saved online write decisions are fixed before any GT is loaded. Sticky first available actually written physical identity anchors diagnosis only. Unknown/ambiguous proposals are excluded, not classified wrong. Precision excludes tautological first anchor initialization. Policies can alter later trajectories, so these are within-run audits, not paired causal counts.',videos={})
    write_json(args.output/'manifest.json',report)
    for sequence in protocol['dev']:
        dp=args.detections/(sequence+'.npz');wp=args.predictions/(sequence+'_writes.jsonl');v=manifest['videos'][sequence]
        if sha256(dp)!=v['detection_sha256'] or sha256(wp)!=v['write_audit_sha256']:raise ValueError('Online observation or proposal bytes changed')
        with np.load(dp) as a:frames,offsets,boxes=[a[k] for k in ['frame_ids','offsets','boxes']]
        labels=np.full(len(boxes),-1,np.int64);frame_of_proposal=np.repeat(frames,np.diff(offsets));ambiguity={}
        for i,frame in enumerate(frames):
            identities,gt_boxes=zip(*gt[sequence][int(frame)]) if gt[sequence][int(frame)] else ([],[])
            lo,hi=offsets[i:i+2];labels[lo:hi],ambiguity[int(frame)]=independent_proposal_labels(boxes[lo:hi],gt_boxes,identities)
        for row in protocol['ambiguous_identity_keys']:
            if row['sequence']==sequence:ambiguity[row['frame_id']].add(row['track_id'])
        records=[json.loads(line) for line in wp.read_text().splitlines()]
        report['videos'][sequence]=dict(**audit_records(records,labels,frame_of_proposal,ambiguity),write_log_sha256=sha256(wp),prediction_sha256=v['prediction_sha256']);write_json(args.output/'manifest.json',report)
    report['status']='COMPLETE';write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
