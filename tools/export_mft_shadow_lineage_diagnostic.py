#!/usr/bin/env python3
"""Publish aggregate lineage diagnostics while retaining all events privately."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.diagnose_mft_shadow_lineage import registered,context,counters
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc


def main():
    r=registered();work,parent,base=context()
    for mode in ['produce','analyse']:
        e=read(base/(mode+'_r1.execution.json'))
        assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    ready=read(base/'source_ready.json');v=read(base/'complete_readonly_diagnostic.json')
    assert v['source_execution_sha256']==sha256(base/'produce_r1.execution.json')
    totals={};count=0
    for name,meta in ready['rows'].items():
        assert sha256(meta['path'])==meta['sha256']
        events=read(meta['path']);assert not events['GT_labels_read']and events['current_only_snapshot_delivery']
        assert len(events['events'])==meta['events']==v['rows'][name]['events']
        assert sum(c['opportunities']for c in v['rows'][name]['counts'].values())==meta['events']
        split=events['split'];count+=meta['events']
        for key,c in v['rows'][name]['counts'].items():
            assert c['opportunities']==sum(n for k,n in c.items()if k!='opportunities')
            total=totals.setdefault(split+'_'+key,counters())
            for k,n in c.items():total[k]+=n
    assert totals==v['totals']and count==3
    audit=dict(status='VERIFIED_DIAGNOSTIC_SOURCE_EXECUTIONS_AND_AGGREGATE_ARITHMETIC',UTC=utc(),
        all8_sources=8,frames=ready['frames'],candidate_events=count,actual_producer_exit0=True,actual_analysis_exit0=True,
        independent_aggregate_sums_rebuilt=True,source_files_verified=len(r['source_hashes']),
        scope='Aggregate arithmetic, event hashes and actual worker/source closure, not a second temporal-event or GT-label producer.',
        own_source_sha256=sha256(Path(__file__)),Test_read=False,goal_met=False,supported_innovations=0)
    frozen_json(base/'publication_aggregate_audit.json',audit)
    doc=ROOT/'docs/fusion_research';d=doc/'perceptual_20261127_shadow_lineage_diagnostic';assert not d.exists();d.mkdir()
    names=['registered.json','worker_r1.supervisor.py','produce_r1.execution.json','produce_r1.stdout.txt',
        'analyse_r1.execution.json','analyse_r1.stdout.txt','source_ready.json','complete_readonly_diagnostic.json','publication_aggregate_audit.json']
    paths=[]
    for name in names:
        p=base/name;assert p.stat().st_size<2*2**20
        shutil.copyfile(p,d/name);assert sha256(p)==sha256(d/name);paths.append(str((d/name).relative_to(ROOT)))
    report=dict(status='PUBLISHED_TRAIN_DEV_SHADOW_LINEAGE_AGGREGATES',UTC=utc(),frames=ready['frames'],rows=v['rows'],totals=v['totals'],
        publication_audit=audit,no_private_events_or_GT_IDs_published=True,no_HOTA_or_tracker_outputs=True,
        prior_Test_exposed=True,Test_read=False,new_GPU_work=False,goal_met=False,supported_innovations=0)
    p=doc/'PERCEPTUAL_MFT_20261127_SHADOW_LINEAGE_DIAGNOSTIC.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');paths.append(str(p.relative_to(ROOT)))
    text='''# Continuous shadow lineage: Train/Dev-only diagnosis

The GT-free producer processes all6Train27,251 and2Dev4,834 frames before a separate worker reads endpoint labels. A candidate new trusted identity must share the same current proposal with a shadow that previously agreed with a different trusted identity for at least3 consecutive frames; that old identity must be absent now, and the current shadow's measured streak must span the entire gap plus3. No identity is merged, no output tracker is created, and no HOTA/model/threshold selection is performed.

Only **2 Train events** and **1 Dev event** meet this fixed rule. The early-Train short-gap event (6–25 frames) has matching known GT endpoints; its long-gap event (26–250) has **different** known GT endpoints. Dev has one short-gap event with matching known endpoints. The final25percent Train validation window has **no event**. All labels retain the original Hungarian/IoU/ignored-region rules, and unknown is not background.

This small inventory does not support building a long-gap automatic identity merger or predict a meaningful HOTA gain. In particular, the only long-gap Train candidate is wrong under endpoint identity labels. The conclusion is local to this registered continuous-shadow rule, not a proof that all reidentification strategies are unhelpful.

Both actual workers exit0. Aggregate/event-hash/source checks are recorded; this publication check is not a second temporal-event or GT-label producer. Private event rows and GT identities are excluded from Git; aggregate counts, execution/source hashes and logs are retained. This diagnostic creates no new Test result and reads no Test labels/metrics. Historical Test exposure remains disclosed.

The previous formal selected PP method remains32.4955880280 HOTA, +2.9666173406 over the original RGB-only YOLOX-X+ByteTrack29.5289706874, below the+5 minimum34.5289706874. Fixed November DiffV2IR/ProtocolV2, seed0, no new neural training/images/descriptors/weights/GPU work. Goal remains active and no new independently attributable multimodal innovation is claimed.
'''
    p=doc/'PERCEPTUAL_MFT_20261127_SHADOW_LINEAGE_DIAGNOSTIC.md';assert not p.exists();p.write_text(text);paths.append(str(p.relative_to(ROOT)))
    p=d/'publication_manifest.json';p.write_text(json.dumps(dict(paths=paths,file_SHA256={n:sha256(ROOT/n)for n in paths},
        total_bytes=sum((ROOT/n).stat().st_size for n in paths),large_assets_included=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_READONLY_LINEAGE_AGGREGATES',paths=len(paths)+1,candidate_events=count)))


if __name__=='__main__':main()
