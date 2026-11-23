#!/usr/bin/env python3
"""Coverage/hash audit and complete export of a registered Dev family."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--config-dir',required=True)
    p.add_argument('--tag',required=True);p.add_argument('--title',required=True);p.add_argument('--note',required=True)
    p.add_argument('--completed-stage',help='Export one fully completed registered stage while other stages continue')
    p.add_argument('--state-file',help='Explicit controller state when its basename differs from the result directory');a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    state_path=Path(a.state_file) if a.state_file else w/(a.stage+'_state.json')
    state=json.loads(state_path.read_text());selected=None
    if a.completed_stage:
        assert state['stages'][a.completed_stage]['status']=='COMPLETE'
        selected={key for key,job in state['jobs'].items()if job['stage']==a.completed_stage}
        assert selected and all(state['jobs'][key]['status']=='COMPLETE'for key in selected)
    else:assert state['status']=='COMPLETE'
    rows=[]
    for cfg in sorted((ROOT/a.config_dir).glob('*.json')):
        if selected is not None and cfg.stem not in selected:continue
        folder=w/a.stage/cfg.stem;path=folder/'metrics.json';v=json.loads(path.read_text());m=v['manifest']
        assert v['status']=='COMPLETE' and m['frames']==4834 and m['sequences']==protocol['dev']
        assert m['config_sha256']==sha256(cfg)
        per=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['dev'],v['trackeval'])
        assert sum(x['CLR_Frames'] for x in per.values())==4834
        hashes={k:value for k,value in m.items() if k.endswith('_source_sha256') or k.endswith('_checkpoint_sha256')}
        rows.append(dict(id=cfg.stem,config=m['config'],metrics=v['trackeval'],per_sequence=per,statistics=m['statistics'],source_hashes=hashes,calibration=m.get('calibration'),config_sha256=sha256(cfg),metrics_sha256=sha256(path),cached_replay_FPS=m['replay_FPS']))
    assert rows
    if selected is not None:assert {row['id']for row in rows}==selected
    report=dict(status='COMPLETE_DEV_FAMILY_GOAL_NOT_ESTABLISHED',scope='Full4834-frame Dev, not Test. All registered rows and failures retained. Cached replay speed excludes pixel/encoder preparation and is not end-to-end FPS.',note=a.note,rows=rows,supported_multimodal_innovations=0,Test_selection=False)
    docs=ROOT/'docs/fusion_research';stem='PERCEPTUAL_MFT_20261123_'+a.tag
    dump(docs/(stem+'.json'),report)
    extra=[(title,key) for title,key in [('Prebirth ID recoveries','prebirth_recoveries'),('Missing-point appearance rows','semantic_rows')]
           if any(any(s.get(key,0) for s in row['statistics'].values()) for row in rows)]
    header='| Candidate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Old-ID weak recoveries | New candidate births |'
    text=['# '+a.title,'',report['scope'],'',a.note,'',header+''.join(' '+title+' |' for title,key in extra),
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'+''.join(' ---: |' for _ in extra)]
    for row in rows:
        m=row['metrics'];s=row['statistics'];recoveries=sum(x.get('weak_recoveries',0) for x in s.values());births=sum(x.get('candidate_births',0) for x in s.values())
        text.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f" | {int(m['IDSW'])} | {recoveries} | {births} |"+''.join(' '+str(sum(s.get(key,0) for s in row['statistics'].values()))+' |' for title,key in extra))
    text+=['','No current Dev family establishes the original >=+5 Test HOTA and2-3 independent contribution objective. Full metric fields, per-sequence values, configuration hashes and exact runtime source versions are in the adjacent JSON.','',
           'Classical threshold changes, generic memory and ordinary supervised heads receive no multimodal novelty credit alone. Historical Test was exposed before this campaign; original RGB pretraining also included present Dev videos.']
    (docs/(stem+'.md')).write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=report['status'],Dev_rows=len(rows),report=str(docs/(stem+'.json')))))


if __name__=='__main__':main()
