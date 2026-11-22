#!/usr/bin/env python3
"""Archive old, Dev-frozen ordinary metric controls without Test-driven choice."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    assert json.loads((w/'metric_selected_test_state.json').read_text())['status']=='COMPLETE'
    rows=[]
    for config in sorted((ROOT/'configs/perceptual_metric_selected_20261122').glob('*.json')):
        folder=w/'metric_selected_test'/config.stem;p=folder/'metrics.json';v=json.loads(p.read_text());m=v['manifest']
        assert v['status']=='COMPLETE' and m['frames']==15985 and m['sequences']==protocol['historical_test']
        assert m['config_sha256']==sha256(config)
        per=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['historical_test'],v['trackeval'])
        assert sum(x['CLR_Frames'] for x in per.values())==15985
        rows.append(dict(id=config.stem,config=m['config'],metrics=v['trackeval'],per_sequence=per,metrics_sha256=sha256(p),manifest=m))
    assert len(rows)==9
    report=dict(status='COMPLETE_FIXED_PREVIOUS_METRIC_TEST_CONTROLS',scope='Nine old ordinary metric-control frontend/head combinations, each complete7 Test sequences and15985 frames. Checkpoints were selected on Dev before Test; all six heads executed24 epochs. This export validates coverage/coherence and does not rank a Test winner or feed later parameter selection.',rows=rows,selection_decision=json.loads((w/'metric_epoch24_decision.json').read_text()),current_method_selection_reads_this_export=False)
    output=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_FIXED_METRIC_TEST.json';dump(output,report)
    # Deliberately print coverage only: subsequent hypotheses are still chosen
    # on Train/Dev, and must not be steered by these accuracy numbers.
    print(json.dumps(dict(status=report['status'],combinations=len(rows),sequences_each=7,frames_each=15985,output=str(output))))


if __name__=='__main__':main()
