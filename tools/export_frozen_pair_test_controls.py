#!/usr/bin/env python3
"""Archive every Dev-frozen ordinary pair control, without choosing a Test winner."""
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    protocol=json.loads((ROOT/'configs/cfp/protocol_v2.json').read_text())
    state=json.loads((work/'pair_selected_test_state.json').read_text())
    if state['status']!='COMPLETE':raise ValueError('All12 fixed ordinary Test controls must finish')
    selection=json.loads((ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json').read_text())
    rows=[]
    for item in selection['rows']:
        cfg=Path(item['configuration']);folder=work/'pair_selected_test'/cfg.stem;path=folder/'metrics.json'
        result=json.loads(path.read_text());manifest=result['manifest']
        if (result['status']!='COMPLETE' or manifest['split']!='test' or manifest['frames']!=15985
                or manifest['sequences']!=protocol['historical_test'] or manifest['config_sha256']!=sha256(cfg)):
            raise ValueError('Frozen ordinary Test coverage/configuration mismatch')
        per=detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',protocol['historical_test'],result['trackeval'])
        if sum(row['CLR_Frames'] for row in per.values())!=15985:raise ValueError('Detailed frame coverage')
        rows.append(dict(id=cfg.stem,frontend=item['frontend'],cue=item['cue'],seed=item['seed'],
            metrics=result['trackeval'],per_sequence=per,Dev_selection=item['selected'],
            config_sha256=sha256(cfg),metrics_sha256=sha256(path),manifest=manifest))
    output=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_FIXED_PAIR_TEST.json'
    dump(output,dict(status='COMPLETE_ALL12_DEV_FROZEN_ORDINARY_PAIR_TEST_CONTROLS',rows=rows,
         scope='Every frontend/cue/seed retained, each7 Test sequences and15985 frames. Selection was frozen after all72 complete Dev snapshots and actual36 training epochs. No Test winner selected. Later method hyperparameters/checkpoints remain Train/Dev selected; historical Test exposure is acknowledged.',
         selection_sha256=sha256(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json')))
    print(json.dumps(dict(controls=len(rows),sequences_each=7,frames_each=15985,output=str(output))))


if __name__=='__main__':main()
