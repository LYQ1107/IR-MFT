#!/usr/bin/env python3
"""Freeze every ordinary-control head from the complete Dev curve, before Test."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    for state in ['association_fit_e36','calibrated_dev','calibrated24_dev','calibrated30_dev','calibrated36_dev']:
        if json.loads((w/(state+'_state.json')).read_text())['status']!='COMPLETE':raise ValueError('Every common36-epoch Dev curve must be complete')
    destination=ROOT/'configs/perceptual_pair_selected_20261123'
    if destination.exists():raise ValueError('Ordinary Test controls already frozen')
    destination.mkdir();rows=[]
    for front in ['new','rgb']:
        for cue in ['rgb','ir']:
            for seed in [0,1,2]:
                complete=json.loads((w/f'association_metric/{cue}_s{seed}/complete.json').read_text())
                if complete['epochs']!=36 or complete['Dev_fit_rows'] or complete['Test_fit_rows']:raise ValueError('Actual common training budget')
                curve=[]
                for epoch in [6,12,18,24,30,36]:
                    suffix='' if epoch<=18 else str(epoch);stage='calibrated'+suffix+'_dev'
                    name=f'G_{front}_{cue}_s{seed}_e{epoch:02d}'
                    cfg=ROOT/f'configs/perceptual_calibrated{suffix}_dev_20261123/{name}.json';config=json.loads(cfg.read_text())
                    path=w/stage/name/'metrics.json';v=json.loads(path.read_text());m=v['manifest']
                    if v['status']!='COMPLETE' or m['split']!='dev' or m['frames']!=4834 or m['config_sha256']!=sha(cfg):raise ValueError('Incomplete/mismatched Dev evidence')
                    curve.append(dict(epoch=epoch,HOTA=v['trackeval']['HOTA'],IDF1=v['trackeval']['IDF1'],Dev_metrics=str(path),Dev_metrics_sha256=sha(path),configuration=str(cfg)))
                selected=min(curve,key=lambda item:(-item['HOTA'],item['epoch']))
                config=json.loads(Path(selected['configuration']).read_text());name=f'K_{front}_{cue}_s{seed}'
                config.update(id=name,registration='Ordinary-control weights frozen only after all six fits completed36 epochs and all72 Dev snapshots completed. Highest complete Dev HOTA, earliest checkpoint on ties. Every frontend/cue/seed control is retained for all7 Test videos; no Test selection.')
                path=destination/(name+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
                rows.append(dict(frontend=front,cue=cue,seed=seed,curve=curve,selected=selected,configuration=str(path),configuration_sha256=sha(path),checkpoint=config['association_checkpoint'],checkpoint_sha256=sha(config['association_checkpoint']),executed_epochs=36))
    report=dict(status='FROZEN_BEFORE_FORMAL_ORDINARY_TEST',selection='Highest complete Dev HOTA, earliest epoch on exact ties. All12 controls retained; no selection by Test.',rows=rows,Test_selection=False,scope='Ordinary learned association controls; fixed detector/encoder and three head seeds, not three detector retrainings. Historical Test was exposed before this campaign.')
    path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json';path.write_text(json.dumps(report,indent=2)+'\n')
    (w/'pair_selected_frozen.json').write_text(json.dumps(dict(status='COMPLETE',selection_sha256=sha(path),controls=12,Test_selection=False),indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],controls=len(rows),Dev_snapshots=sum(len(row['curve']) for row in rows),Test_accuracy_read=False)))


if __name__=='__main__':main()
