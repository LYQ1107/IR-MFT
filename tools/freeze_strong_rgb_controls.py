#!/usr/bin/env python3
"""Retain every matched residual control, selected from complete Dev curves."""
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    for stage in ['strong_rgb_fit_e36','strong_rgb_dev','strong_rgb24_dev','strong_rgb30_dev','strong_rgb36_dev']:
        if json.loads((work/(stage+'_state.json')).read_text())['status'] != 'COMPLETE':
            raise ValueError('All six fits and72 Dev snapshots must finish')
    destination = ROOT/'configs/perceptual_strong_rgb_selected_20261123'
    if destination.exists():
        raise ValueError('Strong RGB controls already frozen')
    destination.mkdir()
    rows = []
    for front in ['new','rgb']:
        for cue in ['rgb','ir']:
            for seed in [0,1,2]:
                completion = json.loads((work/f'strong_rgb_residual/{cue}_s{seed}/complete.json').read_text())
                if completion['epochs'] != 36 or completion['Dev_fit_rows'] or completion['Test_fit_rows']:
                    raise ValueError('Common actual training budget mismatch')
                curve = []
                for epoch in [6,12,18,24,30,36]:
                    suffix = '' if epoch<=18 else str(epoch)
                    name = f'F_{front}_{cue}_s{seed}_e{epoch:02d}'
                    config = ROOT/f'configs/perceptual_strong_rgb{suffix}_dev_20261123/{name}.json'
                    path = work/f'strong_rgb{suffix}_dev'/name/'metrics.json'
                    result = json.loads(path.read_text());manifest=result['manifest']
                    if (result['status']!='COMPLETE' or manifest['frames']!=4834 or manifest['split']!='dev'
                            or manifest['config_sha256']!=sha256(config)):
                        raise ValueError('Incomplete/mismatched Dev record')
                    curve.append(dict(epoch=epoch,HOTA=result['trackeval']['HOTA'],IDF1=result['trackeval']['IDF1'],
                        configuration=str(config),Dev_metrics=str(path),Dev_metrics_sha256=sha256(path)))
                selected = min(curve,key=lambda value:(-value['HOTA'],value['epoch']))
                config = json.loads(Path(selected['configuration']).read_text())
                name = f'L_{front}_{cue}_s{seed}'
                config.update(id=name,registration='All six residual fits completed36 epochs and all72 Dev snapshots finished. Select maximum complete Dev HOTA with earliest exact tie. Retain all12 controls for complete7-video Test; no Test selection.')
                path = destination/(name+'.json');dump(path,config)
                rows.append(dict(frontend=front,cue=cue,seed=seed,curve=curve,selected=selected,
                    configuration=str(path),configuration_sha256=sha256(path),checkpoint=config['association_checkpoint'],
                    checkpoint_sha256=sha256(config['association_checkpoint']),executed_epochs=36))
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRONG_RGB_SELECTION.json',dict(
        status='FROZEN_BEFORE_FORMAL_STRONG_RGB_TEST',rows=rows,Test_selection=False,
        scope='Complete matched residual family, including negative/unstable seeds. Fixed original detector and RGB teacher.'))
    print(json.dumps(dict(controls=len(rows),Dev_snapshots=72,Test_accuracy_read=False)))


if __name__ == '__main__':
    main()
