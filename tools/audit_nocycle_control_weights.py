#!/usr/bin/env python3
"""Compare completed RGB/gray control weights without new fitting or image/GT IO."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root'])
    fits=json.loads((work/'structural_nocycle_fit_state.json').read_text())
    rows=[]
    for cue in ['rgb','gray']:
        assert fits['jobs'][cue+'_no_cycle_s0']['status']=='COMPLETE'
        for epoch in [6,12,18,24]:
            new=work/'structural_nocycle_fit'/f'{cue}_no_cycle_s0/epoch{epoch:02d}.pt'
            old=work/'structural_relative_fit'/f'{cue}_full_s0/epoch{epoch:02d}.pt'
            a=torch.load(new,map_location='cpu',weights_only=False)
            b=torch.load(old,map_location='cpu',weights_only=False)
            assert a['config_sha256']==b['config_sha256'] and a['completed_epochs']==b['completed_epochs']==epoch
            assert a['metadata']['cue']==b['metadata']['cue']==cue
            assert a['metadata']['seed']==b['metadata']['seed']==0
            assert a['adapter'].keys()==b['adapter'].keys()
            changed=[key for key,value in a['adapter'].items()if not torch.equal(value,b['adapter'][key])]
            maximum=max((float((a['adapter'][key]-b['adapter'][key]).abs().max())for key in changed),default=0.)
            rows.append(dict(cue=cue,epoch=epoch,adapter_tensors=len(a['adapter']),changed_tensors=len(changed),
                all_adapter_values_exact=not changed,maximum_absolute_parameter_difference=maximum,
                no_cycle_checkpoint_sha256=sha256(new),initial_full_cycle_checkpoint_sha256=sha256(old)))
    report=dict(status='COMPLETE_RGB_GRAY_CONTROL_PARAMETER_COMPARISON',rows=rows,
        all8_snapshots_exact=all(r['all_adapter_values_exact']for r in rows),fixed_seed=0,additional_seed_work=False,
        source_sha256=sha256(__file__),
        scope='Same seed0/data/budget, four snapshots per RGB/gray fit. Descriptors for each cue compare '
              'RGB structure with itself in the current branch; null and wrong-frame training branches are not self-pairs. '
              'Removing the cycle check also changes those branches. Parameter equality is descriptive evidence about these completed fits; '
              'it does not prove operator equality on arbitrary unseen inputs or substitute for full Dev/Test. '
              'No checkpoint is changed and no accuracy/GT is read.',
        goal_met=False,supported_innovations=0)
    stem=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_NOCYCLE_CONTROL_WEIGHT_AUDIT'
    dump(stem.with_suffix('.json'),report)
    lines=['# Completed RGB/gray control weight comparison','',report['scope'],'',
        '| Cue | Epoch | Changed tensors | Maximum parameter difference | All tensors exact |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['cue']} | {row['epoch']} | {row['changed_tensors']} | "
            f"{row['maximum_absolute_parameter_difference']:.8g} | {row['all_adapter_values_exact']} |")
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(snapshots=8,all_values_exact=report['all8_snapshots_exact'],new_seed_work=False)))


if __name__=='__main__':main()
