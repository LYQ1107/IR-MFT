#!/usr/bin/env python3
"""Audit all fifty original full-Train Dev CSVs after the original frozen choice."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47
from tools.native_pyramid_experiment import read
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();plan=read(ROOT/'configs/perceptual_mft_20261122.json');work=Path(plan['work_root'])
    require_host_memory(8*2**30,maximum_workers=1);require_storage(work,growth_bytes=4*2**20)
    protocol=read(ROOT/plan['protocol']);statepath=work/'structural_full_train_dev_state.json';state=read(statepath)
    selection_path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json'
    selected=read(selection_path);selection_sha=sha256(selection_path)
    assert state['status']=='COMPLETE' and len(state['jobs'])==50
    assert selected['status']=='COMPLETE_ALL50_FULL_TRAIN_DEV_AND_FROZEN_CHOICES'
    assert selected['completed_Dev_state_sha256']==sha256(statepath)
    assert selected['Test_accuracy_used_for_selection'] is False
    expected={v['id']:v for v in selected['rows']};assert set(expected)==set(state['jobs'])
    schema,counts=official_schema(plan['evaluation']['TrackEval_root']);assert len(schema)==47
    rows=[];files={};csv_sources={}
    for name,job in state['jobs'].items():
        assert job['status']=='COMPLETE'
        directory=work/'structural_full_train_dev'/name;path=directory/'metrics.json';value=read(path)
        manifest=value['manifest'];config=manifest['config'];csv=directory/'trackeval_output/pedestrian_detailed.csv'
        assert value['status']=='COMPLETE' and manifest['split']=='dev' and manifest['frames']==4834
        assert manifest['sequences']==protocol['dev'] and config['seed']==0
        assert not any(manifest[k] for k in ['GT_inference_input','future_frame_input','previous_outputs_rewritten'])
        assert manifest['config_sha256']==job['sha256']==sha256(job['config'])
        assert config==expected[name]['config'] and value['trackeval']==expected[name]['metrics']
        assert sha256(path)==expected[name]['metrics_sha256'] and sha256(csv)==expected[name]['official_detailed_csv_sha256']
        assert {s:c['frames'] for s,c in manifest['coverage'].items()}=={'BT-005':1834,'SN-013':3000}
        parsed=official_csv47(csv,schema,'dev',value['trackeval'])
        assert parsed['original_columns']==276
        assert set(parsed['per_sequence47'])==set(protocol['dev']) and len(parsed['combined47'])==47
        rows.append(dict(id=name,config=config,metrics39=value['trackeval'],**parsed,
            metrics_file=str(path),metrics_sha256=sha256(path),original_CSV=str(csv)))
        for source in [path,csv,Path(job['config'])]:files[str(source)]=sha256(source)
        csv_sources[name]=csv
    by_id={r['id']:r for r in rows};primary=by_id[selected['primary']]
    def hota(name):return by_id[name]['selection_metrics']['HOTA']
    primary_hota=hota(primary['id']);control_names=selected['controls']
    controls={cue:dict(id=name,HOTA=hota(name),primary_minus_HOTA=primary_hota-hota(name)) for cue,name in control_names.items()}
    strongest=max(controls,key=lambda cue:controls[cue]['HOTA'])
    removals=[]
    for row in selected['removals']:
        removals.append(dict(mechanism=row['mechanism'],components=row['components'],paired=row['paired'],Dev_best=row['Dev_best'],
            primary_minus_paired_HOTA=primary_hota-hota(row['paired']),primary_minus_removal_own_best_HOTA=primary_hota-hota(row['Dev_best'])))
    wrong=selected['interventions']['IR_image_spatial_roll'];missing=selected['interventions']['unavailable']
    reference='FT_source_RGB_fresh' if primary['config']['policy']=='raw' else 'FT_source_RGB_canonical'
    assert by_id[missing]['combined47']==by_id[reference]['combined47']
    assert by_id[missing]['per_sequence47']==by_id[reference]['per_sequence47']
    assert by_id[missing]['official_CSV_sha256']==by_id[reference]['official_CSV_sha256']
    assert sha256(selection_path)==selection_sha and sha256(statepath)==selected['completed_Dev_state_sha256']
    assert all(sha256(p)==s for p,s in files.items())
    value=dict(status='COMPLETE_INDEPENDENT_ALL50_FULL_TRAIN_DEV_OFFICIAL47_AUDIT',
        source_auditor_sha256=sha256(__file__),original_Dev_choice_file=str(selection_path),original_Dev_choice_sha256=selection_sha,
        completed_Dev_state_sha256=sha256(statepath),original_choice_preserved=True,new_selection_performed=False,
        complete_cases=50,frames_each=4834,sequences=protocol['dev'],official_fields=47,official_CSV_columns=276,
        official_schema=schema,official_family_field_counts=counts,rows=rows,source_file_hashes=files,
        frozen_primary=primary['id'],primary_Dev_HOTA=primary_hota,controls=controls,
        primary_minus_strongest_matched_Dev_HOTA=primary_hota-controls[strongest]['HOTA'],
        primary_minus_canonical_RGB_Dev_HOTA=primary_hota-hota('FT_source_RGB_canonical'),
        primary_minus_fresh_RGB_Dev_HOTA=primary_hota-hota('FT_source_RGB_fresh'),
        removals=removals,interventions=dict(wrong_IR=wrong,wrong_IR_HOTA=hota(wrong),missing_IR=missing,
            primary_minus_wrong_IR_HOTA=primary_hota-hota(wrong),missing_exact_policy_RGB_official47_and_CSV=True),
        actual_full_Train_epochs_each=3,optimizer_updates_each=20736,fixed_seed=0,
        original_Test_baseline_HOTA=29.529,target_Test_HOTA=34.529,new_Test_accuracy_read=False,
        new_GPU_work=False,historical_Test_exposed=True,source_RGB_pretraining_saw_Dev=True,
        auxiliary_is_RGB_derived=True,goal_met=False,supported_innovations=0,
        seconds=time.monotonic()-started,peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope='Independent original276column official47 extraction of all50completeDev cases. Original frozen primary/control/removal choices preserved; no Test accuracy input or new model selection. Dev gains cannot establish Test+5or2-3independent innovations; same-family structural removals are not distinct architectures.')
    assert value['peak_process_RSS_GiB']<8
    output=args.output_directory;output.mkdir(exist_ok=False);table=output/'official_CSVs';table.mkdir()
    for name,path in csv_sources.items():(table/(name+'.csv')).write_bytes(path.read_bytes())
    (output/'complete.json').write_text(json.dumps(value,indent=2)+'\n')
    lines=['# Six full-Train fits: independent complete Dev audit','',value['scope'],'',
        f"Frozen primary: {primary['id']}; Dev HOTA {primary_hota:.6f}.",
        f"Versus strongest matched RGB/gray: {value['primary_minus_strongest_matched_Dev_HOTA']:+.6f} points.",
        f"Versus historical canonical RGB on Dev: {value['primary_minus_canonical_RGB_Dev_HOTA']:+.6f} points.",'',
        '| Case | Dev HOTA | DetA | AssA | IDF1 |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| '+row['id']+' | '+' | '.join(f"{row['combined47'][k]:.6f}" for k in ['HOTA','DetA','AssA','IDF1'])+' |')
    lines+=['','| Removed mechanism | Primary minus paired | Primary minus removal own best |','| --- | ---: | ---: |']
    for row in removals:lines.append(f"| {row['mechanism']} | {row['primary_minus_paired_HOTA']:+.6f} | {row['primary_minus_removal_own_best_HOTA']:+.6f} |")
    lines+=['',f"Primary minus wrong-IR HOTA: {primary_hota-hota(wrong):+.6f} points. Missing IR reproduces its policy RGB in every official field and entire CSV.",'',
        'Dev only. New Test accuracy was not read. Three complete Train traversals per fit, not72full epochs. Goal remains active; supported independent contributions0.']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=value['status'],primary=value['frozen_primary'],HOTA=primary_hota,
        versus_strongest_matched_Dev=value['primary_minus_strongest_matched_Dev_HOTA'],removals=removals,
        versus_wrong_IR=primary_hota-hota(wrong),seconds=value['seconds'])),flush=True)


if __name__=='__main__':main()
