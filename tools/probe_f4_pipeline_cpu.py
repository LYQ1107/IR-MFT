#!/usr/bin/env python3
"""Record actual current complete CPU integration, never real GPU qualification."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_evaluation import read,sha
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_pipeline import current_source_hashes
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_workflow import FusionMemoryWorkflow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt',type=Path,required=True)
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU integration only')
    workflow=FusionMemoryWorkflow();tests=read(args.pytest_receipt)
    require(tests['exit_code']==0 and tests['all_sources_unchanged'] and sha(tests['log'])==tests['log_sha256'],
            'Actual complete current integration checks required')
    require(tests['tests']==['tests/test_fusion_memory_pipeline.py','tests/test_fusion_memory_workflow.py',
        'tests/test_fusion_memory_results.py','tests/test_fusion_memory_engineering.py','tests/test_fusion_memory_runtime.py'],
        'All current controller and component suites must execute')
    hashes=current_source_hashes()
    require(tests['source_hashes']==hashes,'Integration tested different source versions')
    passed=re.search(r'\b(\d+) passed\b',Path(tests['log']).read_text())
    require(passed is not None and int(passed.group(1))>=95,'Actual complete passing pytest coverage missing')
    comparator_path=workflow.work/'fusion_memory_engineering_comparison_CPU_proof.json'
    comparator=read(comparator_path)
    require(comparator['status']=='COMPLETE_ACTUAL_RGB_CPU_ENGINEERING_COMPARATOR_NOT_GPU_QUALIFICATION' and
        comparator['formal_GPU_qualification'] is False,'Real RGB CPU component evidence required')
    executed=['tools/fusion_memory_engineering.py','tools/fusion_memory_online.py','tools/fusion_memory_capture.py',
        'tools/fusion_memory_features.py','tools/fusion_memory_model_loading.py','tools/fusion_memory_observation.py']
    require(all(comparator['source_hashes'][p]==hashes[p] for p in executed),'Actually executed CPU inference core changed')
    audit=read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_F4_RESULT_EXPORT_COMPONENTS_SOURCE_AUDIT.json')
    artifacts={row['registration']:row['registration_sha256'] for row in audit['existing_registrations']}
    artifacts.update({str(comparator_path):sha(comparator_path),str(args.pytest_receipt.resolve()):sha(args.pytest_receipt),
        str(Path(tests['log']).resolve()):sha(tests['log'])})
    require(all(sha(p)==s for p,s in artifacts.items()),'Integration artifact changed')
    preflight=workflow.preflight()
    require(not preflight['missing_pipeline_entrypoints'],'Complete entrypoints are missing')
    output=workflow.work/'fusion_memory_full_workflow_CPU_proof.json'
    require(not output.exists(),'Preserve original complete CPU integration proof')
    write_json(output,dict(status='COMPLETE_F4_WORKFLOW_CPU_INTEGRATION',source_hashes=hashes,artifact_hashes=artifacts,
        actual_pytest_receipt=str(args.pytest_receipt.resolve()),tests_passed=int(passed.group(1)),
        CPU_scope='Synthetic169-job full controller with real complete-length artificial Dev-file freezes and Dev-only choices; actual stdlib CPU process handles/failure/resource boundaries; current component suites. No real GPU/model qualification.',
        actual_RGB_CPU_comparator_reference=dict(path=str(comparator_path),sha256=sha(comparator_path),
            actually_executed_core_hashes={p:hashes[p] for p in executed},
            scope='Previously actual eight Train RGB frames; core hashes unchanged. Unexecuted old GPU-entry hash is not claimed current.'),
        actual_preflight=preflight,previous_CPU_packet_anomaly_cause_unresolved=True,
        actual_GPU_qualification=False,actual_parent_eligibility=False,new_GPU_work=False,new_NN_fits=False,
        new_Dev_Test_accuracy=False,goal_met=False,supported_innovations=0),exclusive=True)
    print(json.dumps(dict(proof=str(output),sha256=sha(output),tests_passed=int(passed.group(1)),new_GPU_work=False)),flush=True)


if __name__=='__main__':main()
