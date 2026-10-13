#!/usr/bin/env python3
"""Audit the unchanged visual recipe with explicit deterministic execution."""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    arguments = sys.argv[1:]
    config_path = Path(arguments[arguments.index('--config')+1])
    output = Path(arguments[arguments.index('--output')+1])
    config = json.loads(config_path.read_text())
    policy = config['determinism']
    if policy['cublas_workspace_config'] != ':4096:8' or not policy['torch_deterministic_algorithms']:
        raise ValueError('Exact declared deterministic policy required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
        raise ValueError('Incompatible inherited cuBLAS workspace')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from cfp.provenance import sha256, snapshot_sources, write_json
    if torch.cuda.is_initialized():
        raise ValueError('Workspace must be configured before CUDA initialization')
    torch.use_deterministic_algorithms(True)
    execution_path = output.with_suffix('.execution.json')
    guard = dict(config_sha256=sha256(config_path), wrapper_sha256=sha256(__file__),
        trainer_sha256=sha256(Path(__file__).with_name('cfp_train_dinov2_fish_identity.py')),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        physical_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__)
    if '--resume' in arguments:
        execution = json.loads(execution_path.read_text())
        if execution['guard'] != guard or execution['status'] != 'CHECKPOINTED_EARLY_STOP':
            raise ValueError('Identical deterministic stopped execution required')
    else:
        if execution_path.exists():
            raise ValueError('New deterministic producer required')
        execution = dict(status='RUNNING', guard=guard, launches=[],
            sources=snapshot_sources(output.with_suffix('.execution.sources'), [__file__, config_path]))
    execution['launches'].append(dict(resume='--resume' in arguments, pid=os.getpid()))
    write_json(execution_path, execution)
    try:
        namespace = runpy.run_path(str(Path(__file__).with_name('cfp_train_dinov2_fish_identity.py')))
        namespace['main']()
        manifest = json.loads((output/'manifest.json').read_text())
        execution.update(status=manifest['status'], training_manifest_sha256=sha256(output/'manifest.json'))
        write_json(execution_path, execution)
    except Exception as error:
        execution.update(status='FAILED_DETERMINISTIC_EXECUTION', error=str(error))
        write_json(execution_path, execution)
        raise


if __name__ == '__main__':
    main()
