#!/usr/bin/env python3
"""Queue every frozen neck72 fresh-image runtime after its accuracy process."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.neck72_experiment import context
from tools.neck72_runtime import read, require_host_memory, require_storage
from tools.neck72_runtime_study import (SPEC, REGISTRY, sources, registered, predecessor_ready, cases)
from tools.run_pseudo_ir_comparison import dump, sha256


def register():
    plan, protocol, work = context()
    path = work / REGISTRY; assert not path.exists()
    proof_path = work / 'neck72_online_inputs_CPU_proof.json'; proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_CPU_NECK72_ONLINE_INPUT_AND_PREFIX_PACKET_PROOF'
    assert proof['source_hashes'] == {p: sha256(ROOT / p) for p in sources()}
    assert proof['all_consumed_real_Dev_point_arrays_exact'] and proof['existing_model_streamed_prefix_tracks_exact']
    assert not proof['GPU_work'] and not proof['new_accuracy']
    parent = work / 'neck72_study_registered.json'
    assert sha256(parent) == read(SPEC)['accuracy_registration_sha256']
    artifacts = dict(read(parent)['artifact_hashes'])
    artifacts[str(parent)] = sha256(parent); artifacts[str(proof_path)] = sha256(proof_path)
    artifacts.update(proof['artifact_hashes'])
    frontend_sources = {}
    for arm, directory in plan['frozen_detection_sources'].items():
        metadata = Path(directory) / 'metrics.json'; protocol = read(metadata)['protocol']
        checkpoint = Path(protocol['ckpt'])
        assert protocol['seed'] == 0 and sha256(checkpoint) == protocol['source_sha256']
        artifacts[str(metadata)] = sha256(metadata); artifacts[str(checkpoint)] = protocol['source_sha256']
        frontend_sources[arm] = dict(manifest=str(metadata), manifest_sha256=sha256(metadata),
            checkpoint=str(checkpoint), checkpoint_sha256=protocol['source_sha256'])
    value = dict(status='REGISTERED_NECK72_FRESH_IMAGE_RUNTIME_PENDING_ACTUAL_GPU_SOURCE_PROOF',
        specification_sha256=sha256(SPEC), source_hashes=proof['source_hashes'], artifact_hashes=artifacts,
        frontend_sources=frontend_sources, parent_accuracy_registration_sha256=sha256(parent),
        full_component_runtime_code_implemented=True, actual_GPU_runtime_measured=False,
        original_accuracy_pipeline_unchanged=True, fixed_seed=0, physical_gpus=['4', '5'],
        maximum_workers=2, no_Test_score_selection=True, goal_met=False, supported_innovations=0)
    require_storage(work); dump(path, value)
    print(value['status'], len(value['source_hashes']), 'sources', flush=True)


class Queue:
    def __init__(self, work, state, specification):
        self.work, self.state, self.specification = work, state, specification
        self.path, self.active = work / 'neck72_fresh_runtime_state.json', {}

    def save(self):
        dump(self.path, self.state)

    def jobs(self, split, selected):
        assert split not in self.state['phases'] and not self.active
        rows = {case['id']: dict(status='PENDING', config=case['config'], command=[sys.executable, '-u',
            'tools/benchmark_neck72_runtime.py', '--case', case['id'], '--split', split]) for case in selected}
        self.state['phases'][split] = rows
        self.state['status'] = 'RUNNING_ALL_FROZEN_'+split.upper()+'_RUNTIME'; self.save()
        while True:
            # Source hashes are cheap; large immutable artifact checks happen
            # once in each worker, never every10seconds in this controller.
            registration = read(self.work / REGISTRY)
            assert all(sha256(ROOT / p) == h for p, h in registration['source_hashes'].items())
            for physical, (key, process) in list(self.active.items()):
                code = process.poll()
                if code is None:
                    continue
                del self.active[physical]; row = rows[key]; row['exit_code'] = code
                if code:
                    row['status'] = 'FAILED'; self.save()
                    raise RuntimeError('Preserve fresh-runtime worker failure: '+row['log'])
                path = self.work / 'neck72_fresh_runtime' / split / key / 'complete.json'
                try:
                    evidence = read(path)
                    assert evidence['status'] == 'COMPLETE_NECK72_FRESH_IMAGE_RUNTIME_MEASUREMENT'
                    assert evidence['specification_sha256'] == sha256(SPEC)
                    assert evidence['id'] == key and evidence['split'] == split
                    assert evidence['config'] == row['config']
                    assert not evidence['cached_bank_substitution_for_inference'] and not evidence['GT_read']
                    assert isinstance(evidence['source_equivalence_passed'], bool)
                except Exception:
                    row['status'] = 'FAILED_EVIDENCE'; self.save(); raise
                row.update(status='COMPLETE', evidence_sha256=sha256(path),
                           source_equivalence_passed=evidence['source_equivalence_passed'])
            for physical in ['4', '5']:
                if physical in self.active:
                    continue
                pending = [(k, j) for k, j in rows.items() if j['status'] == 'PENDING']
                if not pending:
                    continue
                key, row = pending[0]
                bound = self.specification['memory']['maximum_case_growth_bytes']
                try:
                    require_storage(self.work, bound*(len(self.active)+1))
                    require_host_memory(8*2**30, maximum_workers=2)
                except RuntimeError as error:
                    self.state.update(status='WAITING_RUNTIME_RESOURCE_HEADROOM', resource_wait_reason=str(error))
                    continue
                log = self.work / ('neck72.fresh_runtime.'+split+'.'+key+'.log')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical, OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(row['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                pid_stat = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
                row.update(status='RUNNING', pid=process.pid, start_ticks=int(pid_stat[19]),
                           physical_gpu=physical, log=str(log))
                self.active[physical] = key, process
                self.state.pop('resource_wait_reason', None)
            self.save()
            if not self.active and all(row['status'] == 'COMPLETE' for row in rows.values()):
                return
            time.sleep(10)


def export(work, state, selected, split):
    docs = ROOT / 'docs/fusion_research'; rows = []
    for case in selected['cases']:
        path = work / 'neck72_fresh_runtime' / split / case['id'] / 'complete.json'
        value = read(path); assert sha256(path) == state['phases'][split][case['id']]['evidence_sha256']
        public = docs / ('perceptual_neck72_fresh_runtime_'+split+'_20261124') / (case['id']+'.json')
        assert not public.exists()
        dump(public, value)
        rows.append(dict(id=case['id'], source_equivalence_passed=value['source_equivalence_passed'],
            valid_full_component_cost_for_preserved_accuracy=value['valid_full_component_cost_for_preserved_accuracy'],
            measured_FPS=value['measured_FPS'], mean_latency_ms=value['mean_latency_ms'],
            p95_latency_ms=value['p95_latency_ms'], peak_own_allocated_GPU_GiB=value['peak_own_allocated_GPU_GiB'],
            measured_frames=value['measured_frames'], actual_IR_image_read=value['actual_IR_image_read'],
            public_case_file=str(public.relative_to(ROOT)), public_case_sha256=sha256(public)))
    report = dict(status='COMPLETE_NECK72_FRESH_IMAGE_'+split.upper()+'_COST_MEASUREMENTS', rows=rows,
        all_frozen_cases_passed=all(row['source_equivalence_passed'] for row in rows),
        specification_sha256=sha256(SPEC), selection_sha256=sha256(work / 'neck72_Test_selection_frozen.json'),
        runtime_registration_sha256=sha256(work / REGISTRY), roles=selected['roles'], fixed_seed=0,
        no_Test_accuracy_used=True, all_negative_accuracy_cases_retained=True,
        full_Test15985frame_FPS=False, includes_offline_DiffV2IR_generation=False,
        scope=read(SPEC)['scope'], goal_met=False, supported_innovations=0)
    if split == 'dev':
        dump(work / 'neck72_fresh_runtime_Dev_engineering.json', report)
    stem = docs / ('PERCEPTUAL_MFT_20261124_NECK72_FRESH_RUNTIME_'+split.upper())
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# neck72逐条件完整组件输入测速', '', report['scope'], '',
        '每个条件独立从当前图像计算。表内速度仅在来源/轨迹检查通过时能对应冻结准确率；'
        '未通过者保留为工程诊断，不用较宽阈值或Test改选来补作通过。', '',
        '| 条件 | 来源/轨迹通过 | 测量FPS | 均值ms | P95ms | 自身GPU峰值GiB |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["id"]} | {row["source_equivalence_passed"]} | {row["measured_FPS"]:.3f} | '
            f'{row["mean_latency_ms"]:.3f} | {row["p95_latency_ms"]:.3f} | {row["peak_own_allocated_GPU_GiB"]:.3f} |')
    lines += ['', read(SPEC)['timers'], '', '没有新的准确率、架构或贡献证明，Goal仍需完整科学验收。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return report


def run():
    _, _, work, specification, registration = registered(verify_artifacts=True)
    path = work / 'neck72_fresh_runtime_state.json'
    lock = path.with_suffix('.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists(), 'Preserve prior runtime-controller evidence'
    state = dict(status='WAITING_FOR_COMPLETE_NECK72_ACCURACY_AND_ACTUAL_PROCESS_EXIT',
        controller_pid=os.getpid(), specification_sha256=sha256(SPEC), phases={},
        fixed_seed=0, physical_gpus=['4', '5'], maximum_workers=2, goal_met=False)
    queue = Queue(work, state, specification); queue.save()
    try:
        while not predecessor_ready(work):
            assert all(sha256(ROOT / p) == h for p, h in registration['source_hashes'].items())
            time.sleep(10)
        selected = cases(work)
        queue.jobs('dev', selected['cases']); proof = export(work, state, selected, 'dev')
        if not proof['all_frozen_cases_passed']:
            state['status'] = 'NECK72_RUNTIME_DEV_MISMATCH_PRESERVE_AND_DEBUG'; queue.save()
            return
        queue.jobs('test', selected['cases']); result = export(work, state, selected, 'test')
        state['status'] = ('COMPLETE_NECK72_FULL_COMPONENT_PREFIX_RUNTIME_SOURCE_MATCHED'
            if result['all_frozen_cases_passed'] else 'NECK72_RUNTIME_TEST_SOURCE_MISMATCH_PRESERVED')
        queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_NECK72_FRESH_RUNTIME', error=repr(error)); queue.save()
        # Already-running siblings remain tracked; never terminate/restart a
        # healthy worker merely because another worker or observation failed.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().action]()
