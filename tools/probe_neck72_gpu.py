#!/usr/bin/env python3
"""Actual per-card pixel/source,20-update and zero-fallback gates for neck72."""
import copy
import gc
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools import extract_native_neck_eval_features as pixels
from tools.eval_mft26ir import write_sequence_results
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_experiment import registered, config_for, STUDY
from tools.neck72_runtime import REGISTRY as WORKER_REGISTRY, read, require_gpu, require_storage
from tools.neck72_smoke_binding import make_smoke
from tools.run_neck72_replay import load_sequence, tracker_for
from tools.run_pseudo_ir_comparison import dump, sha256


def pixel_prefix(physical, cue, sequence, frames, work):
    namespace, captured = dict(pixels.main.__globals__), []
    def load(which, seed, root, protocol):
        model, info = pixels.load_native_encoder(which, seed, root, protocol)
        assert seed == 0
        captured.append((model, state_sha256(model.state_dict()), info))
        return model, info
    namespace['load_native_encoder'] = load
    entry = types.FunctionType(pixels.main.__code__, namespace)
    assert entry.__code__ is pixels.main.__code__
    previous = sys.argv
    try:
        sys.argv = ['pixel-prefix', '--cue', cue, '--split', 'dev', '--sequence', sequence,
            '--seed-only', '0', '--max-frames', str(frames),
            '--output-family', 'native_neck_seed0_eval_features']
        entry()
    finally:
        sys.argv = previous
    assert len(captured) == 1
    model, before, info = captured[0]
    assert state_sha256(model.state_dict()) == before
    directory = work / 'native_neck_seed0_eval_features/dev' / sequence / cue / 'seed_0' / ('smoke_'+str(frames))
    complete = read(directory / 'complete.json')
    assert complete['status'] == 'SMOKE_COMPLETE' and complete['frames'] == frames
    for arm in ['new', 'rgb']:
        old = work / 'native_neck_eval_features/dev' / sequence / cue / 'seed_0' / (arm+'.npz')
        with np.load(directory / (arm+'.npz')) as current, np.load(old) as source:
            count = int(source['offsets'][frames])
            for key in current.files:
                expected = source[key][:frames] if key == 'frame_ids' else source[key][:frames+1] if key == 'offsets' else source[key][:count]
                np.testing.assert_array_equal(current[key], expected)
    result = dict(cue=cue, sequence=sequence, actual_frames=frames,
        original_GPU_seed0_prefix_arrays_exact=True, frozen_encoder_weights_and_BN_unchanged=True,
        encoder_loaded_parameters=sum(p.numel() for p in model.parameters()),
        encoder_active_parameters=sum(p.numel() for sub in [model.backbone, model.projection, model.neck]
                                      for p in sub.parameters()),
        checkpoint=info, complete_path=str(directory / 'complete.json'),
        complete_sha256=sha256(directory / 'complete.json'))
    del entry, namespace, captured, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def zero_fallback(plan, work, study, head, checkpoint, directory):
    outputs = []
    for algorithm in ['ordered', 'consensus']:
        case = config_for(study, head, 6, algorithm)
        config = case['config']; config['native_query_checkpoint'] = str(checkpoint)
        data, _ = load_sequence(plan, work, 'dev', 'BT-005', config)
        paths, posteriors = [], []
        for intervention in ['missing', 'exact_zero']:
            current = dict(config, disable_native_increment=intervention == 'missing')
            tracker = tracker_for(current)
            frozen_RGB = state_sha256(tracker.model.state_dict())
            inputs = dict(data)
            if intervention == 'missing':
                inputs.pop('native_auxiliary')
                inputs['quality'] = data['quality'].copy()
                inputs['quality'][:, 1] = np.nan
            if intervention == 'exact_zero':
                with torch.no_grad():
                    tracker.native_model.net[-1].weight.zero_(); tracker.native_model.net[-1].bias.zero_()
            rows = []
            for frame in range(1, 9):
                rows.extend(tracker.update(frame, inputs))
            assert inputs['offsets'][1] > 0 and inputs['offsets'][8] > inputs['offsets'][7]
            packet = tracker.query_posterior(8, dict(frame=1, index=int(inputs['offsets'][0])))
            assert packet['probability'].size > 0 and not np.any(packet['delta'])
            posteriors.append({name: packet[name].copy() for name in
                               ['probability', 'null', 'RGB_probability', 'RGB_null']})
            path = directory / (head['id']+'_'+algorithm+'_'+intervention+'.txt')
            write_sequence_results(str(path), rows); paths.append(path)
            assert state_sha256(tracker.model.state_dict()) == frozen_RGB
        assert paths[0].read_bytes() == paths[1].read_bytes()
        for key in posteriors[0]:
            np.testing.assert_array_equal(posteriors[0][key], posteriors[1][key])
        outputs.append(dict(algorithm=algorithm, frames=8, missing_equals_exact_zero_tracks=True,
                            actual_nonempty_query_posteriors_exact=True,
                            missing_auxiliary_absent_and_IR_quality_NaN_ignored=True,
                            files=[dict(path=str(p), sha256=sha256(p)) for p in paths]))
    return outputs


def main():
    plan, protocol, work, study, _ = registered()
    physical, uuid = require_gpu(work, role='probe')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    directory = work / 'neck72_actual_GPU_engineering' / ('gpu'+physical)
    assert not (directory / 'complete.json').exists(), 'Preserve per-card actual engineering evidence'
    # Distinct subdirectories avoid two cards writing the same producer output.
    frames = 8 if physical == '4' else 9
    pixel_proofs = [pixel_prefix(physical, cue, sequence, frames, work)
                    for sequence in protocol['dev'] for cue in ['rgb', 'ir']]
    rows = []
    for head in study['heads']:
        entry, binding, captured = make_smoke(head['kind'], physical)
        previous = sys.argv
        try:
            sys.argv = ['head-smoke', '--frontend', head['frontend'], '--cue', head['cue'],
                        '--seed', '0', '--epochs', '18']
            entry()
        finally:
            sys.argv = previous
        assert len(captured) == 1 and all(state_sha256(m.state_dict()) == before for m, before in captured)
        output = work / binding['output_family'] / f'{head["frontend"]}_{head["cue"]}_s0'
        complete = read(output / 'complete.json')
        assert complete['status'] == 'SMOKE_NECK72_20_TRAIN_UPDATES' and complete['epochs'] == 1
        assert len(complete['history']) == 1 and complete['history'][0]['steps'] == 20
        assert not complete['Dev_fit_rows'] and not complete['Test_fit_rows'] and complete['seed'] == 0
        fallback = zero_fallback(plan, work, study, head, output / 'latest.pt', directory / 'zero_fallback')
        rows.append(dict(head=head['id'], binding=binding, Train_updates=20,
            frozen_RGB_core_unchanged=True, zero_fallback=fallback,
            complete_path=str(output / 'complete.json'), complete_sha256=sha256(output / 'complete.json')))
        del entry, captured
        gc.collect(); torch.cuda.empty_cache(); require_storage(work)
    assert len(rows) == 8
    result = dict(status='COMPLETE_NECK72_GPU_INPUT_OPTIMIZER_AND_EXACT_ZERO_FALLBACK',
        physical_gpu=physical, GPU_uuid=uuid, fixed_seed=0,
        worker_interfaces_sha256=sha256(work / WORKER_REGISTRY), study_sha256=sha256(STUDY),
        Train_updates_each=20, matched_conditions_checked=8, pixel_prefix_proofs=pixel_proofs,
        rows=rows, RGB_core_and_encoder_BN_unchanged=True, Dev_or_Test_fitting=False,
        GT_evaluation=False, formal_fit_started=False, source_sha256=sha256(__file__),
        scope='Actual card-specific engineering only; pixel prefixes8/9frames, exact cached source '
              'comparison, original20Train updates and missing-versus-zero8frame tracks. '
              'No formal18epoch claim or HOTA measurement.')
    dump(directory / 'complete.json', result)
    print(physical, result['status'], flush=True)


if __name__ == '__main__':
    main()
