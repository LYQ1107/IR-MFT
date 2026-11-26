#!/usr/bin/env python3
"""Read the fixed November generator's complete Test timing provenance on CPU.

The saved SAM and DiffV2IR timers cover separate historical inference regions.
Their sum is a component cost, never a newly measured end-to-end system FPS.
No model, image, source metadata or active experiment is modified.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sqlite3
import time


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'BT-002': 493, 'BT-004': 763, 'MSK-003': 629, 'PF-002': 3000,
          'SN-009': 4500, 'SN-011': 4500, 'SN-014': 2100}
CAPTION = 'there are goldfish swimming in the fish tank'
PREPROCESSING = ('aspect-preserving PIL-BILINEAR RGB, 127 pad; NEAREST original SAM, '
                 '0 pad; crop pad and BILINEAR restore; BT.601 uint8 gray, no contrast normalization')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def number(value, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and (value > 0 if positive else value >= 0), 'Invalid recorded timer')
    return float(value)


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use an explicitly CPU-only environment')
    started = time.monotonic()
    output.mkdir(parents=False, exist_ok=False)
    producer = ROOT / 'diffv2ir_final_20261104'
    planpath = ROOT / 'configs/perceptual_mft_20261122.json'
    protocolpath = ROOT / 'configs/cfp/protocol_v2.json'
    configpath = producer / 'final_config.json'
    summarypath = producer / 'plans/queue_summary.json'
    database = producer / 'queue.sqlite3'
    plan, protocol, config, summary = map(read, [planpath, protocolpath, configpath, summarypath])
    generated = Path(plan['generated_ir_root'])
    require(str(generated) == config['output_root'], 'Current fixed PNG source differs')
    require(protocol['historical_test'] == list(COUNTS), 'Original complete Test sequence order changed')
    require(summary['phases']['test'] == dict(frames=15985, sequences=COUNTS), 'Generation frame coverage differs')
    require(config['setting']['fixed_caption'] == CAPTION and config['run_BLIP'] is False and
            config['steps'] == 25 and config['seed_every_frame'] == 20261101,
            'The fixed November generation configuration changed')
    sources = [Path(__file__).resolve(), planpath, protocolpath, configpath, summarypath, database,
               producer/'scripts/config_worker.py', producer/'scripts/final_aux_worker.py',
               producer/'scripts/diff_worker.py', producer/'scripts/production_common.py',
               producer/'evidence/final_audit.json', producer/'progress.json']
    source_hashes = {str(p): sha(p) for p in sources}
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute(
            "SELECT * FROM frames WHERE phase='test' ORDER BY sequence,frame")]
    require(len(rows) == 15985 and all(r['completed'] == 1 for r in rows), 'Incomplete original generation queue')
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row)
    require(dict((s, len(v)) for s, v in grouped.items()) == COUNTS, 'Queue sequence coverage changed')
    fixed_identity = dict(generator='DiffV2IR/M3FD/EMA', weight_sha256=config['weight_sha256'],
        input_size=[512, 512], steps=25, sampler='Euler ancestral', cfg=[7.5, 1.5, 1.5],
        seed=20261101, precision='FP32 parameters FP16 autocast TF32',
        preprocessing=PREPROCESSING, caption=CAPTION, caption_policy='fixed')
    expected_sam = dict(backbone='vit_b', points_per_side=32, points_per_batch=64,
        pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0, min_mask_region_area=0)
    aggregate, samples, receipts = [], [], []
    all_metadata = hashlib.sha256()
    for sequence, length in COUNTS.items():
        records = grouped[sequence]
        require([r['frame'] for r in records] == list(range(1, length+1)), 'Missing or duplicated original frames')
        values, devices, reused, cold = [], collections.Counter(), 0, 0
        metadata_digest, declared_input_digest = hashlib.sha256(), hashlib.sha256()
        peaks = collections.defaultdict(int)
        for row in records:
            frame = row['frame']
            metadata_path = generated/'_metadata/test'/sequence/'infrared_diffv2ir'/f'{frame:06d}.json'
            auxpath = generated/'_metadata/test'/sequence/'sam_mask'/f'{frame:06d}.json'
            payload, auxpayload = metadata_path.read_bytes(), auxpath.read_bytes()
            meta, aux = json.loads(payload), json.loads(auxpayload)
            identity, task = meta['identity'], meta['task']
            expected_identity = dict(fixed_identity, rgb_sha256=identity['rgb_sha256'], mask_sha256=identity['mask_sha256'])
            require(identity == expected_identity, f'Generator identity differs: {sequence}/{frame}')
            fingerprint = hashlib.sha256(encoded(identity)).hexdigest()
            require(meta['fingerprint'] == fingerprint == row['fingerprint'] and
                    meta['output_sha256'] == row['output_sha256'], 'Metadata and completed queue disagree')
            expected_rgb = str(Path(plan['data_root'])/'test'/sequence/'visible'/f'{frame:06d}.jpg')
            require(row['rgb'] == task['rgb'] == aux['rgb'] == expected_rgb and
                    task['sequence'] == aux['sequence'] == sequence and task['frame'] == aux['frame'] == frame and
                    task['queue_phase'] == task['split'] == 'test' and task['aux'] == row['aux'] == str(auxpath) and
                    task['output'] == row['output'] == str(generated/'test'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),
                    'Original frame mapping differs')
            require(aux['caption_source'] == 'fixed' and aux['caption'] == meta['caption'] == CAPTION and
                    aux['blip_seconds'] == 0 and aux['seed'] == meta['seed'] == 20261101 and
                    aux['rgb_sha256'] == identity['rgb_sha256'] and aux['mask_sha256'] == identity['mask_sha256'] and
                    all(aux['sam_settings'][k] == v for k, v in expected_sam.items()), 'Auxiliary provenance differs')
            generation = number(meta['seconds'], positive=True)
            sam_recorded = number(aux['sam_seconds'])
            is_reused = aux.get('sam_reused_from_validated_mask', False)
            require(not is_reused or 'sam_original_seconds' in aux, 'Reused SAM mask lost its original timing')
            sam_original = number(aux.get('sam_original_seconds', sam_recorded), positive=True)
            cold_frame = bool(meta['first_after_load'] or aux['first_after_load'])
            values.append((generation, sam_recorded, sam_original, cold_frame))
            reused += int(is_reused)
            cold += int(cold_frame)
            devices[str(meta['physical_gpu'])] += 1
            require(meta['gpu'] == aux['gpu'] == 'NVIDIA A100-SXM4-40GB', 'Historical timing hardware differs')
            for prefix, record in [('generator', meta), ('sam', aux)]:
                for field in ['peak_allocated_bytes', 'peak_reserved_bytes']:
                    peaks[prefix+'_'+field] = max(peaks[prefix+'_'+field], int(record[field]))
            entry = [sequence, frame, hashlib.sha256(payload).hexdigest(), hashlib.sha256(auxpayload).hexdigest()]
            line = encoded(entry)+b'\n'
            metadata_digest.update(line)
            all_metadata.update(line)
            declared_input_digest.update(encoded([sequence, frame, identity['rgb_sha256'], meta['output_sha256']])+b'\n')
            if frame <= 32 or frame == length:
                current = {'rgb': sha(task['rgb']), 'IR': sha(task['output']), 'SAM_mask': sha(aux['mask_path'])}
                require(current == dict(rgb=identity['rgb_sha256'], IR=meta['output_sha256'], SAM_mask=identity['mask_sha256']),
                        'Current sampled original image bytes differ from their generation record')
                samples.append(dict(sequence=sequence, frame=frame, **current))
        steady = [r for r in values if not r[3]]
        aggregate.append(dict(sequence=sequence, frames=length,
            generator_recorded_seconds=math.fsum(v[0] for v in values),
            SAM_recorded_production_seconds=math.fsum(v[1] for v in values),
            SAM_original_recorded_seconds=math.fsum(v[2] for v in values),
            serial_recorded_inference_component_seconds=math.fsum(v[0]+v[2] for v in values),
            serial_recorded_inference_component_seconds_per_frame=math.fsum(v[0]+v[2] for v in values)/length,
            first_after_either_model_load_frames=cold, steady_paired_frames=len(steady),
            steady_serial_recorded_component_seconds_per_frame=math.fsum(v[0]+v[2] for v in steady)/len(steady),
            SAM_reused_records=reused, generator_physical_GPU_counts=dict(devices),
            original_metadata_ordered_sha256=metadata_digest.hexdigest(),
            declared_original_RGB_IR_bytes_ordered_sha256=declared_input_digest.hexdigest(),
            current_RGB_IR_SAM_sampled_frames=sum(s['sequence'] == sequence for s in samples), **dict(peaks)))
        receipts.append(dict(sequence=sequence, frames=length, metadata_ordered_sha256=metadata_digest.hexdigest()))
        print(json.dumps(dict(status='VERIFIED_COMPLETE_ORIGINAL_SEQUENCE_TIMING_RECORDS', sequence=sequence,
                              frames=length, cumulative_frames=sum(r['frames'] for r in aggregate))), flush=True)
    require(len(samples) == 231 and sum(r['frames'] for r in aggregate) == 15985, 'Incomplete fixed audit')
    require(all(sha(p) == digest for p, digest in source_hashes.items()), 'An original source changed during the audit')
    total_seconds = math.fsum(r['serial_recorded_inference_component_seconds'] for r in aggregate)
    result = dict(status='COMPLETE_FIXED_NOVEMBER_FULL7_TEST_HISTORICAL_GENERATION_COMPONENT_COST_AUDIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), frames=15985, sequences=list(COUNTS),
        rows=aggregate, original_metadata_files_read=31970, metadata_ordered_sha256=all_metadata.hexdigest(),
        current_original_RGB_IR_SAM_sampled_frames=231, current_image_byte_checks=693,
        current_all15985_PNG_bytes_rehashed=False, original_complete_queue_output_hashes_match_all_metadata=True,
        original_48070_frame_generation_audit=source_hashes[str(producer/'evidence/final_audit.json')],
        source_hashes=source_hashes, input_digest_encoding="Canonical compact sorted JSON array [sequence,frame,RGB_SHA256,IR_SHA256] plus LF, in ascending frame order per sequence.",
        serial_recorded_inference_component_seconds=total_seconds,
        serial_recorded_inference_component_seconds_per_frame=total_seconds/15985,
        generation_timer_scope='Historical synchronized region config_worker.py:31: DiffV2IR run including its internal preprocessing; excludes RGB decode before timer and grayscale conversion, PNG/metadata save after timer, model loading and scheduler waits.',
        SAM_timer_scope='Historical synchronized final_aux_worker.py:69-75: SAM generator.generate(rgb); excludes RGB read/decode, union construction, mask serialization, model loading and scheduler waits.',
        summation_scope='Separate historical SAM-original and generator synchronized elapsed regions on shared A100 GPUs; their sum is component accounting, not GPU-exclusive compute time, simultaneous pipeline latency, end-to-end FPS or a fresh benchmark.',
        inherited_November2_BLIP_probe_used_as_current_cost=False, fixed_caption_BLIP_executed=False,
        model_loading_image_IO_scheduler_waits_and_downstream_MFT_costs_included=False,
        complete_generation_inclusive_system_runtime_available=False,
        remaining_cost_work='Combine declared component costs with each frozen model runtime using explicit scopes; actual full-system latency with generation, decode/encode, model lifetime and tracking remains unmeasured.',
        new_GPU_work=False, new_model_load_or_generation=False, Test_GT_or_accuracy_read=False,
        seed_for_new_experiment=None, current_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        active_experiment_or_original_data_modified=False, supported_innovations=0, goal_met=False,
        actual_CPU_wall_seconds=time.monotonic()-started,
        actual_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    write(output/'current_sampled_image_bytes.json', samples)
    write(output/'per_sequence_metadata_receipts.json', receipts)
    with (output/'per_sequence_component_costs.csv').open('x', newline='') as stream:
        fields=[k for k,v in aggregate[0].items() if not isinstance(v, dict)]
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(aggregate)
    write(output/'complete.json', result)
    print(json.dumps(dict(status=result['status'], frames=15985,
        serial_component_seconds_per_frame=result['serial_recorded_inference_component_seconds_per_frame'],
        CPU_seconds=result['actual_CPU_wall_seconds'], GPU_work=False)), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args=parser.parse_args()
    run(args.output_directory)
