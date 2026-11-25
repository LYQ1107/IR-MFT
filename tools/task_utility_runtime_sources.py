"""Runtime references from original complete observations; no accuracy or CUDA."""
import hashlib
from pathlib import Path

import numpy as np

from tools.evaluate_native_pyramid import packet
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import read
from tools.observe_task_utility import verify_observations
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_design import CANONICAL,FRESH,require_test_freeze
from tools.task_utility_runtime import literal_prefix,packet_digest
from tools.task_utility_study import STUDY,REGISTRY

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'configs/perceptual_task_utility_runtime_20261125.json'


def runtime_policy():
    value=read(RUNTIME)
    if (value['study_sha256']!=sha256(STUDY) or value['online_B1_cases']!=25 or
            value['canonical_RGB_B4_cases']!=1 or value['total_individually_measured_cases']!=26 or
            value['prefix_frames_each_video']!=32 or value['warmup_frames_each_video']!=8 or
            value['timed_frames_each_case']!=168 or value['accuracy_or_training_budget_changed']):
        raise ValueError('Original frozen runtime clarification and same accuracy/training study required')
    policy=value['B4_policy']
    if (policy['batch_size']!=4 or policy['actual_distinct_observations'] is not True or
            policy['duplicate_current_frame_emulation'] is not False or
            policy['source_RGB_sha256']!=read(STUDY)['source_RGB_sha256'] or
            sha256(ROOT/policy['source_code'])!=policy['source_code_sha256']):
        raise ValueError('Actual original-source B4 policy required')
    return value


def runtime_routes(base,specification):
    """Every originally frozen Test object, including the distinct B4 reference."""
    frozen=require_test_freeze(base,specification);cases=frozen['cases']
    if len(cases)!=26 or len({c['id'] for c in cases})!=26:raise ValueError('All twenty-six original cost objects required')
    rows=[];fresh=False
    for item in cases:
        path=Path(item['config']);case=read(path)
        if (sha256(path)!=item['config_sha256'] or case['id']!=item['id'] or case['split']!='test' or
                case['specification_sha256']!=sha256(specification) or case['model']['id']!=case['id']):
            raise ValueError('Original frozen cost-case file changed')
        canonical=case['model']==CANONICAL
        fresh=fresh or case['model']==FRESH
        if case.get('canonical') is not canonical:raise ValueError('Historical source route must be explicit')
        rows.append(dict(item,NN_batch_size=4 if canonical else 1,
            route='historical_canonical_RGB_B4' if canonical else 'current_image_B1'))
    if not fresh or sum(r['NN_batch_size']==4 for r in rows)!=1 or sum(r['NN_batch_size']==1 for r in rows)!=25:
        raise ValueError('Exactly25B1methods and1originalB4reference required')
    return rows


def runtime_references(specification,config,base,registration,plan,protocol):
    policy=runtime_policy();case=read(config)
    if case['split']!='test':raise ValueError('Individual formal costs require the frozen full Test observations')
    routes=runtime_routes(base,specification)
    if dict(id=case['id'],config=str(config),config_sha256=sha256(config)) not in [
            {k:r[k] for k in ['id','config','config_sha256']} for r in routes]:
        raise ValueError('Case is absent from the original complete frozen Test recipes')
    manifest=verify_observations(specification,config,base,registration,plan,protocol)
    source_manifest=manifest;fresh_config=None
    if case.get('canonical'):
        matches=[r for r in routes if r['id']==FRESH['id']]
        if len(matches)!=1:raise ValueError('Original complete fresh B1image-byte source required for historical B4 cost')
        fresh_config=Path(matches[0]['config'])
        if read(fresh_config)['model']!=FRESH:raise ValueError('Original fresh RGB source descriptor changed')
        source_manifest=verify_observations(specification,fresh_config,base,registration,plan,protocol)
    directory=Path(case['stream_directory']);pins={};references={}
    def pin(path,expected=None):
        path=str(Path(path));digest=sha256(path)
        if expected is not None and digest!=expected:raise ValueError('Original runtime input changed: '+path)
        if path in pins and pins[path]!=digest:raise ValueError('Original runtime reference changed')
        pins[path]=digest
    for path in [RUNTIME,STUDY,Path(base)/REGISTRY,Path(base)/'Test_selection_frozen.json',Path(specification),Path(config)]:pin(path)
    if fresh_config is not None:pin(fresh_config)
    for sequence,count in LENGTHS['test'].items():
        original=directory/'sequence_manifests'/(sequence+'.json');current=read(original)
        if (current['status']!='COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT' or
                current['frames']!=count or current['prefix_frames']!=32 or current['GT_read'] or
                current['future_frame_input'] or current['previous_outputs_rewritten'] or
                current['provenance']['pipeline_registration_sha256']!=sha256(Path(base)/REGISTRY) or
                current['provenance']['specification_sha256']!=sha256(specification)):
            raise ValueError('Original complete current-image source and its32frame prefix required')
        if Path(source_manifest['sources'][sequence]['path'])!=original:
            raise ValueError('Original complete B1image-byte source path changed')
        pin(original,source_manifest['sources'][sequence]['sha256']);source_rows=current['prefix32_source_rows']
        if case.get('canonical'):
            fresh_stream=source_manifest['coverage'][sequence]
            pin(fresh_stream['track_path'],fresh_stream['track_sha256'])
        if case.get('canonical'):
            cp=Path(manifest['sources'][sequence]['path']);canonical=read(cp);pin(cp,manifest['sources'][sequence]['sha256'])
            packet_path=Path(canonical['source_packet']);pin(packet_path,canonical['source_packet_sha256'])
            values=packet(packet_path)
            if len(values['frame_ids'])!=count:raise ValueError('Complete original RGB packet source required')
            digest=hashlib.sha256()
            for frame in range(1,33):
                lo,hi=values['offsets'][frame-1:frame+1]
                observations=np.c_[values['boxes'][lo:hi],values['scores'][lo:hi]].astype(np.float32)
                packet_digest(digest,frame,observations)
            del values
            stream=canonical['streams'][case['id']];packet_sha=digest.hexdigest()
            track_sha=literal_prefix(stream['track_path'])
            source_kind='Original B4 frozen packets and independently replayed sequence-local MOT;actual B4NN must reproduce them.'
        else:
            stream=current['streams'][case['id']];packet_sha=current['prefix32_packet_sha256'][case['id']]
            track_sha=current['prefix32_literal_MOT_sha256'][case['id']]
            if literal_prefix(stream['track_path'])!=track_sha:raise ValueError('Actual original MOT prefix changed')
            source_kind='Original complete current-image B1production packets and literal MOT.'
        pin(stream['track_path'],stream['track_sha256'])
        references[sequence]=dict(id=case['id'],sequence=sequence,prefix_frames=32,source_rows=source_rows,
            packet_sha256=packet_sha,literal_MOT_sha256=track_sha,source_kind=source_kind,
            RGB_byte_reference_scope='Current full B1image-delivery manifest;historical B4image hashes are not retroactively certified.')
    return dict(case=case,policy=policy,references=references,file_hashes=pins,complete_manifest=manifest)


def unchanged_runtime_sources(evidence):
    if not all(sha256(p)==s for p,s in evidence['file_hashes'].items()):
        raise ValueError('Original frozen runtime references changed')
