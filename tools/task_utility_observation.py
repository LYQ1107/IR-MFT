"""Sequential task-utility packet/track receipts; no GT or GPU authorization."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tools.evaluate_native_pyramid import packet
from tools.fusion_memory_observation import LENGTHS,write_json
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.run_pseudo_ir_comparison import sha256


def replay_canonical_sequence(plan,split,sequence,directory,provenance,*,resource_check=lambda:None):
    """Replay an entire original RGB packet file; no CUDA/model selection or GT."""
    from tools.task_utility_design import CANONICAL
    if split not in ['dev','test'] or sequence not in LENGTHS[split]:
        raise ValueError('Complete original canonical Dev/Test sequence required')
    directory=Path(directory);count=LENGTHS[split][sequence]
    destination=directory/'sequence_manifests'/(sequence+'.canonical.json')
    if destination.exists():raise ValueError('Preserve previous canonical replay')
    source=Path(plan['frozen_detection_sources']['rgb_'+split])/'detections'/(sequence+'.npz')
    original_sha=sha256(source);values=packet(source)
    if len(values['frame_ids'])!=count:raise ValueError('Canonical packets must cover the entire sequence')
    writer=StreamTrackWriter(directory,sequence,[CANONICAL['id']]);started=time.monotonic()
    try:
        resource_check()
        for frame in range(1,count+1):
            lo,hi=values['offsets'][frame-1:frame+1]
            writer.append(frame,{CANONICAL['id']:np.c_[values['boxes'][lo:hi],values['scores'][lo:hi]].astype(np.float32)})
            if frame%100==0:resource_check()
        streams=writer.close(count)
        if sha256(source)!=original_sha:raise ValueError('Original frozen packets changed during replay')
        value=dict(provenance,status='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT',split=split,
            sequence=sequence,frames=count,streams=streams,source_packet=str(source),source_packet_sha256=original_sha,
            GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,
            seconds=time.monotonic()-started,formal_authorization_established_here=False)
        resource_check();write_json(destination,value,exclusive=True);return value
    except Exception as error:
        write_json(directory/'sequence_manifests'/(sequence+f'.canonical_failure_{time.time_ns()}.json'),
            dict(error=repr(error),emitted_frames=writer.frames,declared_frames=count,split=split,sequence=sequence,
                 complete_receipt=False,GT_read=False,prior_outputs_retained=True),exclusive=True)
        raise
    finally:writer.abort()


def observe_sequence(engine,images,models,directory,expected_image_delivery_sha256,provenance,*,
                     resource_check=lambda:None,verify_streams=lambda streams:None,diagnostics=lambda:{}):
    """Complete declared image range; Train prefixes remain engineering only."""
    names=[m['id'] for m in models]
    if [m['id'] for m in engine.rows]!=names or images.last_frame!=0 or engine.sequence is not None:
        raise ValueError('Fresh ordered current-image engine matching the exact recipes required')
    sequence,split,count=images.sequence,images.split,images.frames
    if split in ['dev','test'] and count!=images.complete_source_frames:
        raise ValueError('No truncated formal observations')
    directory=Path(directory);manifest=directory/'sequence_manifests'/(sequence+'.json')
    if manifest.exists():raise ValueError('Preserve original complete observations')
    writer=StreamTrackWriter(directory,sequence,names);source=hashlib.sha256()
    prefixes={n:hashlib.sha256() for n in names};source_rows=[];frame=0;started=time.monotonic()
    try:
        resource_check()
        for frame in range(1,count+1):
            rgb,ir,row=images.read_frame(frame,engine.needs_actual_IR_image)
            source.update(json.dumps(row,separators=(',',':')).encode()+b'\n')
            packets,information=engine.step(rgb,ir,sequence,frame)
            if information['GT_read'] or information['future_frame_input'] or information['previous_outputs_rewritten']:
                raise ValueError('Current image decisions may not read GT/future or rewrite past outputs')
            writer.append(frame,packets)
            if frame<=32:
                source_rows.append(row)
                for name,values in packets.items():
                    prefixes[name].update(np.asarray([frame,len(values)],dtype='<i8').tobytes())
                    prefixes[name].update(np.asarray(values,dtype='<f4').tobytes())
            if frame%100==0 or frame==count:
                writer.flush();resource_check()
                write_json(directory/'sequence_manifests'/(sequence+'.progress.json'),
                    dict(frame=frame,frames=count,split=split,sequence=sequence,models=len(models),
                         seconds=time.monotonic()-started,GT_read=False,complete_receipt=False))
        if source.hexdigest()!=expected_image_delivery_sha256:raise ValueError('Actual fixed image-byte delivery differs')
        streams=writer.close(count)
        if verify_streams(streams) is False:raise ValueError('Original source/stream postcondition failed')
        prefix_tracks={}
        for name,stream in streams.items():
            digest=hashlib.sha256()
            with Path(stream['track_path']).open() as handle:
                for line in handle:
                    if int(line.split(',',1)[0])>32:break
                    digest.update(line.encode())
            prefix_tracks[name]=digest.hexdigest()
        value=dict(status='COMPLETE_TASK_UTILITY_TRAIN_PREFIX_ENGINEERING_OBSERVATIONS' if split=='train' else
            'COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT',split=split,sequence=sequence,frames=count,
            complete_source_sequence_frames=images.complete_source_frames,formal_Dev_Test_sequence_coverage=split in ['dev','test'],
            models=models,provenance=provenance,streams=streams,original_image_bytes_delivery_sha256=source.hexdigest(),
            prefix32_source_rows=source_rows,prefix_frames=min(32,count),
            prefix32_packet_sha256={k:v.hexdigest() for k,v in prefixes.items()},prefix32_literal_MOT_sha256=prefix_tracks,
            gate_statistics=diagnostics(),actual_IR_byte_frames_hashed=images.hashed_IR_frames,
            actual_IR_frames_decoded=images.decoded_IR_frames,unused_IR_bytes_hashed_for_provenance=True,
            independent_input_batch_size=1,GT_read=False,future_frame_input=False,previous_outputs_rewritten=False,
            current_image_cache=False,feature_or_detection_disk_cache=False,formal_authorization_established_here=False,
            seconds=time.monotonic()-started,runtime_scope='Shared instrumented production;not individual methodFPS.')
        resource_check();write_json(manifest,value,exclusive=True);return value
    except Exception as error:
        write_json(directory/'sequence_manifests'/(sequence+f'.failure_{time.time_ns()}.json'),
            dict(error=repr(error),attempted_frame=frame,emitted_frames=writer.frames,declared_frames=count,
                 split=split,sequence=sequence,complete_receipt=False,GT_read=False,prior_outputs_retained=True),exclusive=True)
        raise
    finally:writer.abort()
