#!/usr/bin/env python3
"""Whole raw causal video: decode -> IR -> padded8V3 -> padded128RGB -> tracker."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
from cfp.raw_profile_tail_runtime import RawProfileTailValidatedRuntime as RawProfileDetectorRuntime
from cfp.online import CurrentFrameFrontend,proposal_crop_arrays,jpeg_response,V3_CHECKPOINT_SHA256
from cfp.pseudo_ir import CausalPseudoIR
from cfp.dinov2_native_encoder import CurrentDinoIdentityEncoder
from cfp.visible_token_native_encoder import CurrentVisibleTokenIdentityEncoder
from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.lifecycle_crossover import load_bot_reference,make_cfp_lifecycle
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256,snapshot_sources,write_json
from yolox.tracker.basetrack import BaseTrack


def idle(physical):
    if physical not in ['0','1','2','3']:raise ValueError('Onlyapproved GPU0–3')
    values=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,util=[int(x.strip()) for x in values.split(',')]
    if memory>64 or util:raise ValueError('GPU notidle, preserveotherjobs')


def main():
    ap=argparse.ArgumentParser()
    for name in ['config','root','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--asset-root',type=Path,required=True)
    ap.add_argument('--video',required=True)
    ap.add_argument('--kind',choices=['visible_token_masked','visible_token_target'],required=True)
    ap.add_argument('--training',type=Path,required=True)
    ap.add_argument('--visible-mode',choices=['masked_visible_tokens','visible_target'],required=True)
    ap.add_argument('--visible-seed',type=int,choices=[0,1,2],required=True)
    ap.add_argument('--expected-inner',required=True)
    ap.add_argument('--expected-fit',nargs='+',required=True)
    a=ap.parse_args();c=json.loads(a.config.read_text())
    if c['state']!='FROZEN' or a.video not in c['videos'] or a.output.exists():
        raise ValueError('New declared visible-token full-video raw profile required')
    expected_kind = {'masked_visible_tokens':'visible_token_masked', 'visible_target':'visible_token_target'}[a.visible_mode]
    if a.kind != expected_kind:
        raise ValueError('Visible-token kind/mode mismatch')
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','');idle(physical)
    if shutil.disk_usage(a.root).free<40*2**30:raise OSError('40GiBreserve')
    data=Path(c['data_root']);index=a.asset_root/'identity_two_crop_index_v1';im=json.loads((index/'manifest.json').read_text());path=index/(a.video+'.frames.json');rows=json.loads(path.read_text());detections=Path(c['detections'])/(a.video+'.npz');dm=json.loads((Path(c['detections'])/'manifest.json').read_text())
    if sha256(path)!=im['videos'][a.video]['frame_paths_sha256'] or sha256(detections)!=dm['caches'][a.video]['sha256']:raise ValueError('Immutable currentframe/proposal provenance changed')
    with np.load(detections,allow_pickle=False) as source:frames,offsets,expected_boxes,expected_scores=[source[k].copy() for k in ['frame_ids','offsets','boxes','scores']]
    if not np.array_equal(frames,np.arange(1,len(frames)+1)) or [r['frame_id'] for r in rows]!=frames.tolist():raise ValueError('Every native frame required')
    if sha256(c['detector_checkpoint'])!=V3_CHECKPOINT_SHA256:raise ValueError('FrozenV3 changed')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0);idle(physical);torch.cuda.set_per_process_memory_fraction(.2)
    encoder=None;predictor=None
    encoder=CurrentVisibleTokenIdentityEncoder(c['dino_reference'],c['dino_weights'],c['dino_source_proof'],a.training,a.video,a.visible_mode,a.visible_seed,a.expected_fit,a.expected_inner,'cuda',128)
    encoder_meta=encoder.metadata
    detector=RawProfileDetectorRuntime(c['detector_python'],c['detector_checkpoint'],c['scripts_root']);generator=CausalPseudoIR(c['scripts_root']);a.output.mkdir()
    sources=[__file__,a.config,'cfp/raw_profile_tail_runtime.py','tools/cfp_detector_worker_raw_profile_v2.py','cfp/detector_runtime.py','cfp/detector_runtime_device_pad.py','cfp/online.py','cfp/pseudo_ir.py','cfp/reid.py','cfp/dinov2_native_encoder.py','cfp/dinov2_identity.py','cfp/dinov2_fish_encoder.py','cfp/visible_token_native_encoder.py','cfp/native_local_identity.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/association.py','cfp/predictive_association.py','cfp/history.py','cfp/provenance.py','tools/cfp_cache_detection_features.py']
    report=dict(status='RUNNING',config=c,config_sha256=sha256(a.config),video=a.video,kind=a.kind,visible_mode=a.visible_mode,visible_seed=a.visible_seed,training_checkpoint=str(a.training),expected_fit=a.expected_fit,expected_inner=a.expected_inner,asset_root=str(a.asset_root),physical_gpu=physical,frames=len(frames),completed_frames=0,encoder=encoder_meta,detector=dict(detector.metadata),detection_reference_sha256=sha256(detections),frame_paths_sha256=sha256(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=snapshot_sources(a.output/'sources',sources),scope=c['scope']+' Visible-token seed-specific current-only raw profile; no cached embedding reference is fed online.',loaded_GT_annotation_labels=False,precomputed_IR_used_for_inference=False,detector_IR_origin='CurrentRGB_prefix_EMA_generated_by_parent',cached_detection_or_feature_used_for_inference=False,feature_reference_comparison='not_applicable_visible_token_current_only')
    write_json(a.output/'manifest.json',report);BaseTrack._count=0
    if predictor:tracker=make_cfp_lifecycle(load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT'),'gru',predictor,.3)
    else:tracker=PredictiveAssociationTracker(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False),frame_rate=25,predictor=RecentIdentityControl('fifo'),memory='predictor',history_view='recent',appearance_weight=.3,proximity_iou=.1,ema=.9)
    stage={};latencies=[];parity=dict(current_IR_pixel_equal_frames=0,proposal_byte_equal_frames=0,maximum_feature_component_difference=0.,feature_rows=0);row_count=0;proposal_index=0;digest=hashlib.sha256();observed_boxes=[];observed_scores=[];observed_embeddings=[];observed_offsets=[0];start_all=time.perf_counter()
    try:
        with (a.output/(a.video+'.txt')).open('w') as stream:
            for i,frame in enumerate(frames):
                start_frame=time.perf_counter();start=time.perf_counter();rgb=cv2.imread(str(data/'train'/rows[i]['rgb_file']));decode=time.perf_counter()-start
                if rgb is None:raise ValueError('CurrentRGB decodefailed')
                start=time.perf_counter();pseudo=generator(rgb);ir_generate=time.perf_counter()-start;start=time.perf_counter();ir=jpeg_response(pseudo);ir_jpeg=time.perf_counter()-start
                start=time.perf_counter();current,det_times=detector.process(rgb,ir);detector_rtt=time.perf_counter()-start;boxes,scores=current['boxes'],current['scores'];lo,hi=offsets[i:i+2]
                result=encoder.encode(rgb,boxes);embedding=result['embedding'];encoder_times=result['stage_seconds']
                start=time.perf_counter();indices=np.arange(proposal_index,proposal_index+len(boxes));proposal_index+=len(boxes)
                if predictor:tracks=tracker.update(np.c_[boxes,scores],embedding,indices)
                else:tracks=tracker.update(np.c_[boxes,scores],(1,1),(1,1),embedding,proposal_indices=indices)
                association=time.perf_counter()-start;start=time.perf_counter()
                for t in tracks:
                    if np.prod(t.tlwh[2:])<=100:continue
                    index=int(t.current_proposal_index)-int(indices[0]) if len(indices) else -1
                    if t.frame_id!=frame or not 0<=index<len(boxes):raise ValueError('Current matched measurement required')
                    b=boxes[index].copy();b[2:]-=b[:2];stream.write(f'{frame},{int(t.track_id)},{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f},{float(t.score):.6f},-1,-1,-1\n');row_count+=1
                output=time.perf_counter()-start;elapsed=time.perf_counter()-start_frame;latencies.append(elapsed)
                times=dict(RGB_decode_seconds=decode,causal_IR_generation_seconds=ir_generate,IR_JPEG_roundtrip_seconds=ir_jpeg,detector_IPC_full_RTT_seconds=detector_rtt,association_and_memory_seconds=association,measurement_output_seconds=output,**encoder_times)
                for name,value in times.items():stage[name]=stage.get(name,0.)+value
                # Nested worker timings are reported separately, never added to
                # completecriticalpathRTT a second time.
                for name,value in det_times.items():report.setdefault('detector_nested_stage_seconds',{})[name]=report.get('detector_nested_stage_seconds',{}).get(name,0.)+value
                # Save read-only current outputs for an independent post-run replay, never inputs.
                observed_boxes.append(boxes.copy());observed_scores.append(scores.copy());observed_embeddings.append(embedding.copy());observed_offsets.append(observed_offsets[-1]+len(boxes))
                # Validation-only current references are read AFTER measured
                # inference/output. They never feed tracker or future state.
                reference=cv2.imread(str(data/'train'/rows[i]['ir_file']));equal=np.array_equal(ir,reference);parity['current_IR_pixel_equal_frames']+=int(equal)
                proposal_equal=np.array_equal(boxes,expected_boxes[lo:hi]) and np.array_equal(scores,expected_scores[lo:hi]);parity['proposal_byte_equal_frames']+=int(proposal_equal)
                if not equal:raise ValueError('ActualcurrentIRpixel parity failed at frame'+str(frame))
                if not proposal_equal:
                    tail_size=sum(v['frames'] for v in dm['caches'].values())%8
                    final_video=list(dm['caches'])[-1]
                    first_tail=int(frames[-1])-tail_size+1
                    if dm['config']['batch_size']!=8 or a.video!=final_video or not tail_size or frame<first_tail:raise ValueError('Non-tail currentV3proposal discrepancy')
                    validation=detector.validate_tail(rgb,ir,tail_size,int(frame-first_tail))
                    if not np.array_equal(validation['boxes'],expected_boxes[lo:hi]) or not np.array_equal(validation['scores'],expected_scores[lo:hi]):raise ValueError('Oldpartialbatch cache not exactly reconstructed fromcurrent-only pixels')
                    report.setdefault('detector_cache_tail_shape_validation',[]).append(dict(frame=int(frame),original_batch_size=int(tail_size),normal_online_batch=8,online_boxes_sha256=hashlib.sha256(boxes.tobytes()).hexdigest(),online_scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),current_only_zeros_other_positions=True,reference_reconstructed_exactly=True,online_proposals_preserved=True,online_rows=len(boxes),reference_rows=int(hi-lo)))
                    # The old reference encoder saw different proposal boxes at
                    # this verified cache-tail frame. Comparing those features
                    # would mix coordinates; do not loosen prefix tolerances.
                    report.setdefault('feature_comparison_excluded_detector_tail_frames',[]).append(int(frame))
                    digest.update(int(frame).to_bytes(8,'little'));digest.update(hashlib.sha256(rgb.tobytes()).digest());digest.update(hashlib.sha256(ir.tobytes()).digest());digest.update(boxes.tobytes());digest.update(scores.tobytes())
                    report.update(completed_frames=i+1,output_rows=row_count,stage_seconds=stage,reference_parity=parity)
                    write_json(a.output/'manifest.json',report)
                    continue
                parity['feature_rows']+=len(embedding)
                digest.update(int(frame).to_bytes(8,'little'));digest.update(hashlib.sha256(rgb.tobytes()).digest());digest.update(hashlib.sha256(ir.tobytes()).digest());digest.update(boxes.tobytes());digest.update(scores.tobytes())
                report.update(completed_frames=i+1,output_rows=row_count,stage_seconds=stage,reference_parity=parity)
                if (i+1)%32==0 or i+1==len(frames):
                    write_json(a.output/'manifest.json',report);print(json.dumps(dict(video=a.video,kind=a.kind,frames=i+1,total=len(frames),measured_seconds=sum(latencies))),flush=True)
                    if shutil.disk_usage(a.root).free<40*2**30:raise OSError('40GiBreserve')
        trace=a.output/'actual_current_outputs.npz';np.savez_compressed(trace,frame_ids=frames,offsets=np.asarray(observed_offsets,np.int64),boxes=np.concatenate(observed_boxes),scores=np.concatenate(observed_scores),embedding=np.concatenate(observed_embeddings));report['actual_current_output_trace_sha256']=sha256(trace)
        report.update(status='COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE',measured_critical_path_seconds=sum(latencies),whole_raw_FPS=len(frames)/sum(latencies),frame_latency_p50_p95_ms=(1000*np.percentile(latencies,[50,95])).tolist(),elapsed_with_validation_seconds=time.perf_counter()-start_all,current_RGB_IR_proposal_digest=digest.hexdigest(),prediction_sha256=sha256(a.output/(a.video+'.txt')),parent_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),timing_scope='Perframeactualcomplete criticalpath, includingRGBdecode/causalIR/JPEG/IPC+prep+batch8det+NMS/currentRGBcrop+pad128encode/association+memory+output. Referencevalidation/progressserialization/diagnosticoutputtrace/loadstartup excluded andreportedelapsed separately; no future/cachedinputs. Detectornestedtimings notdouble-counted. Single shared-machine run, norepeatability claim.')
        report['cache_reference_batch_policy']='OriginalglobalDataLoader batches8 withunpadfinalpartialbatch; declaredrawonlinealwaysrepeatscurrent8. Verifiedtail proposalsmaydiffer; no visible-token cached feature/prediction is used online.'
        report['cache_reference_prediction_byte_equal']=False
        report['cache_reference_prediction_comparison']='not_applicable_visible_token_no_cached_prediction_reference'
        write_json(a.output/'manifest.json',report)
    except Exception as e:
        report.update(status='FAILED_ACTUAL_RAW_EXECUTION',error=str(e));write_json(a.output/'manifest.json',report);raise
    finally:
        report['detector_shutdown']=detector.close();write_json(a.output/'manifest.json',report)

if __name__=='__main__':main()
