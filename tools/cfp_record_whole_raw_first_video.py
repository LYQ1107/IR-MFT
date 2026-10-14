#!/usr/bin/env python3
"""Completed fullraw baselines only, numeric/provenance publication record."""
import argparse,json
from pathlib import Path
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Newimmutable actualcomplete report required')
    runs={}
    for kind,version in [('legacy_gru','v2'),('dino_frozen_fifo','v1')]:
        p=a.root/f'raw_current_{kind}_SN-001_{version}';m=json.loads((p/'manifest.json').read_text());metrics=Path(str(p)+'.trackeval/metrics.json');official=json.loads(metrics.read_text())
        assert m['status']=='COMPLETE_WHOLE_RAW_CURRENT_VIDEO_PROFILE' and m['frames']==m['completed_frames']==684 and m['cache_reference_prediction_byte_equal']
        assert m['reference_parity']['current_IR_pixel_equal_frames']==m['reference_parity']['proposal_byte_equal_frames']==684
        assert not m['loaded_GT_annotation_labels'] and not m['precomputed_IR_used_for_inference'] and not m['cached_detection_or_feature_used_for_inference'];assert sha256(p/'SN-001.txt')==m['prediction_sha256']==official['artifacts']['SN-001']['prediction_sha256']
        fields=['config','config_sha256','video','kind','physical_gpu','frames','completed_frames','detector','detection_reference_sha256','frame_paths_sha256','source_commit','sources','scope','loaded_GT_annotation_labels','precomputed_IR_used_for_inference','cached_detection_or_feature_used_for_inference','detector_IR_origin','output_rows','stage_seconds','reference_parity','detector_nested_stage_seconds','legacy_cache_tail_shape_validation','measured_critical_path_seconds','whole_raw_FPS','frame_latency_p50_p95_ms','elapsed_with_validation_seconds','current_RGB_IR_proposal_digest','prediction_sha256','parent_peak_gpu_allocated_bytes','timing_scope','cache_reference_prediction_byte_equal','detector_shutdown']
        evidence={k:m[k] for k in fields if k in m};encoder=m['encoder'];evidence['encoder']={k:encoder[k] for k in ['checkpoint_sha256','seed','embedding_stage','backbone_pretrained_sha256','source_commit','dimension','encoder_precision','coordinate_kind','coordinate_sha256','trained_encoder','fitting_scope'] if k in encoder}
        runs[kind]=dict(path=str(p),manifest_sha256=sha256(p/'manifest.json'),observed_with_validation_FPS=m['frames']/m['elapsed_with_validation_seconds'],evidence=evidence,official_metrics_path=str(metrics),official_metrics_sha256=sha256(metrics),official_metrics=official['metrics'])
    proofs={name:dict(path=str(a.root/name),sha256=sha256(a.root/name),evidence=json.loads((a.root/name).read_text())) for name in ['raw_current_causal_ir_prefix_parity_v1.json','legacy_encoder_tail_batch_shape_diagnosis_v1.json']}
    write_json(a.output,dict(status='IN_PROGRESS',completed_stage='COMPLETE_BOTH_FULL_RAW_SN001_CURRENT_PIPELINES',runs=runs,proofs=proofs,failed_legacy_v1=dict(path=str(a.root/'raw_current_legacy_gru_SN-001_v1/manifest.json'),manifest_sha256=sha256(a.root/'raw_current_legacy_gru_SN-001_v1/manifest.json'),actual_error=json.loads((a.root/'raw_current_legacy_gru_SN-001_v1/manifest.json').read_text())['error']),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),scope='684actualrawcurrentRGBframes forbothpipelines, generatedcausalRGBresponse(notrealIRsensor), trueV3FP16repeatcurrentbatch8/crop128/memory/association/output; referencespost-inferencevalidationonly. Single sharedmachine measurements, measuredcriticalpath andobservedvalidationwalltime separate. Legacyenc/GRUexposure/privateDINOpretrainoverlap explicit. Allsix/3seed/finalmethod/repeatability/novelty/fullgoal stillpending; no conditioned-cacheFPS substitution.'))
    print(json.dumps({kind:dict(FPS=row['evidence']['whole_raw_FPS'],wallFPS=row['observed_with_validation_FPS'],HOTA=row['official_metrics']['SN-001']['HOTA']) for kind,row in runs.items()}))

if __name__=='__main__':main()
