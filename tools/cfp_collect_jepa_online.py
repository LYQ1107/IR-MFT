#!/usr/bin/env python3
"""Verify full causal artifacts, official results, latency and terminal-batch differences."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    root=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5');record=dict(status='COMPLETE',source_sha256=sha256(__file__),models={},scope='Full4834frame strictcausal ProtocolB, representative seed0J3/ordinaryTransformer. ProtocolA three-seed comparison independent. Timing includes currentdecode/causalIR/V3/NMS/crops/RGBencoding/student+association/output; shared GPU/CPUload, notdedicated deployment. Startup model loading excluded. CUDAallocated peaks perprocess, notwholeGPU NVMLmemory.')
    lines=['# Fish-JEPA complete causal online evaluation','',record['scope'],'','|Model|Frames|FPS|p50 ms|p95 ms|HOTA|AssA|DetA|IDF1|MOTA|IDSW|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in ['J3','ordinary_transformer']:
        folder=root/f'jepa_online_{variant}_seed0_full_v1';manifest=json.loads((folder/'manifest.json').read_text());evaluation=folder.with_name(folder.name+'_evaluation')/'metrics.json';official=json.loads(evaluation.read_text())
        if manifest['status']!='COMPLETE' or manifest['probe']:raise ValueError('Only completefullonline')
        all_times=[];stage_values={};boundary={}
        for seq,count in [('BT-005',1834),('SN-013',3000)]:
            pred=folder/(seq+'.txt');timing=folder/(seq+'.timing.jsonl');entry=manifest['videos'][seq]
            if entry['frames']!=count or sha256(pred)!=entry['prediction_sha256'] or sha256(pred)!=official['artifacts'][seq]['prediction_sha256'] or sha256(timing)!=entry['timing_sha256']:raise ValueError('Incomplete/changed artifacts')
            rows=[json.loads(line) for line in timing.read_text().splitlines()]
            if [r['frame'] for r in rows]!=list(range(1,count+1)):raise ValueError('Missing timingframes')
            if not np.isclose(sum(r['total_seconds'] for r in rows),entry['pipeline_seconds'],rtol=1e-12):raise ValueError('Timing sum drift')
            all_times.extend(r['total_seconds'] for r in rows)
            for row in rows:
                for key,value in row['stage_seconds'].items():stage_values.setdefault(key,[]).append(value)
            reference=root/f'jepa_replay_v1_{variant}_s0_w0.3'/(seq+'.txt')
            actual_lines=pred.read_text().splitlines();reference_lines=reference.read_text().splitlines()
            if len(actual_lines)!=len(reference_lines):raise ValueError('Predictionrow count changed')
            differences=[]
            for a,b in zip(actual_lines,reference_lines):
                x,y=a.split(','),b.split(',')
                if x[:2]!=y[:2]:raise ValueError('Online/cache identitypair changed')
                if a!=b:differences.append(dict(frame=int(x[0]),track_id=int(x[1]),online=a,cached=b))
            if differences and (seq!='SN-013' or any(d['frame'] not in [2999,3000] for d in differences)):raise ValueError('Unexplained input/runtime difference')
            boundary[seq]=dict(cache_sha256=sha256(reference),online_sha256=sha256(pred),identity_pairs_equal=True,byte_equal=not differences,different_rows=differences)
        combined=dict(frames=len(all_times),pipeline_seconds=sum(all_times),fps=len(all_times)/sum(all_times),latency_p50_p95_ms=(np.quantile(all_times,[.5,.95])*1000).tolist(),stage_p50_p95_ms={k:(np.quantile(v,[.5,.95])*1000).tolist() for k,v in stage_values.items()})
        item=dict(manifest_sha256=sha256(folder/'manifest.json'),evaluation_sha256=sha256(evaluation),combined_timing=combined,metrics=official['metrics'],parameters=manifest['parameters'],videos=manifest['videos'],boundary=boundary,
          full_parent_cuda_peak_bytes=manifest.get('peak_parent_gpu_memory_bytes'),full_detector_cuda_peak_bytes=manifest['detector_runtime'].get('peak_gpu_memory_bytes'),completion_recovery=manifest.get('completion_recovery'))
        probe=root/'jepa_online_ordinary_transformer_seed0_shutdown_probe64_v2'
        if variant=='ordinary_transformer' and (probe/'manifest.json').exists():
            pm=json.loads((probe/'manifest.json').read_text())
            if pm['status']!='COMPLETE' or not pm['probe'] or any(v['frames']!=64 for v in pm['videos'].values()):raise ValueError('Incomplete memory/shutdown probe')
            for seq in ['BT-005','SN-013']:
                prefix='\n'.join(l for l in (folder/(seq+'.txt')).read_text().splitlines() if int(l.split(',')[0])<=64)+'\n'
                if (probe/(seq+'.txt')).read_text()!=prefix:raise ValueError('Shutdownrepair changesprefix inference')
            item['separate_probe_memory']=dict(frames_per_video=64,parent_cuda_peak_bytes=pm['peak_parent_gpu_memory_bytes'],detector_cuda_peak_bytes=pm['detector_runtime']['peak_gpu_memory_bytes'],manifest_sha256=sha256(probe/'manifest.json'),scope='Separate completed64frame probe aftershutdownfix. These are NOT measuredfullrun peaks; no FPS substitution.')
        record['models'][variant]=item
        q=official['metrics']['COMBINED_SEQ'];p50,p95=combined['latency_p50_p95_ms']
        lines.append(f"|{variant} seed0|{combined['frames']}|{combined['fps']:.4f}|{p50:.2f}|{p95:.2f}|"+'|'.join(f"{q[k]:.4f}" for k in ['HOTA','AssA','DetA','IDF1','MOTA','IDSW'])+'|')
    lines+=['','Completeper-video metric vectors, allstage latencies, prediction/timing/source/checkpoint hashes andterminal-frame rows areinthe measuredJSON. Every4834frame/track IDpair matches cachedProtocolA; fullBT005 byteequal, SN013differences confined to2999/3000 duehistoricalbatch2 vs current-onlypaddedbatch8. Do notcall allMOTbytes equal.','','The ordinaryTransformer worker acknowledgedstop afterallframes, but original10secondprocess teardown timedout. Outputs/timingwere closedandhashed; originalmanifest/log preserved, completedmetadata recoveredwithoutre-inference. Fullrun CUDApeak fields are unavailable. A separatelycompleted64frame-per-video probe measuresmemory andverifies shutdownrepair/prefix byteparity; its peaks areexplicitly notfullrun peaks. J3fullrun peaks remain directlymeasured. Theclosefix onlycleans upthisownedworker andpreserves acknowledgedstatistics; receive/processingerrors stillfail.','','Formal ProtocolA candidates useallthree seeds. No coreidentity method passedthe strongerordinarybaseline, so theseProtocolB runs are representativeimplementation/cost validation, notthree-seedmethodsuccess. Completeonlineseed1/2/newthirdroundcandidate runs areunperformed; noend-to-end three-seedmean claimed.']
    write_json(args.output,record);args.report.write_text('\n'.join(lines)+'\n')
    print({name:item['combined_timing']['fps'] for name,item in record['models'].items()},flush=True)


if __name__=='__main__':main()
