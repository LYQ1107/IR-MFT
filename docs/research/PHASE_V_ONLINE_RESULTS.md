# Fish-JEPA complete causal online evaluation

Full4834frame strictcausal ProtocolB, representative seed0J3/ordinaryTransformer. ProtocolA three-seed comparison independent. Timing includes currentdecode/causalIR/V3/NMS/crops/RGBencoding/student+association/output; shared GPU/CPUload, notdedicated deployment. Startup model loading excluded. CUDAallocated peaks perprocess, notwholeGPU NVMLmemory.

|Model|Frames|FPS|p50 ms|p95 ms|HOTA|AssA|DetA|IDF1|MOTA|IDSW|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|J3 seed0|4834|0.7933|1268.02|1646.83|52.6318|37.7247|73.4850|60.1641|93.7173|80.0000|
|ordinary_transformer seed0|4834|0.7962|1235.38|1746.05|55.4745|41.8173|73.6349|64.0967|93.7109|79.0000|

Completeper-video metric vectors, allstage latencies, prediction/timing/source/checkpoint hashes andterminal-frame rows areinthe measuredJSON. Every4834frame/track IDpair matches cachedProtocolA; fullBT005 byteequal, SN013differences confined to2999/3000 duehistoricalbatch2 vs current-onlypaddedbatch8. Do notcall allMOTbytes equal.

The ordinaryTransformer worker acknowledgedstop afterallframes, but original10secondprocess teardown timedout. Outputs/timingwere closedandhashed; originalmanifest/log preserved, completedmetadata recoveredwithoutre-inference. Fullrun CUDApeak fields are unavailable. A separatelycompleted64frame-per-video probe measuresmemory andverifies shutdownrepair/prefix byteparity; its peaks areexplicitly notfullrun peaks. J3fullrun peaks remain directlymeasured. Theclosefix onlycleans upthisownedworker andpreserves acknowledgedstatistics; receive/processingerrors stillfail.

Formal ProtocolA candidates useallthree seeds. No coreidentity method passedthe strongerordinarybaseline, so theseProtocolB runs are representativeimplementation/cost validation, notthree-seedmethodsuccess. Completeonlineseed1/2/newthirdroundcandidate runs areunperformed; noend-to-end three-seedmean claimed.
