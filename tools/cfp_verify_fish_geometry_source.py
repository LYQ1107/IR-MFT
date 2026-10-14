#!/usr/bin/env python3
"""Licensed pure-function parity; never import/execute a foreign tracker."""
import argparse,ast,json,math,subprocess
from pathlib import Path
import numpy as np
from cfp.fish_geometry import geometry_similarity
from cfp.provenance import sha256,snapshot_sources,write_json


def function(path,name):
    tree=ast.parse(path.read_text());nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(nodes)!=1:raise ValueError('Exact unique sourcefunction required')
    isolated=ast.Module(body=nodes,type_ignores=[]);env={'np':np,'math':math};exec(compile(ast.fix_missing_locations(isolated),str(path),'exec'),env)
    return env[name]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--references',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('Newimmutable exactsource proof required')
    roots={'SU-T':a.references/'SU-T','TIDE':a.references/'TIDE'};paths={'SU-T':'trackers/association.py','TIDE':'src/lib/tracker/matching.py'};proof={}
    for name,root in roots.items():
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();path=root/paths[name];blob=subprocess.check_output(['git','-C',str(root),'show',commit+':'+paths[name]]);assert blob==path.read_bytes();assert sha256(root/'LICENSE')==sha256(Path('third_party/notices')/(name+'_LICENSE.txt'));proof[name]=dict(commit=commit,source_sha256=sha256(path),license_sha256=sha256(root/'LICENSE'))
    rng=np.random.default_rng(12);xy=rng.uniform(-30,500,(50,2));wh=rng.uniform(5,120,(50,2));boxes=np.c_[xy,xy+wh];xy2=rng.uniform(-30,500,(40,2));wh2=rng.uniform(5,120,(40,2));other=np.c_[xy2,xy2+wh2]
    fish=function(roots['SU-T']/paths['SU-T'],'fish_iou_batch');expected=fish(boxes,other);actual=geometry_similarity(boxes,other,'fish_source');error=float(np.max(np.abs(expected-actual)));assert error<3e-14
    agc=function(roots['TIDE']/paths['TIDE'],'AGCv1');expected=np.array([[np.clip(agc(x,y,1e-6),0,1) for y in other] for x in boxes]);actual=geometry_similarity(boxes,other,'agc_clipped');agc_error=float(np.max(np.abs(expected-actual)));assert agc_error<3e-14
    singular=False
    try:agc(boxes[0],boxes[0],1e-6)
    except ZeroDivisionError:singular=True
    assert singular
    write_json(a.output,dict(status='EXACT_LICENSED_GEOMETRY_FUNCTION_PARITY_VERIFIED',source=proof,pairs=2000,maximum_fish_source_error=error,maximum_finite_clipped_AGCv1_error=agc_error,source_AGCv1_zero_center_distance_raises=True,controlled_AGC_zero_distance_clipped_limit=1.,sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/fish_geometry.py','third_party/notices/SU-T_LICENSE.txt'][:-1]),scope='OnlyexactMIT pureNumPy/math functions isolatedbyAST, neverfullforeigntracker import/execution. SourceFishcentralbug andarea1000factor preserved, centralrepair separate. AGCv1inactive insource andsingular; finiteclipping disclosed, notpaperAGCIoU/fulltracker claim. No dataset/GT/model inputs used.'))
    print(json.dumps(dict(fish_error=error,agc_error=agc_error,singular=singular)))

if __name__=='__main__':main()
