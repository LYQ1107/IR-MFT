#!/usr/bin/env python3
"""Fetch verified public text blobs at pinned commits after Git TLS failures.

Reference-only acquisition, no model/data download or external code execution.
Each fetched blob must match its Git tree SHA before it is materialized. The
bounded selection is for audit entry points, not a complete repository clone.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler, Request

from cfp.provenance import sha256, write_json


def public_bytes(url, limit):
    opener=build_opener(ProxyHandler({'https':'http://127.0.0.1:7890'}))
    errors=[]
    for _ in range(3):
        try:
            with opener.open(Request(url,headers={'User-Agent':'IR-MFT-source-audit','Cache-Control':'no-cache'}),timeout=15) as stream:
                data=stream.read(limit+1)
            if len(data)>limit:raise ValueError('Public text blob exceeds acquisition budget')
            return data
        except Exception as error:errors.append(repr(error))
    raise RuntimeError('; '.join(errors))


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True,timeout=30).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-files',type=int,default=24)
    args=parser.parse_args()
    if args.output.exists() or not 4<=args.maximum_files<=32:raise ValueError('New bounded audit acquisition required')
    plan=json.loads(args.plan.read_text());root=Path(plan['references_root'])
    report=dict(status='RUNNING',semantic_audit_status='IN_PROGRESS',plan_sha256=sha256(args.plan),
                source_sha256=sha256(__file__),repositories={},scope='Pinned public source entry points only. Not full semantic audit or model execution.')
    write_json(args.output,report)

    def job(spec):
        name,slug,relative=spec;path=root/relative
        entry=dict(name=name,github='https://github.com/'+slug,files={},errors={},semantic_audit='NOT_COMPLETE')
        try:
            if (path/'.git').exists():
                commit=git(path,'rev-parse','HEAD');tree=[]
                for line in git(path,'ls-tree','-r','HEAD').splitlines():
                    header,file=line.split('\t',1);mode,kind,digest=header.split()
                    if kind=='blob' and mode!='120000':tree.append((file,digest))
            else:
                metadata=json.loads(public_bytes('https://api.github.com/repos/'+slug+'/commits?per_page=1',2**20))
                commit=metadata[0]['sha']
                response=json.loads(public_bytes(f'https://api.github.com/repos/{slug}/git/trees/{commit}?recursive=1',12*2**20))
                if response.get('truncated'):raise ValueError('Truncated source tree cannot support accurate entry selection')
                tree=[(v['path'],v['sha']) for v in response['tree'] if v['type']=='blob' and v['mode']!='120000']
                path=root/'phase7'/('http_'+name)
            entry.update(commit=commit,local_path=str(path))
            if not relative.startswith('phase7/'):
                entry['status']='REUSE_HISTORICAL_NO_MUTATION'
                return name,entry
            def priority(item):
                file=item[0];base=Path(file).name.lower();parts=Path(file).parts
                if len(parts)==1 and base.startswith(('license','readme','copying')):return 0
                if base in ('train.py','main.py','requirements.txt','hubconf.py','predictor.py','id_decoder.py','id_criterion.py','train_mae.py','loss.py','sampler.py','model.py'):return 1
                return 2
            selected=[v for v in tree if Path(v[0]).suffix.lower() in {'.py','.md','.txt','.yaml','.yml'} or Path(v[0]).name.lower().startswith(('license','copying'))]
            selected=[v for v in selected if not any(p.lower() in {'checkpoints','weights','pretrained','assets','trackeval','third_party','thirdparty'} for p in Path(v[0]).parts)]
            selected=sorted(selected,key=lambda v:(priority(v),len(Path(v[0]).parts),v[0]))[:args.maximum_files]
            for file,digest in selected:
                try:
                    if shutil.disk_usage(root).free<40*2**30:raise OSError('40GiB reserve')
                    target=(path/file).resolve()
                    if not target.is_relative_to(path.resolve()):raise ValueError('Reference source path escapes exact snapshot')
                    if target.exists():
                        data=target.read_bytes()
                        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                        if actual!=digest:raise ValueError('Existing text differs from exact tree; preserve it')
                        how='VERIFIED_EXISTING'
                    else:
                        data=public_bytes(f'https://raw.githubusercontent.com/{slug}/{commit}/{quote(file,safe="/")}',5*2**20)
                        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                        if actual!=digest:raise ValueError('Downloaded public text blob SHA differs from pinned Git tree')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with target.open('xb') as stream:stream.write(data)
                        how='PUBLIC_PINNED_RAW_BLOB'
                    entry['files'][file]=dict(git_blob_sha=digest,sha256=sha256(target),bytes=len(data),acquisition=how)
                except Exception as error:entry['errors'][file]=repr(error)
            entry['status']='SELECTED_SOURCE_COMPLETE' if not entry['errors'] else 'SELECTED_SOURCE_PARTIAL'
        except Exception as error:entry.update(status='ACQUISITION_FAILED',error=repr(error))
        return name,entry

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,spec) for spec in plan['repositories']]):
            name,entry=future.result();report['repositories'][name]=entry;write_json(args.output,report)
            print(name,entry['status'],len(entry['files']),len(entry['errors']),flush=True)
    report['status']='ACQUISITION_COMPLETE_WITH_EXPLICIT_GAPS';write_json(args.output,report)


if __name__=='__main__':main()
