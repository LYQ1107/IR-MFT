#!/usr/bin/env python3
"""Register or run the complete final F4 queue, after all original predecessors."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fusion_memory_pipeline import register,run
from tools.fusion_memory_parent_io import require
from tools.fusion_memory_workflow import FusionMemoryWorkflow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','register','run'])
    parser.add_argument('--CPU-proof',type=Path)
    args=parser.parse_args();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller/registration/preflight must be CPU only')
    if args.action=='preflight':print(json.dumps(FusionMemoryWorkflow().preflight(),indent=2),flush=True)
    elif args.action=='register':
        require(args.CPU_proof is not None,'Complete current CPU integration proof required')
        register(args.CPU_proof)
    else:run()


if __name__=='__main__':main()
