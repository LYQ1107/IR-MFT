#!/usr/bin/env python3
"""Registered repair of query overreach, using the same frozen fitted heads."""
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    folder=ROOT/'configs/perceptual_counterfactual_query_dev_20261123'
    if folder.exists():raise ValueError('Counterfactual query repair already registered')
    folder.mkdir();outputs={}
    for path in sorted((ROOT/'configs/perceptual_native_query_dev_20261123').glob('*.json')):
        cfg=json.loads(path.read_text());name='C'+cfg['id'][1:]
        cfg.update(id=name,backend='counterfactual_query',counterfactual_minimum_gain=.05,counterfactual_preserve_clear=True,
                   registration='Registered after native-query Dev overreach diagnosis and before this repair replay. Keep complete fitted RGB point matcher by default. Native cues may correct only when improving the same frozen RGB/null query by>=0.05. Protect a unique RGB body correspondence. Zero native evidence exactly preserves original fitted RGB tracking, including with prebirth consultation enabled. Same fitted heads and source data; no Test selection.')
        output=folder/(name+'.json');dump(output,cfg);outputs[name]=dict(path=str(output),sha256=sha256(output),source_configuration_sha256=sha256(path))
    dump(work/'counterfactual_query_dev_registration.json',dict(status='REGISTERED_BEFORE_COUNTERFACTUAL_QUERY_DEV_REPLAY',outputs=outputs,count=len(outputs),
        mechanism='Counterfactual risk correction, explicit no-correspondence, immutable retained identity consultation',source_sha256=sha256(__file__),Test_selection=False))
    print(json.dumps(dict(configurations=len(outputs))))


if __name__=='__main__':main()
