#!/usr/bin/env python3
"""Check V3 evaluation/NMS independence from altered future batch entries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from cfp.provenance import sha256, write_json
from yolox.exp import get_exp
from yolox.data import RGBTValTransform
from yolox.utils import postprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    generation = json.loads((args.data_root / 'manifest.json').read_text())
    if generation['status'] != 'COMPLETE':
        raise ValueError('Use complete or complete diagnostic causal inputs')
    protocol = json.loads(args.protocol.read_text())
    exp = get_exp('exps/yolox_mft26ir_rgbt_v3.py', None)
    exp.data_dir = str(args.data_root)
    exp.input_size = exp.test_size
    transform = RGBTValTransform(rgb_means=exp.rgb_means, std=exp.rgb_std, ir_means=exp.ir_means, ir_std=exp.ir_std)
    dataset = exp._dataset('train', 'train.json', transform)
    positions = [i for i, ann in enumerate(dataset.annotations) if ann[2].split('/')[0] == protocol['train'][0]][:8]
    if len(positions) != 8:
        raise ValueError('At least eight consecutive training frames needed')
    batch = np.stack([dataset[i][0] for i in positions])
    tensor = torch.from_numpy(batch).cuda().half()
    altered = tensor.clone()
    altered[4:] = 0
    model = exp.get_model().cuda().half().eval()
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    del state
    with torch.inference_mode():
        raw = model(tensor)
        raw_altered = model(altered)
        raw_prefix_exact = bool(torch.equal(raw[:4], raw_altered[:4]))
        suffix_changed = bool(not torch.equal(raw[4:], raw_altered[4:]))
        outputs = postprocess(raw, 1, .1, .7)
        outputs_altered = postprocess(raw_altered, 1, .1, .7)
    equality = []
    for original, changed in zip(outputs[:4], outputs_altered[:4]):
        equality.append(original is None and changed is None or original is not None and changed is not None and torch.equal(original, changed))
    report = dict(raw_prefix_exact=raw_prefix_exact, post_nms_prefix_exact=[bool(x) for x in equality],
                  altered_suffix_raw_outputs_changed=suffix_changed, checkpoint_sha256=sha256(args.checkpoint),
                  source_sha256=sha256(__file__), protocol_sha256=sha256(args.protocol),
                  generation_manifest_sha256=sha256(args.data_root / 'manifest.json'),
                  input_batch_sha256=hashlib.sha256(batch.tobytes()).hexdigest(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                  batch_size=8, preserved_prefix=4, fp16=True,
                  scope='Detector/NMS component batch-independence check; complements causal generation prefix tests; batch timing is not online latency')
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not raw_prefix_exact or not all(equality) or not suffix_changed:
        raise AssertionError('Batch future-dependency check failed')


if __name__ == '__main__':
    main()
