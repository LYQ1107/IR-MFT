#!/usr/bin/env python3
"""Build a compact all-sequence/all-TrackEval-field historical audit report."""
import argparse
import csv
import json
from pathlib import Path


CORE = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'CLR_FP', 'CLR_FN', 'Frag']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='Phase-eight audit root containing eval_* directories')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-tsv', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    args = parser.parse_args()

    methods = {
        'historical_official': args.root / 'eval_historical_official' / 'metrics.json',
        'bytetrack_v3_exact': args.root / 'eval_bytetrack_exact' / 'metrics.json',
        'bot_sort_c0_rgb_seed0': args.root / 'eval_bot_sort_c0' / 'metrics.json',
        'hybrid_sort_c2_rgb_seed0': args.root / 'eval_hybrid_sort_c2' / 'metrics.json',
    }
    loaded = {name: json.loads(path.read_text()) for name, path in methods.items()}
    sequences = loaded['historical_official']['sequences']
    fields = list(loaded['historical_official']['metrics']['COMBINED_SEQ'].keys())
    protocol = {
        'protocol': 'configs/cfp/protocol_v2.json',
        'protocol_sha256': loaded['historical_official']['artifacts'][sequences[0]]['gt_sha256'],
        'partition': 'historical_test',
        'sequences': sequences,
        'frames': int(loaded['historical_official']['metrics']['COMBINED_SEQ']['CLR_Frames']),
        'gt_dets': int(loaded['historical_official']['metrics']['COMBINED_SEQ']['GT_Dets']),
        'trackeval_commit': loaded['historical_official']['trackeval_commit'],
        'metric_fields': fields,
    }
    # Keep protocol hash explicit and independent from the first GT file hash.
    protocol_path = Path('configs/cfp/protocol_v2.json')
    if protocol_path.is_file():
        import hashlib
        protocol['protocol_sha256'] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    report = {
        'schema_version': 1,
        'title': 'Historical-test public-method audit',
        'protocol': protocol,
        'methods': {},
        'deltas_vs_historical_official': {},
        'deltas_vs_bytetrack_v3_exact': {},
        'scope': {
            'proposal_cache': 'Frozen V3 reproduction proposals; shared byte-for-byte by all new public-method rows',
            'appearance': 'RGB seed 0 post-neck 512-D fish ReID features; BoT-SORT/HybridSORT use an explicit precomputed-feature adapter',
            'selection': 'BoT-SORT c0 and HybridSORT c2 were selected on Protocol-A dev only; no historical-test selection',
            'association_solver': 'External replay used a scipy linear_sum_assignment compatibility shim for the unavailable compiled lapjv package; the pinned tracker source was unchanged',
            'historical_baseline': 'historical_official is the previously frozen historical replay and is re-evaluated here under protocol_v2 derived GT',
            'limitation': 'The historical_test split was tuned previously, so this is a frozen audit rather than an independent blind benchmark',
        },
    }
    for name, payload in loaded.items():
        report['methods'][name] = {
            'metrics': payload['metrics'],
            'prediction_artifacts': payload['artifacts'],
        }
    base = loaded['historical_official']['metrics']['COMBINED_SEQ']
    byte = loaded['bytetrack_v3_exact']['metrics']['COMBINED_SEQ']

    def delta(a, b):
        out = {}
        for field in fields:
            av, bv = a[field], b[field]
            if isinstance(av, list) or isinstance(bv, list):
                out[field] = [x - y for x, y in zip(av, bv)]
            elif isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                out[field] = av - bv
            else:
                out[field] = None
        return out

    for name, payload in loaded.items():
        if name != 'historical_official':
            report['deltas_vs_historical_official'][name] = delta(payload['metrics']['COMBINED_SEQ'], base)
        if name != 'bytetrack_v3_exact':
            report['deltas_vs_bytetrack_v3_exact'][name] = delta(payload['metrics']['COMBINED_SEQ'], byte)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    rows = []
    for name, payload in loaded.items():
        for seq in sequences + ['COMBINED_SEQ']:
            row = {'method': name, 'sequence': seq}
            row.update({field: payload['metrics'][seq][field] for field in fields})
            rows.append(row)
    with args.output_tsv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'sequence'] + fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, list) else value
                             for key, value in row.items()})

    lines = [
        '# Historical-test public-method audit',
        '',
        f"评估集包含 `{len(sequences)}` 条序列、`{protocol['frames']}` 帧；使用 TrackEval `{protocol['trackeval_commit']}` 和 protocol_v2 派生 GT。完整 47 个字段及逐序列结果在 JSON/TSV 中。",
        '',
        '| 方法 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | CLR_FP | CLR_FN | Frag |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, payload in loaded.items():
        m = payload['metrics']['COMBINED_SEQ']
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} |'.format(
            name, m['HOTA'], m['DetA'], m['AssA'], m['IDF1'], m['MOTA'],
            int(m['IDSW']), int(m['CLR_FP']), int(m['CLR_FN']), int(m['Frag'])))
    lines += [
        '',
        '方法范围：原始 BYTETracker、冻结历史轨迹、BoT-SORT c0、HybridSORT c2。两个外部关联器共用同一 V3 proposal 和同一 RGB 鱼类特征缓存；没有声称复现其原生 person-ReID 权重。',
        '',
        '历史测试划分过去曾参与调参，因此表格用于跨序列冻结审计，不作为新的盲测结论。',
    ]
    args.output_md.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'methods': list(methods), 'sequences': sequences, 'fields': len(fields),
                      'json_bytes': args.output_json.stat().st_size,
                      'tsv_bytes': args.output_tsv.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
