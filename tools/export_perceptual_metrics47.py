#!/usr/bin/env python3
"""Expose all47 official metric names from unmodified276-column published CSVs.

No evaluation, fitting, new metric definition, sequence averaging or selection.
The8fields absent from the39headline values already exist in detailed output.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def official_schema(trackeval_root):
    sys.path.insert(0, str(trackeval_root))
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.clear import CLEAR
    from trackeval.metrics.identity import Identity
    from trackeval.metrics.count import Count
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = [HOTA(), CLEAR(), Identity(), Count()]
    schema, headline, class_counts = [], [], {}
    for metric in metrics:
        class_counts[metric.get_name()] = len(metric.fields)
        headline.extend(metric.summary_fields)
        for field in metric.fields:
            array = field in metric.float_array_fields + metric.integer_array_fields
            if field in metric.integer_array_fields:
                unit, scale = 'count_mean_over_19_HOTA_thresholds', 1
            elif field in metric.integer_fields:
                unit, scale = 'count', 1
            elif field == 'FP_per_frame':
                unit, scale = 'false_positives_per_frame', 1
            elif field == 'MOTP_sum':
                unit, scale = 'sum_of_matched_IoU', 1
            else:
                unit, scale = 'percent', 100
            schema.append(dict(metric=field, family=metric.get_name(),
                source_column=field+'___AUC' if array else field,
                unit=unit, scale_from_raw_detailed=scale, in_official_headline=field in metric.summary_fields))
    assert len(schema) == len({s['metric'] for s in schema}) == 47 and len(headline) == 39
    assert class_counts == {'HOTA': 15, 'CLEAR': 22, 'Identity': 6, 'Count': 4}
    return schema, class_counts


def read_table(path, schema, protocol):
    encoded = path.read_bytes()
    reader = csv.DictReader(io.StringIO(encoded.decode()))
    fields, rows = reader.fieldnames, list(reader)
    assert fields is not None and len(fields) == 276
    expected = {s['source_column'] for s in schema}
    aggregate = {k for k in fields if k != 'seq' and ('___' not in k or k.endswith('___AUC'))}
    assert aggregate == expected and len(aggregate) == 47
    labels = {r['seq'] for r in rows}
    split = next((split for split, names in [('dev', protocol['dev']), ('test', protocol['historical_test'])]
                  if labels == set(names) | {'COMBINED'}), None)
    assert split is not None and len(rows) == len(labels)
    converted = []
    for row in rows:
        values = {}
        for item in schema:
            value = float(row[item['source_column']])
            assert math.isfinite(value)
            if item['unit'] == 'count':
                assert value.is_integer()
                values[item['metric']] = int(value)
            else:
                values[item['metric']] = value * item['scale_from_raw_detailed']
        # These checks catch swapped fields and accidental percentage conversion.
        assert values['CLR_TP']+values['CLR_FN'] == values['GT_Dets']
        assert values['IDTP']+values['IDFN'] == values['GT_Dets']
        assert values['IDTP']+values['IDFP'] == values['Dets']
        assert math.isclose(values['HOTA_TP']+values['HOTA_FN'], values['GT_Dets'], rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(values['HOTA_TP']+values['HOTA_FP'], values['Dets'], rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(values['FP_per_frame'], values['CLR_FP']/max(1, values['CLR_Frames']), rel_tol=1e-12, abs_tol=1e-12)
        converted.append(dict(candidate=path.stem, split=split, seq=row['seq'], **values))
    combined = next(r for r in converted if r['seq'] == 'COMBINED')
    assert combined['CLR_Frames'] == {'dev': 4834, 'test': 15985}[split]
    return converted, dict(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(encoded).hexdigest(),
                          split=split, rows=len(rows), source_columns=276)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    source = ROOT / 'docs/fusion_research/perceptual_20261122_official_tables'
    destination = ROOT / 'docs/fusion_research/perceptual_20261122_metrics47'
    groups, sources, raw_rows = {}, [], 0
    for path in sorted(source.rglob('*.csv')):
        stage = str(path.parent.relative_to(source))
        assert stage != '.' and '..' not in path.relative_to(source).parts
        rows, provenance = read_table(path, schema, protocol)
        groups.setdefault(stage, []).extend(rows)
        sources.append(dict(stage=stage, **provenance))
        raw_rows += len(rows)
    assert sources
    existing = destination / 'index.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        current = {s['source']: s['sha256'] for s in sources}
        assert all(current.get(s['source']) == s['sha256'] for s in previous['sources']), 'Preserve source tables and previous results'
    outputs, total_bytes = [], 0
    for stage, rows in groups.items():
        path = destination / (stage.replace('/', '__')+'.csv')
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=['candidate', 'split', 'seq']+[s['metric'] for s in schema])
        writer.writeheader(); writer.writerows(rows)
        encoded = buffer.getvalue().encode()
        assert len(encoded) < 5*2**20
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != encoded:
                temporary = path.with_suffix('.csv.tmp'); temporary.write_bytes(encoded); temporary.replace(path)
        total_bytes += len(encoded)
        outputs.append(dict(stage=stage, file=str(path.relative_to(ROOT)), rows=len(rows), bytes=len(encoded),
                            sha256=hashlib.sha256(encoded).hexdigest()))
    report = dict(status='COMPLETE_METRICS47_VIEW_OF_CURRENT_PUBLISHED_OFFICIAL_TABLES',
        official_metric_names=47, official_headline_names=39, fields_per_original_detailed_CSV=276,
        metric_class_counts=counts, schema=schema, additional_to_headline=[s['metric'] for s in schema if not s['in_official_headline']],
        source_tables=len(sources), metric_rows=raw_rows, output_stage_tables=len(outputs),
        output_bytes=total_bytes, sources=sources, outputs=outputs,
        TrackEval_source_hashes={name: sha256(Path(plan['evaluation']['TrackEval_root']) / 'trackeval/metrics' / name)
            for name in ['hota.py', 'clear.py', 'identity.py', 'count.py', '_base_metric.py']},
        source_sha256=sha256(__file__), main_plan_sha256=sha256(ROOT / 'configs/perceptual_mft_20261122.json'),
        no_new_fit_or_inference_or_Test_selection=True, no_sequence_reaggregation=True,
        scope='Derived view of already published complete official CSVs only; not proof that every formal candidate exists or has been evaluated.',
        units='Ratios displayed as percent. HOTA_TP/FN/FP are unrounded means across19thresholds, not single-threshold integer counts. '
            'FP_per_frame and MOTP_sum retain raw units. COMBINED is copied from official combination, never mean of sequence scores.',
        goal_met=False, supported_innovations=0)
    if not args.dry_run: dump(existing, report)
    print(json.dumps(dict(status='DRY_RUN' if args.dry_run else report['status'], source_tables=len(sources),
        metric_rows=raw_rows, stages=len(outputs), bytes=total_bytes, fields=47,
        extras=report['additional_to_headline'])), flush=True)


if __name__ == '__main__': main()
