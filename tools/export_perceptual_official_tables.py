#!/usr/bin/env python3
"""Publish small unmodified official metric tables, never arrays or tracks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    destination=ROOT/'docs/fusion_research/perceptual_20261122_official_tables';destination.mkdir(exist_ok=True)
    rows=[]
    for path in sorted(work.glob('*/*/metrics.json')):
        metrics=json.loads(path.read_text())
        if metrics.get('status')!='COMPLETE' or 'manifest' not in metrics:continue
        manifest=metrics['manifest'];split=manifest['split'];expected=protocol['dev' if split=='dev' else 'historical_test']
        if split not in ['dev','test'] or manifest['sequences']!=expected or manifest['frames']!={'dev':4834,'test':15985}[split]:raise ValueError('Incomplete protocol coverage')
        source=path.parent/'trackeval_output/pedestrian_detailed.csv'
        with source.open(newline='') as handle:table=list(csv.DictReader(handle))
        if len(table)!=len(expected)+1 or not set(expected).issubset(row['seq'] for row in table):raise ValueError('Missing official detailed rows')
        stage,name=path.parent.parent.name,path.parent.name
        if not all(re.fullmatch(r'[A-Za-z0-9_-]+',value) for value in [stage,name]):raise ValueError('Unsafe table name')
        output=destination/stage/(name+'.csv');output.parent.mkdir(exist_ok=True)
        if output.exists():
            if digest(output)!=digest(source):raise ValueError('Published official table changed')
        else:shutil.copy2(source,output)
        rows.append(dict(stage=stage,candidate=name,split=split,frames=manifest['frames'],sequences=expected,columns=len(table[0]),bytes=output.stat().st_size,sha256=digest(source),source=str(source),file=str(output.relative_to(ROOT)),configuration_sha256=manifest['config_sha256'],source_metrics_sha256=digest(path)))
    report=dict(status='COMPLETE_FOR_CURRENT_FINISHED_RUNS',scope='Unmodified TrackEval detailed CSV, all per-sequence and combined rows and every threshold-specific column. No model, feature bank, crop bank, image dataset or raw track prediction is published. Test metric values are archived without selecting later candidate parameters from them.',tables=len(rows),bytes=sum(x['bytes'] for x in rows),rows=rows)
    path=destination/'index.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');os.replace(temporary,path)
    print(json.dumps(dict(tables=report['tables'],bytes=report['bytes'],Test_tables=sum(x['split']=='test' for x in rows),accuracy_values_printed=False)))


if __name__=='__main__':main()
