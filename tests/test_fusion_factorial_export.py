import csv
import json
from pathlib import Path

import pytest

from tools import export_fusion_factorial as export
from tools.run_pseudo_ir_comparison import sha256

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/fusion_research/perceptual_20261122_official_tables/strong_rgb24_dev/F_rgb_rgb_s0_e24.csv'


def evidence():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    schema,_=export.official_schema(plan['evaluation']['TrackEval_root'])
    rows,provenance=export.read_table(SOURCE,schema,protocol)
    with SOURCE.open() as stream:raw=list(csv.DictReader(stream))
    return schema,protocol,rows,raw


def test_all47_fields_and_unrounded_combined_come_from_original_official_csv():
    before=sha256(SOURCE);schema,protocol,rows,raw=evidence()
    assert len(schema)==47 and len(rows)==3
    assert {r['seq'] for r in rows}==set(protocol['dev'])|{'COMBINED'}
    combined=next(r for r in rows if r['seq']=='COMBINED')
    original=next(r for r in raw if r['seq']=='COMBINED')
    assert combined['HOTA']==100*float(original['HOTA___AUC']) and combined['CLR_Frames']==4834
    assert sha256(SOURCE)==before


def test_count_means_and_raw_units_are_not_mislabeled_as_percentages():
    schema,protocol,rows,raw=evidence()
    combined=next(r for r in rows if r['seq']=='COMBINED');original=next(r for r in raw if r['seq']=='COMBINED')
    assert combined['HOTA_TP']==float(original['HOTA_TP___AUC'])
    assert combined['MOTP_sum']==float(original['MOTP_sum'])
    assert combined['FP_per_frame']==float(original['FP_per_frame'])
    assert next(r for r in schema if r['metric']=='HOTA_TP')['unit']=='count_mean_over_19_HOTA_thresholds'


def test_derived_view_roundtrip_preserves_every_value_and_existing_file(tmp_path):
    schema,protocol,rows,raw=evidence();path=tmp_path/'view47.csv'
    export.write47(path,rows,schema);before=sha256(path)
    export.write47(path,rows,schema);assert sha256(path)==before
    with path.open() as stream:actual=list(csv.DictReader(stream))
    assert len(actual[0])==50
    for left,right in zip(actual,rows):
        assert all(float(left[s['metric']])==right[s['metric']] for s in schema)


@pytest.mark.parametrize('damage',['missing_video','wrong_count'])
def test_incomplete_coverage_or_broken_official_accounting_cannot_export_as_complete(tmp_path,damage):
    schema,protocol,rows,raw=evidence()
    with SOURCE.open() as stream:fields=csv.DictReader(stream).fieldnames
    if damage=='missing_video':raw=[r for r in raw if r['seq']!=protocol['dev'][0]]
    else:next(r for r in raw if r['seq']=='COMBINED')['GT_Dets']='1'
    path=tmp_path/'invalid.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(raw)
    with pytest.raises(AssertionError):export.read_table(path,schema,protocol)


def test_public_report_guard_rejects_large_objects_before_writing(tmp_path):
    path=tmp_path/'too_large.json'
    with pytest.raises(AssertionError):export.small_json(path,dict(text='x'*(2**20)))
    assert not path.exists()
