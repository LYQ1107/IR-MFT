#!/usr/bin/env python3
"""Summarize pinned primary-source evidence without inferring license grants."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    original = ROOT / 'docs/fusion_research/OPEN_SOURCE_20_CODE_AUDIT.json'
    rows = json.loads(original.read_text())
    missing = {r['id']: r for r in rows if not r['license']}
    assert len(rows) == 20 and set(missing) == {'R03', 'R04', 'R06', 'R08', 'R09', 'R13'}
    base = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122/source_license_recheck_20261124')
    trees = json.loads((base / 'trees_manifest.json').read_text())
    texts = json.loads((base / 'license_text_manifest.json').read_text())
    assert len(trees) == 6 and len(texts) == 20
    output = []
    for tree in trees:
        reference = missing[tree['id']]
        assert tree['status'] == 200 and tree['truncated'] is False
        assert tree['commit'] == reference['commit']
        assert sha256(tree['tree_file']) == tree['tree_sha256']
        actual_tree = json.loads(Path(tree['tree_file']).read_text())
        paths = [entry['path'] for entry in actual_tree['tree'] if entry['type'] == 'blob']
        named = [p for p in paths if any(token in Path(p).name.lower() for token in ['license', 'licence', 'copying', 'copyright'])]
        assert named == tree['license_candidates'] == []
        evidence = []
        for item in texts:
            if item['id'] != tree['id']: continue
            assert item['status'] == 200 and sha256(item['cached_file']) == item['sha256']
            if item['role'] == 'original_inspected_source': assert item['matches_original_source_sha256']
            evidence.append({k: v for k, v in item.items() if k not in ['cached_file', 'id', 'expected_sha256']})
        assert len([e for e in evidence if e['role'] == 'root_readme']) == len(tree['root_readmes'])
        assert len([e for e in evidence if e['role'] == 'original_inspected_source']) == len(reference['files'])
        output.append(dict(id=tree['id'], name=reference['name'], repo=reference['url'], commit=reference['commit'],
            recursive_tree_url=tree['url'], recursive_tree_sha256=tree['tree_sha256'],
            recursive_tree_truncated=False, tracked_blob_count=len(paths), standard_named_license_paths=named,
            root_license_grant='UNCONFIRMED_FROM_INSPECTED_PINNED_SOURCES',
            checked_files=evidence, inherited_notice_found=any(e['license_mentions'] for e in evidence),
            primary_fetch_UTC=tree['UTC']))
    assert sum(len(r['checked_files']) for r in output) == 20
    stem = ROOT / 'docs/fusion_research/OPEN_SOURCE_20_LICENSE_EVIDENCE_20261124'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    report = dict(status='COMPLETE_PINNED_PRIMARY_SOURCE_LICENSE_EVIDENCE_RECHECK_FOR_SIX_UNRECORDED_ENTRIES',
        original20_source_audit_sha256=sha256(original), source_sha256=sha256(__file__), rows=output,
        trees_manifest_sha256=sha256(base / 'trees_manifest.json'), text_manifest_sha256=sha256(base / 'license_text_manifest.json'),
        primary_recursive_trees=6, primary_readmes=6, rechecked_original_source_files=14,
        all_rechecked_original_file_hashes_match=True, all_tree_responses_HTTP200_and_not_truncated=True,
        all_standard_named_license_files_absent_in_returned_trees=True,
        unconfirmed_root_license_grants=6, repository_wide_license_clearance_claim=False,
        no_third_party_source_copied=True, new_GPU_or_model_or_dataset_work=False,
        scope='Evidence at the same6commits, English standard license filenames across the complete returned tree, root READMEs and previously inspected code. '
            'No claim that other files lack notices, that projects are unlicensed, or that a dependency/file notice determines every original project file.',
        scientific_innovation_or_accuracy_evidence=False, goal_met=False, supported_innovations=0)
    dump(stem.with_suffix('.json'), report)
    lines = ['# 20项原源码审计：六项许可字段的证据补充', '',
        '本次保持原20项审计和原固定提交不变，补查其中未记录许可的六项。官方GitHub API六次完整递归树响应均为HTTP200且未截断；六份根README和十四份原被查源码也均返回HTTP200，十四份文件SHA256全部与原审计一致。没有安装仓库、执行第三方代码或新增GPU/模型/数据工作。', '',
        '| ID | 固定仓库/提交 | 全树blob数 | 标准命名许可文件 | 根项目许可范围 |',
        '| --- | --- | ---: | --- | --- |']
    for row in output:
        url = row['repo']+'/tree/'+row['commit']
        lines.append(f"| {row['id']} | [{row['name']} @ {row['commit'][:12]}]({url}) | {row['tracked_blob_count']} | 未检出 | 未确认 |")
    lif = next(r for r in output if r['id'] == 'R09')
    notice = next(e for e in lif['checked_files'] if e['license_mentions'])
    lines += ['',
        '“未检出”精确指返回完整树中，文件名包含license/licence/copying/copyright的文件未检出；不是对所有文件内容的穷尽法律判断，也不等于宣布仓库没有任何许可。被查README未检出相应许可关键词，仍不补写不存在的项目授权。原审计中的六个空字段现在有具体的检索范围、响应、时间和哈希来源，根项目许可范围继续标为未确认。', '',
        f"R09的[`ultralytics/nn/modules/block.py`]({notice['url']})第1行直接给出AGPL-3.0声明。这是所查文件头的直接证据；原报告将其概称为整个仓库AGPL过宽，本补充收紧为文件级事实，不从这一处声明推定所有自定义项目文件的许可范围。没有复制该第三方实现。", '',
        '另五项被查源码未检出许可关键词，不据此推断未查文件或继承依赖的许可。完整树API链接、原文件URL、行号、字节数、响应时间和SHA256在JSON中保留，原20项源码审计JSON不被重写。许可证据补查不算科学创新或准确率进展，也不使尚未达到的MFT目标完成。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], primary_trees=6, source_hashes_matched=14,
                         root_grants_unconfirmed=6, copied_source=False)), flush=True)


if __name__ == '__main__': main()
