#!/usr/bin/env python3
"""Publish every registered Train cue contrast; never choose a tracking model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.diagnose_structural_train_mechanism import read,sha
from tools.neck72_runtime_study import exited


def main():
    spec_path=ROOT/'configs/perceptual_train_cue_specificity_20261125.json'
    spec=read(spec_path);output=Path(spec['output_directory']);work=output.parent
    value=read(output/'complete.json');execution_path=work/'structural_cue_specificity_20261125.execution.json'
    execution=read(execution_path);identity_path=work/'structural_cue_specificity_20261125.identity_verified.json'
    identity=read(identity_path)
    assert value['status']=='COMPLETE_TRAIN_CURRENT_PAST_CUE_SPECIFICITY_NOT_TRACKING_EFFICACY'
    assert execution['status']=='CPU_DIAGNOSIS_EXITED' and execution['exit_code']==0
    assert execution['specification_sha256']==value['specification_sha256']==sha(spec_path)
    assert sha(execution['log'])==execution['log_sha256']
    assert identity['pid']==execution['identity']['pid']==read(output/'state.json')['pid']
    assert identity['start_ticks']==execution['identity']['start_ticks'] and exited(identity['pid'],identity=identity)
    assert identity['exact_command']==execution['command']
    assert value['cases']==spec['cases'] and value['source_hashes']==spec['source_hashes']
    assert all(sha(ROOT/p)==s for p,s in spec['source_hashes'].items())
    assert value['source_head_and_all_adapters_unchanged'] and value['all_conditions_before_frame_GT_indexing']
    assert not any(value[k] for k in ['model_fit','new_GPU_work','Dev_read','Test_read','new_tracking_metrics','goal_met'])
    assert value['peak_process_RSS_GiB']<=4 and value['unique_current_frames']==66 and value['Train_domains']==6
    expected={(s['sequence'],f) for s in spec['samples'] for f in s['frame_ids']}
    assert len(expected)==66 and len(value['frame_support'])==198
    cases=spec['cases'];sequences=[s['sequence'] for s in spec['samples']]
    for case in cases:
        rows=[r for r in value['frame_support'] if r['case']==case['id']]
        assert len(rows)==66 and {(r['sequence'],r['frame']) for r in rows}==expected
        assert all(r['past_frame']<r['frame'] for r in rows)
    def support(identifier,scope):
        return next(r for r in value['aggregated_support'] if r['case']==identifier and r['scope']==scope and r['threshold']=='0.6')
    def mean(identifier,scope,region):
        return next(r['mean'] for r in value['anchor_statistics'] if r['case']==identifier and r['scope']==scope and
            r['region']==region and r['variable']=='current_minus_past')
    def auc(identifier,scope):
        return next(r['current_minus_past_foreground_ranking_AUC'] for r in value['contrast_AUC'] if r['case']==identifier and r['scope']==scope)
    for row in value['aggregated_support']:
        assert row['current']-row['reference']==row['current_gain_vs_reference']-row['current_loss_vs_reference']
        assert row['current']-row['past']==row['current_gain_vs_past']-row['current_loss_vs_past']
    for scope in sequences+['ALL_TRAIN_SAMPLE']:
        assert len({support(c['id'],scope)['reference'] for c in cases})==1
    summary=[]
    for case in cases:
        identifier=case['id'];s=support(identifier,'ALL_TRAIN_SAMPLE')
        summary.append(dict(case=identifier,cue=case['cue'],reference_support06=s['reference'],current_support06=s['current'],
            past_support06=s['past'],current_minus_reference_support06=s['current']-s['reference'],
            current_minus_past_support06=s['current']-s['past'],
            weak_foreground_mean_current_minus_past=mean(identifier,'ALL_TRAIN_SAMPLE','weak_foreground'),
            background_mean_current_minus_past=mean(identifier,'ALL_TRAIN_SAMPLE','far_background'),
            current_minus_past_foreground_ranking_AUC=auc(identifier,'ALL_TRAIN_SAMPLE'),
            domains_positive_current_minus_past_support06=sum(support(identifier,s)['current']>support(identifier,s)['past'] for s in sequences),
            domains_positive_weak_foreground_contrast=sum(mean(identifier,s,'weak_foreground') is not None and mean(identifier,s,'weak_foreground')>0 for s in sequences)))
    ir=next(c['id'] for c in cases if c['cue']=='ir');controls=[c['id'] for c in cases if c['cue']!='ir']
    summary_scope=dict(IR_domains_contrast_AUC_above_both_capacity_controls=sum(auc(ir,s) is not None and all(
        auc(c,s) is not None and auc(ir,s)>auc(c,s) for c in controls) for s in sequences),
        no_threshold_model_or_Test_primary_selection=True,full_Train_or_heldout_tracking_efficacy=False,supported_innovations=0,goal_met=False)
    docs=ROOT/'docs/fusion_research';stem=docs/'PERCEPTUAL_MFT_20261125_TRAIN_CUE_SPECIFICITY'
    targets=[stem.with_suffix(s) for s in ['.json','.md']]+[Path(str(stem)+'_EXECUTION.json')]
    assert not any(p.exists() for p in targets)
    raw=(output/'complete.json').read_bytes();assert len(raw)<2**20
    stem.with_suffix('.json').write_bytes(raw)
    receipt=dict(status='EXPORTED_COMPLETE_ALL3_CUES_ALL6_TRAIN_DOMAINS_NO_TRACKING_CLAIM',UTC=datetime.now(timezone.utc).isoformat(),
        full_result_sha256=sha(output/'complete.json'),specification_sha256=sha(spec_path),source_hashes=value['source_hashes'],
        exporter_sha256=sha(__file__),actual_execution=execution,post_start_verified_identity=identity,
        actual_execution_sha256=sha(execution_path),post_start_verified_identity_sha256=sha(identity_path),
        summary=summary,scope_checks=summary_scope,goal_met=False,supported_innovations=0)
    Path(str(stem)+'_EXECUTION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    def fmt(v):return 'unknown' if v is None else f'{v:.6f}'
    lines=['# Train当前／过去辅助输入的任务相关性诊断','',
        '三个同预算已完成模型、六个Train视频、固定66个当前帧，全部条件完成。'
        '没有训练、GPU或Dev/Test读取；不是跟踪准确率或多模态创新证明。','',
        '| 输入 | RGB原支持 | 当前支持 | 过去支持 | 当前−RGB | 当前−过去 | 弱前景平均概率差 | 背景平均概率差 | 前景排序AUC |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append('| '+row['cue']+' | '+' | '.join(str(row[k]) for k in ['reference_support06','current_support06','past_support06',
            'current_minus_reference_support06','current_minus_past_support06'])+' | '+' | '.join(fmt(row[k]) for k in [
            'weak_foreground_mean_current_minus_past','background_mean_current_minus_past','current_minus_past_foreground_ranking_AUC'])+' |')
    lines+=['','支持量指原RGB固定解码锚点IoU≥0.5时，每个Train框的最高语义概率是否达到原定0.6。'
        '原0.1／0.7阈值、逐帧和全部增减量亦完整保存；不搜索阈值。没有NMS或跟踪，不能当作Recall/AP/HOTA。','',
        '| 视频 | 输入 | 当前−RGB支持 | 当前−过去支持 | 弱前景平均概率差 | 背景平均概率差 | 前景排序AUC |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for sequence in sequences:
        for case in cases:
            identifier=case['id'];s=support(identifier,sequence)
            lines.append(f"| {sequence} | {case['cue']} | {s['current']-s['reference']:+d} | {s['current']-s['past']:+d} | "+
                ' | '.join(fmt(v) for v in [mean(identifier,sequence,'weak_foreground'),mean(identifier,sequence,'far_background'),auc(identifier,sequence)])+' |')
    lines+=['','当前减过去的概率差用于区分原RGB锚点的框前景与远背景，AUC等于前景差值大于背景差值的比例加半数同值。'
        '它不是检测AP、身份识别率、概率校准或因果效应；汇总按锚点数加权，逐域结果不能省略。', '',
        '三个模型均完成72宏循环／三遍完整Train、相同145600参数、seed0。过去辅助固定为同视频第1帧，'
        '严格早于所有当前帧；只替换辅助FPN及结构key/valid，当前RGB及几何不变。'
        '这不是正式IR原图roll干预，也没有重新编码图像或宣称66帧新GPU等价。','',
        '原72帧采样去掉六个首帧，留下66帧；三个输入全部预测后才索引各帧Train标签。'
        '144个唯一当前／过去RGB/IR源文件与完成拟合的内容清单一致，原缓存数组SHA全部核对。'
        '检测头和全部adapter状态及梯度未变。','',
        f"CPU用时{value['seconds']:.3f}秒，峰值RSS{value['peak_process_RSS_GiB']:.3f}GiB。"
        '启动瞬间命令行曾为空，原记录保留；后续相同start_ticks及真实完整命令已单独核实，'
        '退出0由实际Popen句柄确认，不能把空命令摘要当作有效身份。','',
        '原始完整标量见相邻JSON，执行证明见_EXECUTION.json；不保存预测数组、图像或权重。'
        '历史Test暴露、RGB预训练见过Dev及伪IR由RGB生成的限制保持。独立创新0，Goal未达。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=receipt['status'],summary=summary,scope_checks=summary_scope),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
