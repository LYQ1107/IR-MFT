#!/usr/bin/env python3
"""Publish analysis/figures from already audited Test and motion Dev only."""
import datetime
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/fusion_research'
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    inputs=[(DOC/'PERCEPTUAL_MFT_20261127_TRACKLET_R2_FULL_TEST.json','b5b4d338fee63e2170787ae171a67a3682f94ed1c6063d9db4bc2ef94afda1ba'),
            (DOC/'PERCEPTUAL_MFT_20261127_TRAIN_MOTION_PRIOR_FULL_DEV.json','90ffbdc2c7ec85d3484a3a60003c8b08cd1265ddfcdf618feb3ac049d57698a2')]
    for path,digest in inputs:assert sha(path)==digest
    test,motion=map(lambda item:read(item[0]),inputs)
    assert test['official_numeric_fields_checked']==1880 and motion['official_numeric_fields_checked']==1269
    selected=next(row for row in test['rows']if row['case']==test['chosen_case'])
    original,fresh,parent=test['references']
    model_rows=[dict(id=r['id'],combined47=r['combined47'],per_sequence47=r['per_sequence47'])for r in [original,fresh,parent]]
    model_rows.append(dict(id=selected['id'],combined47=selected['combined47'],per_sequence47=selected['per_sequence47']))
    base=original['combined47']['HOTA'];delta=selected['combined47']['HOTA']-base
    target,stretch=base+5,base+10
    per_video=[]
    p={r['seq']:r for r in parent['per_sequence47']}
    for row in selected['per_sequence47'][:-1]:
        per_video.append(dict(sequence=row['seq'],HOTA=row['HOTA'],parent_HOTA=p[row['seq']]['HOTA'],
            gain_over_same_IR_parent_pp=row['HOTA']-p[row['seq']]['HOTA'],
            identity_links=selected['coverage'][row['seq']]['identity_links']))
    nonzero_motion=[r for r in motion['rows']if r['case']['motion']!='original']
    best_new=max(nonzero_motion,key=lambda r:r['combined47']['HOTA'])
    result=dict(status='ANALYSIS_OF_AUDITED_FROZEN_TRACKLET_FULL_TEST_AND_NEGATIVE_MOTION_DEV',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_reports=[dict(file=str(p.relative_to(ROOT)),sha256=h)for p,h in inputs],
        selected_case=selected['case'],complete_Test_frames=15985,complete_Test_sequences=7,
        original_RGB_Test_HOTA=base,selected_Test_HOTA=selected['combined47']['HOTA'],
        HOTA_gain_over_original_RGB_pp=delta,HOTA_gain_over_same_IR_parent_pp=selected['combined47']['HOTA']-parent['combined47']['HOTA'],
        HOTA_gain_over_fresh_B1_RGB_pp=selected['combined47']['HOTA']-fresh['combined47']['HOTA'],
        minimum_target_HOTA=target,stretch_target_HOTA=stretch,gap_to_plus5_HOTA=target-selected['combined47']['HOTA'],
        target_plus5_met=delta>=5,target_plus10_met=delta>=10,models=model_rows,per_video=per_video,
        selected_minus_original_RGB_all47={k:selected['combined47'][k]-original['combined47'][k]for k in selected['combined47']},
        selected_minus_same_IR_parent_all47={k:selected['combined47'][k]-parent['combined47'][k]for k in selected['combined47']},
        positive_videos_over_same_IR_parent=sum(r['gain_over_same_IR_parent_pp']>0 for r in per_video),
        negative_videos_over_same_IR_parent=sum(r['gain_over_same_IR_parent_pp']<0 for r in per_video),
        equal_videos_over_same_IR_parent=sum(r['gain_over_same_IR_parent_pp']==0 for r in per_video),
        Test_selection_from_complete_Dev_only=True,all_related_tracklet_workers_actual_exit_zero=27,
        new_Test_stage_workers_actual_exit_zero=16,Test_independent_fields_checked=1880,
        motion_Dev_rows=[dict(case=r['case'],combined47=r['combined47'])for r in motion['rows']],
        best_new_motion_Dev_case=best_new['case'],best_new_motion_Dev_HOTA=best_new['combined47']['HOTA'],
        motion_new_Test=False,new_tracking_accuracy_created_by_this_analysis=False,GPU_work=False,
        historical_Test_previously_exposed=True,RGB_pretraining_included_current_Dev=True,IR_generated_from_RGB=True,
        goal_met=False,supported_innovations=0)
    report=DOC/'PERCEPTUAL_MFT_20261127_TRACKLET_TEST_MOTION_ANALYSIS_ZH.json'
    if report.exists():
        previous=read(report)
        assert {k:v for k,v in previous.items()if k!='UTC'}=={k:v for k,v in result.items()if k!='UTC'}
        result=previous  # A cosmetic redraw must never rewrite the verified data.
    else:
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    figures=DOC/'perceptual_20261127_tracklet_test_motion_figures';figures.mkdir(exist_ok=True)
    private=WORK/'tracklet_test_motion_figures';private.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'mft-tracklet-20261127'})
    labels=['Original RGB (fixed baseline)','Fresh B1 RGB','Native November IR','Native IR + late identity']
    colors=['#7b8794','#a2b6c8','#527aa5','#dc934b']
    fig,ax=plt.subplots(figsize=(8.4,4.1));values=[r['combined47']['HOTA']for r in model_rows]
    ax.barh(range(4),values,color=colors,height=.56);ax.set_yticks(range(4),labels);ax.invert_yaxis()
    for i,v in enumerate(values):ax.text(v+.14,i,f'{v:.3f}',va='center',fontsize=9)
    ax.axvline(target,color='#9c3038',linestyle='--',label=f'+5 target {target:.3f}')
    ax.axvline(stretch,color='#72518a',linestyle=':',label=f'+10 target {stretch:.3f}')
    ax.set_xlim(0,42);ax.set_xlabel('Official COMBINED Test HOTA (absolute points)');ax.set_title('All 7 Test sequences / 15,985 frames',pad=30)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,fontsize=8,frameon=False);fig.tight_layout()
    fig.savefig(figures/'full_Test_fixed_original_RGB.svg');fig.savefig(private/'full_Test_fixed_original_RGB.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.1,3.7));values=[r['gain_over_same_IR_parent_pp']for r in per_video]
    ax.bar([r['sequence']for r in per_video],values,color=['#397f65'if x>0 else '#a86565'for x in values])
    ax.axhline(0,color='#66717b',linewidth=.7)
    for i,v in enumerate(values):ax.text(i,v+(.012 if v>=0 else -.012),f'{v:+.3f}',ha='center',va='bottom'if v>=0 else'top',fontsize=9)
    ax.set_ylim(-.17,.62);ax.set_ylabel('HOTA gain over identical native-IR parent');ax.set_title('Dev-selected late identity: complete Test per sequence')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    fig.savefig(figures/'full_Test_sequence_deltas.svg');fig.savefig(private/'full_Test_sequence_deltas.png',dpi=150);plt.close(fig)
    md='# 当前冻结方案完整Test：+0.373 HOTA，目标仍未达到\n\n'
    md+=f"固定原始RGB-only YOLOX-X＋ByteTrack完整Test **{base:.6f}**，Dev预选IR加身份续接 **{selected['combined47']['HOTA']:.6f}**，提升 **{delta:+.6f}个绝对HOTA点**。最低目标{target:.6f}，还差{result['gap_to_plus5_HOTA']:.6f}点；进一步目标{stretch:.6f}。没有达到+5或+10。\n\n"
    md+='| 完整Test条件 | HOTA | 对原始RGB | DetA | AssA | IDF1 | MOTA | IDSW |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for label,row in zip(labels,model_rows):
        m=row['combined47'];md+=f"| {label} | {m['HOTA']:.6f} | {m['HOTA']-base:+.6f} | {m['DetA']:.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']} |\n"
    md+='\n![完整Test与固定+5/+10目标](perceptual_20261127_tracklet_test_motion_figures/full_Test_fixed_original_RGB.svg)\n\n'
    md+=f"身份续接相对同IR视觉父模型只有{result['HOTA_gain_over_same_IR_parent_pp']:+.6f}点，相对补充fresh B1 RGB为{result['HOTA_gain_over_fresh_B1_RGB_pp']:+.6f}。原始RGB仍是用户固定的主基线。Dev上的+0.314156不能与Test的+0.270555相加。相对原始RGB，HOTA正向，但IDF1仍低{original['combined47']['IDF1']-selected['combined47']['IDF1']:.6f}点，IDSW多{selected['combined47']['IDSW']-original['combined47']['IDSW']}，不宣称所有指标均提高。\n\n"
    md+='身份续接对同IR父模型的逐视频为1正、4负、2相同，主要正向集中SN-009；没有按视频选型或删除负序列。25帧方案实际输出框、分数和顺序保留，只改首次输出身份，身份相关的HOTA匹配及其汇总可能随之变化。\n\n'
    md+='![逐视频增益与退化](perceptual_20261127_tracklet_test_motion_figures/full_Test_sequence_deltas.svg)\n\n'
    md+='全程固定新版11月DiffV2IR、seed0、原YOLOX-X父模型，物理4/5最多两卡。相关链27个实验worker实际0，其中完整Test阶段16个；独立Test审计实际0、1880官方字段。完整7段15985帧、全部47字段、原276列CSV、银行/图像/检测包来源和真实MOT核验见[正式Test报告](PERCEPTUAL_MFT_20261127_TRACKLET_R2_FULL_TEST.md)。不上传银行、MOT、图像或权重。\n\n'
    md+='[六项Train运动Q完整Dev](PERCEPTUAL_MFT_20261127_TRAIN_MOTION_PRIOR_FULL_DEV.md)也全部出齐，8个相关worker实际0、1269字段独立审计实际0。四项新Q条件全部明显低于late-only，不追加Test。完整Q提高检测侧DetA，却破坏身份关联；该负结果不删除。下一步按已登记的当前形状/宽高比方案继续，保护原中心与身份连续性；本分析不增加任何新精度。\n\n'
    md+='缓存回放排除编码、检测、图像IO和离线IR生成，不是端到端FPS。历史Test暴露、RGB原预训练见过当前Dev、伪IR由RGB派生等限制继续成立。整体系统差值不等于红外专属贡献。当前+5目标未达，独立贡献计数0，Goal仍active。\n'
    report.with_suffix('.md').write_text(md)
    (figures/'README.md').write_text('Graphs use already audited official COMBINED and all7 per-video Test rows; no averaging per-video HOTA, new evaluation, NN work or seeds. Fixed original RGB reference and absolute+5/+10 targets. No error bars from nonexistent seed repeats. SVGs are public; PNG verification previews remain private. Source: tools/plot_mft_tracklet_test_motion_analysis.py.\n')
    for name in ['full_Test_fixed_original_RGB.svg','full_Test_sequence_deltas.svg']:
        path=figures/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    files=[str(report.relative_to(ROOT)),str(report.with_suffix('.md').relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))]+[str((figures/n).relative_to(ROOT))for n in ['full_Test_fixed_original_RGB.svg','full_Test_sequence_deltas.svg','README.md']]
    allow=dict(files=files,hashes={n:sha(ROOT/n)for n in files},total_bytes=sum((ROOT/n).stat().st_size for n in files))
    (WORK/'tracklet_Test_motion_analysis_publication_allowlist_20261127_r1.json').write_text(json.dumps(allow,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],Test_HOTA=selected['combined47']['HOTA'],gain_pp=delta,gap_to_plus5=result['gap_to_plus5_HOTA'],files=len(files),bytes=allow['total_bytes'],new_accuracy=False,GPU_work=False)),flush=True)


if __name__=='__main__':main()
