# 20项原源码审计：六项许可字段的证据补充

本次保持原20项审计和原固定提交不变，补查其中未记录许可的六项。官方GitHub API六次完整递归树响应均为HTTP200且未截断；六份根README和十四份原被查源码也均返回HTTP200，十四份文件SHA256全部与原审计一致。没有安装仓库、执行第三方代码或新增GPU/模型/数据工作。

| ID | 固定仓库/提交 | 全树blob数 | 标准命名许可文件 | 根项目许可范围 |
| --- | --- | ---: | --- | --- |
| R03 | [SMSTracker @ ab85a8a6e6f1](https://github.com/Leezed525/SMSTracker/tree/ab85a8a6e6f1793a0d0ac20c268aae687f1abcab) | 168 | 未检出 | 未确认 |
| R04 | [FlexTrackV2 @ 30a5ff1b39b8](https://github.com/supertyd/FlexTrackV2/tree/30a5ff1b39b8f3004dde4018018b901e6cc1bf54) | 363 | 未检出 | 未确认 |
| R06 | [MOSSTrack @ 95f33ab2c927](https://github.com/lyssyl157/MOSSTrack/tree/95f33ab2c927c396cad9bdb8ee213fa04a49e0a4) | 476 | 未检出 | 未确认 |
| R08 | [PFGF @ df2a3c03cc21](https://github.com/liting1018/PFGF/tree/df2a3c03cc217204ae26f6d0afb8b0e0a628ec3f) | 6552 | 未检出 | 未确认 |
| R09 | [M2D-LIF @ 210c7ca22b76](https://github.com/Zhao-Tian-yi/M2D-LIF/tree/210c7ca22b7670e7d4629cca995427bf7e74653f) | 1251 | 未检出 | 未确认 |
| R13 | [UnTrack @ 8eec76ec912c](https://github.com/Zongwei97/UnTrack/tree/8eec76ec912c19e326e2b0020444f8f20c7d4355) | 269 | 未检出 | 未确认 |

“未检出”精确指返回完整树中，文件名包含license/licence/copying/copyright的文件未检出；不是对所有文件内容的穷尽法律判断，也不等于宣布仓库没有任何许可。被查README未检出相应许可关键词，仍不补写不存在的项目授权。原审计中的六个空字段现在有具体的检索范围、响应、时间和哈希来源，根项目许可范围继续标为未确认。

R09的[`ultralytics/nn/modules/block.py`](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/210c7ca22b7670e7d4629cca995427bf7e74653f/ultralytics/nn/modules/block.py)第1行直接给出AGPL-3.0声明。这是所查文件头的直接证据；原报告将其概称为整个仓库AGPL过宽，本补充收紧为文件级事实，不从这一处声明推定所有自定义项目文件的许可范围。没有复制该第三方实现。

另五项被查源码未检出许可关键词，不据此推断未查文件或继承依赖的许可。完整树API链接、原文件URL、行号、字节数、响应时间和SHA256在JSON中保留，原20项源码审计JSON不被重写。许可证据补查不算科学创新或准确率进展，也不使尚未达到的MFT目标完成。
