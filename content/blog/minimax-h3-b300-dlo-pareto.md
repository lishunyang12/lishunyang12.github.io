---
slug: "/blog/minimax-h3-b300-dlo-pareto"
date: "2026-08-08"
title: "MiniMax-H3 on 8×B300：工业级 DP×SP / DLO Pareto 验证"
description: "8×B300 上的 MiniMax-H3 50-step 长稳态测试：T2VA n=20、FL2VA/Ref2VA、rank-local 与 allgather、P50/P95、功耗、能耗和显存。"
---

<div class="tldr"><strong>TL;DR</strong>：固定 8 块 B300、MiniMax-H3 BF16、768×1344、124 帧、50-step 时，没有一个拓扑全面胜出。<strong>DP1×SP8 + allgather</strong> 给出最低延迟（wave P50/P95 34.55/35.25 s）；<strong>DP4×SP2 + allgather</strong> 是生产均衡点（151.89 videos/hour，P95 95.31 s）；<strong>DP8×SP1 + rank-local</strong> 给出最高吞吐和最低能耗（183.78 videos/hour，43.97 Wh/video）。不要在 DP8 使用 allgather：实测吞吐下降 4.1%，单卡峰值从 20.03 GiB 升到 94.03 GiB。</div>

MiniMax-H3 的视频 DiT 可以把同一组 GPU 用在两个方向：更多 **sequence parallelism（SP）** 降低单请求延迟，更多 **data parallelism（DP）** 提高节点并发。DLO（distributed layerwise offload）又有两种推进方式：DP wave allgather，或每个 rank 独立推进的 rank-local offload。

上一篇版本用两轮短 5-step wave 快速确定了 Pareto 形状。本文完成后续工业验证：三个推荐点各有 **20 个完整 50-step T2VA measured wave**，跨两次独立 engine 生命周期；另对 FL2VA first-frame I2VA 和 Ref2VA image+audio 各运行三个推荐点、每点 5 个 50-step wave。所有正式请求都校验输出 shape，并用外部 GPU telemetry 计算功耗、能耗和显存峰值。

## 结论先行

<div class="table-caption">表 1. T2VA 三个生产候选点。每点 n=20 measured wave；延迟为 wave 分位数，功率是 8 卡板级功率之和。</div>

| 服务目标 | 拓扑 / DLO 模式 | P50 (s) | P95 (s) | Videos/hour | CV | Peak/GPU (GiB) | Mean node power | Energy/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 最低延迟 | **DP1×SP8 / allgather** | **34.55** | **35.25** | 103.84 | 1.38% | 26.37 | 7.07 kW | 68.08 Wh |
| 生产均衡 | **DP4×SP2 / allgather** | 94.73 | 95.31 | 151.89 | 0.35% | 25.11 | 7.87 kW | 51.76 Wh |
| 最大吞吐 | **DP8×SP1 / rank-local** | 156.74 | 157.03 | **183.78** | **0.15%** | **20.05** | 8.08 kW | **43.97 Wh** |

这三个点都位于延迟—吞吐 Pareto frontier 上。DP1 适合低队列深度和交互请求；DP8 需要持续排队才能兑现吞吐；DP4 是更稳妥的默认起点。DP4 到 DP8 的吞吐只再增加 21.0%，wave P50 却增加 65.5%。

<figure>
<div class="chartbox"><svg viewBox="0 0 760 390" xmlns="http://www.w3.org/2000/svg" class="chart" role="img" aria-label="MiniMax-H3 B300 industrial T2VA latency throughput Pareto frontier"><line x1="80" y1="310" x2="700" y2="310" class="grid"/><line x1="80" y1="240" x2="700" y2="240" class="grid"/><line x1="80" y1="170" x2="700" y2="170" class="grid"/><line x1="80" y1="100" x2="700" y2="100" class="grid"/><line x1="80" y1="30" x2="700" y2="30" class="grid"/><text x="68" y="314" class="tick" text-anchor="end">90</text><text x="68" y="244" class="tick" text-anchor="end">115</text><text x="68" y="174" class="tick" text-anchor="end">140</text><text x="68" y="104" class="tick" text-anchor="end">165</text><text x="68" y="34" class="tick" text-anchor="end">190</text><line x1="80" y1="30" x2="80" y2="310" stroke="#6b7280"/><line x1="80" y1="310" x2="700" y2="310" stroke="#6b7280"/><text x="80" y="334" class="tick" text-anchor="middle">20</text><text x="287" y="334" class="tick" text-anchor="middle">70</text><text x="493" y="334" class="tick" text-anchor="middle">120</text><text x="700" y="334" class="tick" text-anchor="middle">170</text><text x="390" y="365" class="axlab" text-anchor="middle">Wave P50 latency (s) →</text><text x="18" y="170" class="axlab" transform="rotate(-90 18 170)" text-anchor="middle">Throughput (videos/hour) →</text><polyline points="140,271 389,137 645,47" fill="none" stroke="#76b900" stroke-width="3"/><circle cx="140" cy="271" r="7" fill="#1B5A8E"/><circle cx="389" cy="137" r="9" fill="#76b900"/><circle cx="645" cy="47" r="7" fill="#1B5A8E"/><text x="154" y="294" class="pt" fill="#1B5A8E">DP1×SP8 / AG</text><text x="403" y="159" class="pt" fill="#558b00">DP4×SP2 / AG · balanced</text><text x="633" y="74" class="pt" fill="#1B5A8E" text-anchor="end">DP8×SP1 / rank-local</text></svg></div>
<figcaption>图 1. 完整 50-step、n=20 的 T2VA frontier。向左是低延迟，向上是高吞吐。</figcaption>
</figure>

## 工业测试方法

| 项目 | 配置 |
|---|---|
| 节点 | 8× NVIDIA B300 SXM6 AC，NV18 全互联，单 NUMA |
| 模型 | MiniMax-H3 / FL2VA；Ref2VA 使用官方 Ref2VA transformer |
| 精度 / attention | BF16 / `CUDNN_ATTN` |
| 输出 | 768×1344，124 frames，5 秒音频 |
| DLO | `dlo_resident_layers=0`；allgather 或 rank-local |
| 执行 | `enforce_eager=True`；每个 engine 先跑一轮完整 50-step warmup |
| T2VA 长稳态 | DP1、DP4、DP8 各 20 个 measured wave，来自 5+15 两次 engine 生命周期 |
| FL2VA / Ref2VA | 每点 5 个 measured wave |
| 并发 | 等于 DP；DP4 wave 同时生成 4 条，DP8 同时生成 8 条 |
| Telemetry | 外部 `nvidia-smi`，目标间隔 0.5 s；实测中位 0.758 s、P95 1.134 s |

脚本请求 `num_inference_steps=50`，当前 MiniMax-H3 调度实际执行 **49 次 denoise update**。初始化、warmup、measured wave 和 shutdown 分开计时；P50/P95 使用排序样本上的线性插值。吞吐按 measured 输出总数除以 measured wave 总时间计算。

能耗是 8 卡 `power.draw` 之和的梯形积分，不减 idle baseline；采样缺口超过 2 秒不插值，并报告覆盖率。三个 n=20 T2VA 点的覆盖率是 99.89–100%。显存同时计算 engine 整个生命周期峰值和 measured-window 峰值；表 1 的三个推荐点两者相同。

每个 measured 输出必须同时满足：视频 shape 为 `[124, 768, 1344, 3]`，音频 shape 为 `[1, 2, 165600]`。本测试验证性能、稳定性和输出契约，不替代 VBench、CLIP 或主观音视频质量评估。

## allgather 与 rank-local：不能全局选一个

<div class="table-caption">表 2. 同一 T2VA workload 下的配对对照，每格来自 5 个 50-step wave。</div>

| 拓扑 | 模式 | P50 (s) | P95 (s) | Videos/hour | Measured peak/GPU | Energy/video |
|---|---|---:|---:|---:|---:|---:|
| DP1×SP8 | rank-local | 78.92 | 78.98 | 45.62 | 26.26 GiB | 94.06 Wh |
| DP1×SP8 | **allgather** | **34.24** | **34.78** | **104.64** | **23.53 GiB** | **67.75 Wh** |
| DP4×SP2 | rank-local | 96.53 | 96.78 | 149.23 | **23.91 GiB** | 51.87 Wh |
| DP4×SP2 | **allgather** | **94.38** | **94.69** | **152.53** | 24.97 GiB | **51.67 Wh** |
| DP8×SP1 | **rank-local** | **156.92** | **157.17** | **183.55** | **20.03 GiB** | **43.97 Wh** |
| DP8×SP1 | allgather | 162.84 | 165.51 | 176.05 | 94.03 GiB | 45.08 Wh |

结论是分段的：

- **DP1**：该分支在 DP=1、SP>1 时让 DLO allgather 使用 SP group，因此不是 no-op。吞吐提高 129.4%，P50 降低 56.6%，能耗降低 28.0%。
- **DP4**：allgather 仍有约 2.2% 吞吐收益，显存增加约 1.06 GiB；适合作为均衡点。
- **DP8**：allgather 反而让吞吐下降 4.1%、P50 增加 3.8%，并在 GPU0 形成约 94 GiB 的非对称峰值。这里应使用 rank-local。

`dlo_use_allgather=False` 下，每个 rank 独立持有和迁移自己的 layerwise-offload 状态，显存不跨 rank 共享。CPU tensor/loader 对象也是进程本地；操作系统可能复用文件页缓存，但这不等于 DLO 显式共享一份 host tensor。

## FL2VA 与 Ref2VA frontier

T2VA 配对矩阵选出了三种模式：DP1 allgather、DP4 allgather、DP8 rank-local。以下用同样的分辨率、frames、50-step 和固定输入，验证 image/video/audio 条件路径。

<div class="table-caption">表 3. 多模态任务，每点 n=5。Lifecycle peak 包含模型初始化，Measured peak 只覆盖正式请求。</div>

| Workload | 拓扑 / 模式 | P50 (s) | P95 (s) | Videos/hour | Lifecycle / measured peak | Energy/video |
|---|---|---:|---:|---:|---:|---:|
| FL2VA first-frame I2VA | DP1×SP8 / allgather | **37.92** | **38.32** | 94.97 | 34.67 / 23.59 GiB | 73.87 Wh |
| FL2VA first-frame I2VA | DP4×SP2 / allgather | 102.90 | 103.13 | 139.91 | 25.16 / 25.16 GiB | 55.98 Wh |
| FL2VA first-frame I2VA | DP8×SP1 / rank-local | 170.22 | 178.90 | **165.93** | 93.84 / 31.14 GiB | **46.88 Wh** |
| Ref2VA image+audio | DP1×SP8 / allgather | **55.44** | **56.29** | 64.65 | 102.18 / 97.21 GiB | 103.67 Wh |
| Ref2VA image+audio | DP4×SP2 / allgather | 167.01 | 174.78 | 85.26 | 98.29 / 98.29 GiB | 85.52 Wh |
| Ref2VA image+audio | DP8×SP1 / rank-local | 287.62 | 289.60 | **100.07** | **21.00 / 19.36 GiB** | **74.06 Wh** |

FL2VA 与 Ref2VA 仍呈相同方向：SP 买低延迟，DP 买吞吐和每视频能效。但显存行为不能从 T2VA 外推。Ref2VA 的 allgather 点在条件编码与正式请求期间接近 100 GiB/GPU；DP8 rank-local 虽延迟最高，却显著降低显存并给出最高吞吐和最低能耗。

Ref2VA checkpoint 由官方 Ref2VA transformer 加模型索引组成；text encoder、video/audio VAE、processor 和 tokenizer 的 85 个共享文件先按 Hugging Face LFS SHA256 确认与 FL2VA 逐文件一致，再复用 FL2VA 文件。Ref2VA transformer 放在 `/dev/shm`，因此本文不拿 Ref2VA 冷启动时间做可移植结论；measured steady-state 不受这一存储路径比较影响。

## 实验发现并修复的 subgroup broadcast bug

[vLLM-Omni #5911](https://github.com/vllm-project/vllm-omni/pull/5911) 继续支持 `dlo_use_allgather=False` 下的 rank-local 独立请求；[#5864](https://github.com/vllm-project/vllm-omni/pull/5864) 则主要修复 `dlo_use_allgather=True` 的 DP wave、结果路由和异常请求死锁。

组合 DP×SP 时，DiT process group 的 local rank 0 不一定是 global rank 0。最初只修正了 tensor helper 的 shape/output broadcast；正式 DP4 Ref2VA warmup 又命中了音频条件路径：

```text
ValueError: Global rank 0 is not part of group <ProcessGroup ...>
```

最终检查并修正了该 pipeline 中全部 8 个 DiT subgroup broadcast site，包括 tensor shape/output、standalone audio 长度与 tensor、reference-video object/shape、video audio 长度和 `has_audio` 标志。统一模式是：

```diff
- dist.broadcast(tensor, src=0, group=group)
+ dist.broadcast(
+     tensor,
+     src=dist.get_global_rank(group, 0),
+     group=group,
+ )
```

修复后先运行 DP4×SP2 Ref2VA 2-step image+audio smoke，验证 4 个并发视频和音频 shape；通过后才重新运行 5×50-step 正式 case。失败日志、smoke 和正式结果分别保存，没有覆盖第一次失败。这个 8-site 修复是基于 #5911 commit 的本地 diff，不应误认为已包含在该 PR commit 中。

## 冷启动、退出和错误审计

T2VA 两次独立 engine 生命周期的初始化范围是：DP1 322.7–327.0 s，DP4 310.2–311.9 s，DP8 420.8–420.9 s。rank-local 或 allgather 解决稳态权重推进方式，并不会消除 checkpoint 加载和多进程初始化。生产服务应复用长生命周期 engine。

15 个成功 case（12 个正式 n=5 case + 3 个 soak n=15 case）均满足：`status=passed`、输出 shape 正确、无 OOM、无 worker death、无 segfault、无 attention operation-creation failure。7 个 case 在完成输出后出现一条：

```text
[AsyncOmniEngine] Orchestrator did not stop within 30.0 seconds; continuing cleanup
```

它是 teardown 延迟，不是推理失败；外部检查最终确认 8 卡回到 0 MiB / 0% utilization。最早的 DP1 rank-local summary 标记为 `delayed-reap`，其余为 clean。频繁扩缩容场景仍应单独修复这个退出长尾。

## 复现与原始数据

核心命令形态如下；生产候选点把 `DP/SP/ALLGATHER` 分别设为 `1/8/1`、`4/2/1`、`8/1/0`：

```bash
PYTHONPATH=/path/to/vllm-omni \
DLO_USE_ALLGATHER=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python run_dp_sp_point.py \
  --model /path/to/MiniMax-H3/FL2VA \
  --dp 4 --sp 2 --num-gpus 8 \
  --steps 50 --warmup-steps 50 --repeats 15 \
  --output summary.json
```

可下载的聚合与复现文件：

- [全部正式 case CSV](/data/minimax-h3-b300-dlo-industrial/industrial_results.csv)
- [T2VA n=20 汇总 CSV](/data/minimax-h3-b300-dlo-industrial/t2va_soak_n20.csv)
- [全部 wave 原始样本](/data/minimax-h3-b300-dlo-industrial/wave_samples.csv)
- [完整 JSON（含 per-GPU memory 与审计）](/data/minimax-h3-b300-dlo-industrial/industrial_results.json.txt)
- [环境、commit、diff 与输入 SHA256](/data/minimax-h3-b300-dlo-industrial/environment.json.txt)
- [T2VA runner](/data/minimax-h3-b300-dlo-industrial/run_dp_sp_point.py) / [multimodal runner](/data/minimax-h3-b300-dlo-industrial/run_multimodal_dp_sp_point.py)

测试基于 source commit `9e73ee1a50ce247c638052011914d8027d717f28` 加上述本地 subgroup fix，PyTorch `2.11.0+cu130`。环境同时打印了 **vLLM-Omni `0.26.0rc2.dev11` 与 vLLM `0.24.0` major/minor 不匹配警告**。所有本文 workload 均通过，但复现和生产部署应优先使用仓库声明的匹配版本，不能把这次通过当成任意版本组合都受支持。

最终建议很简单：**低延迟选 DP1 allgather，生产均衡选 DP4 allgather，持续高队列吞吐选 DP8 rank-local；模式必须随 DP 改变。**
