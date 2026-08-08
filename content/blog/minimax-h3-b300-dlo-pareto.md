---
slug: "/blog/minimax-h3-b300-dlo-pareto"
date: "2026-08-08"
title: "MiniMax-H3 on 8×B300：DP×SP Pareto Frontier"
description: "在 8 块 NVIDIA B300 上实测 MiniMax-H3 rank-local DLO 的四组 DP×SP 拓扑：延迟、吞吐、显存，以及 DP4×SP2 为何是均衡点。"
---

<div class="tldr"><strong>TL;DR</strong>：固定使用 8 块 B300 时，<strong>DP1×SP8</strong> 延迟最低，<strong>DP8×SP1</strong> 吞吐最高，<strong>DP4×SP2</strong> 是更均衡的部署点——相对 DP1×SP8，吞吐提高 3.29×，wave 延迟只增加 21.8%。四个点都位于 Pareto frontier 上；选择取决于服务更看重单请求延迟还是节点总吞吐。</div>

MiniMax-H3 的视频 DiT 可以把同一组 GPU 用在两个方向：用更多 **sequence parallelism（SP）** 加速单个请求，或用更多 **data parallelism（DP）** 同时处理独立请求。两者的乘积固定为 8，因此这不是单纯的“并行度越大越快”，而是一个延迟—吞吐交换问题。

本文在同一台 8× NVIDIA B300 SXM6 AC 节点上完整测量四个点：`(DP, SP) = (1,8), (2,4), (4,2), (8,1)`。DLO 使用 rank-local 独立请求模式（`dlo_use_allgather=False`），不再要求 DP ranks 对每层计算组成同一个 all-gather wave。

## 实验设置

| 项目 | 配置 |
|---|---|
| 模型 | MiniMax-H3 / FL2VA |
| GPU | 8× NVIDIA B300 SXM6 AC |
| 精度 / attention | BF16 / `CUDNN_ATTN` |
| 任务 | T2VA，768×1344，124 frames |
| DLO | `dlo_use_allgather=False`，`dlo_resident_layers=0` |
| 测量 | 2-step warmup；随后两轮 5-step wave |
| 并发 | 等于 DP；每个 DP replica 处理一个独立请求 |

脚本请求 `num_inference_steps=5`，当前 MiniMax-H3 调度实际执行 4 次 denoise update。以下吞吐和延迟均来自 warmup 之后两轮 measured wave 的平均值，不包含 engine 初始化；显存是外部 `nvidia-smi` 采样得到的单卡峰值。

## 四点结果

<div class="table-caption">表 1. 固定 8 GPU 的 DP×SP 延迟—吞吐矩阵。吞吐越高越好，延迟和显存越低越好。</div>

| 拓扑 | 并发 | Engine init (s) | Mean wave (s) | Mean request (s) | Videos/hour | Peak/GPU (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| **DP1×SP8** | 1 | 422.601 | **16.806** | **16.806** | 214.222 | n/a |
| **DP2×SP4** | 2 | 381.139 | 18.255 | 17.914 | 394.425 | 23.457 |
| **DP4×SP2** | 4 | **369.491** | 20.463 | 19.757 | 704.913 | 21.176 |
| **DP8×SP1** | 8 | 424.653 | 28.362 | 27.484 | **1017.884** | **20.014** |

<figure>
<div class="chartbox"><svg viewBox="0 0 760 390" xmlns="http://www.w3.org/2000/svg" class="chart" role="img" aria-label="MiniMax-H3 B300 DP SP latency throughput Pareto frontier"><line x1="80" y1="310" x2="700" y2="310" class="grid"/><line x1="80" y1="240" x2="700" y2="240" class="grid"/><line x1="80" y1="170" x2="700" y2="170" class="grid"/><line x1="80" y1="100" x2="700" y2="100" class="grid"/><line x1="80" y1="30" x2="700" y2="30" class="grid"/><text x="68" y="314" class="tick" text-anchor="end">0</text><text x="68" y="244" class="tick" text-anchor="end">275</text><text x="68" y="174" class="tick" text-anchor="end">550</text><text x="68" y="104" class="tick" text-anchor="end">825</text><text x="68" y="34" class="tick" text-anchor="end">1100</text><line x1="80" y1="30" x2="80" y2="310" stroke="#6b7280"/><line x1="80" y1="310" x2="700" y2="310" stroke="#6b7280"/><text x="80" y="334" class="tick" text-anchor="middle">15</text><text x="287" y="334" class="tick" text-anchor="middle">20</text><text x="493" y="334" class="tick" text-anchor="middle">25</text><text x="700" y="334" class="tick" text-anchor="middle">30</text><text x="390" y="365" class="axlab" text-anchor="middle">Mean wave latency (s) →</text><text x="18" y="170" class="axlab" transform="rotate(-90 18 170)" text-anchor="middle">Throughput (videos/hour) →</text><polyline points="155,255 215,210 306,131 632,51" fill="none" stroke="#76b900" stroke-width="3"/><circle cx="155" cy="255" r="7" fill="#1B5A8E"/><circle cx="215" cy="210" r="7" fill="#1B5A8E"/><circle cx="306" cy="131" r="9" fill="#76b900"/><circle cx="632" cy="51" r="7" fill="#1B5A8E"/><text x="168" y="278" class="pt" fill="#1B5A8E">DP1×SP8</text><text x="228" y="202" class="pt" fill="#1B5A8E">DP2×SP4</text><text x="319" y="151" class="pt" fill="#558b00">DP4×SP2 · knee</text><text x="620" y="78" class="pt" fill="#1B5A8E" text-anchor="end">DP8×SP1</text></svg></div>
<figcaption>图 1. 延迟—吞吐 Pareto frontier。向左代表更低 wave 延迟，向上代表更高节点吞吐；四点均不被其他点同时支配。</figcaption>
</figure>

从 DP1 增加到 DP2，吞吐提高 **1.84×**，wave 延迟增加 8.6%；从 DP2 增加到 DP4，吞吐再提高 **1.79×**，延迟增加 12.1%。而从 DP4 增加到 DP8，吞吐只再提高 **1.44×**，wave 延迟却增加 38.6%。因此 DP4×SP2 是曲线上最明显的 knee：它已经获得大部分并发吞吐，同时保留了两路 SP 对单请求的加速。

显存也随 SP 缩小而下降：DP2×SP4、DP4×SP2、DP8×SP1 的单卡峰值依次为 23.46、21.18、20.01 GiB。这里 DLO 已把全部 DiT block 设为非常驻，因此峰值差异主要来自每个请求的 activation、attention workspace 和通信缓冲区，而不是常驻层数。

## 如何选部署拓扑

| 目标 | 推荐 | 原因 |
|---|---|---|
| 最低单请求延迟 | **DP1×SP8** | 16.806 s/wave，全部 GPU 服务一个请求 |
| 延迟与吞吐均衡 | **DP4×SP2** | 704.9 videos/hour，wave 仅 20.463 s |
| 最大节点吞吐 | **DP8×SP1** | 1017.9 videos/hour，8 个独立请求并发 |
| 中间并发档 | **DP2×SP4** | 比 DP1 吞吐高 84%，延迟只高 8.6% |

实际服务还应根据流量形态选择。如果队列通常只有一个请求，DP8 的理论吞吐不会兑现；如果请求持续排队，DP1 则会让七份潜在并发长期闲置。生产默认我会从 **DP4×SP2** 开始，再按队列深度向两端调整。

## 实验中发现的 DP×SP 通信问题

[vLLM-Omni #5911](https://github.com/vllm-project/vllm-omni/pull/5911) 的核心是继续支持 `dlo_use_allgather=False` 下的 rank-local 独立请求；相比之下，[#5864](https://github.com/vllm-project/vllm-omni/pull/5864) 主要修复 `dlo_use_allgather=True` 的 DP wave、结果路由和异常请求死锁。

在 MiniMax-H3 组合 DP×SP 时，我们还遇到一个更具体的问题：第二个及之后的 SP process group 不包含 global rank 0，但 pipeline 的 group broadcast 仍把 `src=0` 当成 global rank。以 DP2×SP4 为例，第二组 ranks 是 `[4,5,6,7]`，因此会直接报错：

```text
ValueError: Global rank 0 is not part of group <ProcessGroup ...>
```

修复是把 group 内 local rank 0 映射回该 group 的 global rank：

```diff
- dist.broadcast(tensor, src=0, group=group)
+ dist.broadcast(tensor, src=dist.get_global_rank(group, 0), group=group)
```

这处两行修复用于 DP2×SP4 与 DP4×SP2 的测试；DP1×SP8 的唯一 group 本就包含 global rank 0，DP8×SP1 则每个 group 的 world size 为 1，因此不会触发该错误。也就是说，这不是性能调优，而是组合 DP×SP 正确性所需的 group-rank 修复。

## 冷启动与退出清理

四个配置的 engine 初始化都在 **369–425 秒**。rank-local DLO 解决的是稳态请求如何独立推进，并不会消除大模型 checkpoint 加载、进程初始化和 compile 的冷启动成本。部署时应该复用长生命周期 engine，不要为每个请求重启。

另一个观察是：部分点在所有 worker 已打印 `Shutdown complete` 后，orchestrator 仍可能超过 30 秒退出等待；GPU 最终都回到 0 MiB / 0% utilization。本文把它记录为 **cleanup delayed-reap**，不把已经完成且有完整输出的 compute wave 判为失败，但这仍值得单独修复，尤其是频繁启停或自动扩缩容场景。

## 限制与下一步

这次实验用于快速确定 Pareto 形状，只覆盖 FL2VA T2VA，并使用两轮短 5-step wave。它足以比较四种拓扑的相对位置，却不能替代最终的 50-step 长稳态验证。下一步应在相同 prompt、分辨率和 frames 下，对三个推荐点（DP1、DP4、DP8）各运行多轮 50-step，并补齐 P50/P95、功耗、DP1 峰值显存，以及 I2VA/Ref2VA 的对应 frontier。

当前结论很清楚：**SP 买低延迟，DP 买吞吐；8×B300 上的默认均衡解是 DP4×SP2。**
