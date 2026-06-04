---
slug: "/blog/modelopt-fp8-nvfp4"
date: "2026-06-05"
title: "vLLM-Omni 量化推理实践(1)"
description: "Qwen-Image-2512 的 FP8 / 混合 FP8/NVFP4：显存、保真度与并发吞吐。"
---

训练后量化是在不重训的前提下降低大型扩散 Transformer 显存与延迟成本的主要手段。对文生图而言，工程上要回答的问题很具体：能回收多少显存、输出保真度是否保持，以及在何种并发下低精度内核的算术收益开始压过其逐次调用的反量化开销。本文针对通过 **vLLM-Omni** 服务、运行在两块 NVIDIA B300（Blackwell Ultra，计算能力 10.3 / SM103，张量并行为 2）上的扩散模型 **Qwen-Image-2512** 给出实证回答。

评估两类量化。*离线*量化加载预量化的 ModelOpt checkpoint，由 vLLM-Omni 自动识别，测试了全 FP8 与混合 FP8/NVFP4 两个 checkpoint；*在线*量化则在加载时通过 `--quantization <method>` 对 BF16 checkpoint 施加量化。全部测量使用固定负载：1024x1024, 20 inference steps, seed 42, prompt='A ceramic teapot on a wooden table'。

## 实验设置与样例输出

预量化 checkpoint 无需 `--quantization` 标志，由 checkpoint 配置选择路径，且已在启动时核验内核选择、无静默回退到 BF16。

```bash
# BF16 基线
vllm serve Qwen/Qwen-Image-2512 --omni --tensor-parallel-size 2

# 离线 ModelOpt FP8（预量化 checkpoint，自动识别）
vllm serve feizhai123/qwen-image-2512-modelopt-fp8-dynamic-all --omni \
  --tensor-parallel-size 2 --linear-backend cutlass --force-cutlass-fp8 --trust-remote-code

# 离线 混合 FP8/NVFP4
vllm serve feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy --omni \
  --tensor-parallel-size 2 --linear-backend cutlass --force-cutlass-fp8 --trust-remote-code

# 在线 FP8（加载时对 BF16 权重做动态量化）
vllm serve Qwen/Qwen-Image-2512 --omni --tensor-parallel-size 2 --quantization fp8
```

三个量化输出与基线、彼此之间的校验和均不同，证实是真实的不同计算，而非缓存或静默回退。各配置的视觉保真度均得到保持，差异仅限于细小的高频细节，且在混合 FP8/NVFP4 配置上最明显——与其更激进的权重压缩一致。

<figure>
<div class="gal">
<div><img src="/images/modelopt/bf162512.png" alt="BF16"><div class="c">BF16（基线）</div></div>
<div><img src="/images/modelopt/online_fp8.png" alt="在线 FP8"><div class="c">在线 FP8</div></div>
<div><img src="/images/modelopt/fp8.png" alt="离线 FP8"><div class="c">离线 ModelOpt FP8</div></div>
<div><img src="/images/modelopt/mixed.png" alt="混合"><div class="c">混合 FP8/NVFP4</div></div>
</div>
<figcaption>图 1. 四种精度下的输出，提示词 “A ceramic teapot on a wooden table”，1024×1024，20 步，种子 42。</figcaption>
</figure>

## 单请求：延迟、显存与保真度

下表给出各配置的单请求延迟、双卡峰值显存、磁盘 checkpoint 体积，以及相对 BF16 基线的重建保真度（以 SSIM 与 PSNR 量化，均在匹配的提示词、种子与步数下计算）。

<div class="table-caption">表 1. 各配置相对 BF16 基线的单请求特征。</div>
<table>
<thead><tr><th>配置</th><th>延迟 (s)</th><th>峰值显存 (MiB)</th><th>Checkpoint</th><th>SSIM</th><th>PSNR (dB)</th></tr></thead>
<tbody>
<tr><td>BF16</td><td>1.160</td><td>82,804</td><td>54 GB</td><td>基线</td><td>基线</td></tr>
<tr><td>在线 FP8</td><td>1.157</td><td>70,504</td><td>复用 BF16</td><td>0.9984</td><td>39.34</td></tr>
<tr><td>离线 ModelOpt FP8</td><td>1.203</td><td>69,896</td><td>35 GB</td><td>0.983</td><td>29.1</td></tr>
<tr><td>混合 FP8/NVFP4</td><td>1.772</td><td>68,844</td><td>31 GB</td><td>0.968</td><td>26.3</td></tr>
</tbody>
</table>

峰值显存降低 15.6%（离线 FP8）与 16.9%（混合）；checkpoint 缩减 35% 与 43%。在线 FP8 取得最高保真度（PSNR 39.3 dB，SSIM 0.9984），因为它在加载时对 BF16 权重做动态量化，而非依赖单独标定的产物。

<figure>
<div class="chartbox"><svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" class="chart"><line x1="60" y1="250.0" x2="704" y2="250.0" class="grid"/><text x="51" y="254.0" class="tick" text-anchor="end">0</text><line x1="60" y1="192.0" x2="704" y2="192.0" class="grid"/><text x="51" y="196.0" class="tick" text-anchor="end">22</text><line x1="60" y1="134.0" x2="704" y2="134.0" class="grid"/><text x="51" y="138.0" class="tick" text-anchor="end">45</text><line x1="60" y1="76.0" x2="704" y2="76.0" class="grid"/><text x="51" y="80.0" class="tick" text-anchor="end">68</text><line x1="60" y1="18.0" x2="704" y2="18.0" class="grid"/><text x="51" y="22.0" class="tick" text-anchor="end">90</text><text x="13" y="134.0" class="axlab" transform="rotate(-90 13 134.0)" text-anchor="middle">峰值显存, 双卡 (GiB)</text><rect x="103.5" y="41.6" width="74.1" height="208.4" rx="2" fill="#6C7A89"/><text x="140.5" y="33.6" class="pt" fill="#6C7A89" text-anchor="middle">81</text><text x="140.5" y="271.0" class="tick" text-anchor="middle">BF16</text><rect x="264.5" y="72.5" width="74.1" height="177.5" rx="2" fill="#76b900"/><text x="301.5" y="64.5" class="pt" fill="#76b900" text-anchor="middle">69</text><text x="301.5" y="271.0" class="tick" text-anchor="middle">在线 FP8</text><rect x="425.5" y="74.0" width="74.1" height="176.0" rx="2" fill="#1B5A8E"/><text x="462.5" y="66.0" class="pt" fill="#1B5A8E" text-anchor="middle">68</text><text x="462.5" y="271.0" class="tick" text-anchor="middle">离线 ModelOpt FP8</text><rect x="586.5" y="76.7" width="74.1" height="173.3" rx="2" fill="#BD3613"/><text x="623.5" y="68.7" class="pt" fill="#BD3613" text-anchor="middle">67</text><text x="623.5" y="271.0" class="tick" text-anchor="middle">混合 FP8/NVFP4</text></svg></div>
<figcaption>图 2. 各配置双卡峰值显存，越低越好。约 16% 的节省主要来自权重存储，而非混合格式层的精度。</figcaption>
</figure>

<figure>
<div class="chartbox"><svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" class="chart"><line x1="60" y1="250.0" x2="704" y2="250.0" class="grid"/><text x="51" y="254.0" class="tick" text-anchor="end">0.00</text><line x1="60" y1="192.0" x2="704" y2="192.0" class="grid"/><text x="51" y="196.0" class="tick" text-anchor="end">0.50</text><line x1="60" y1="134.0" x2="704" y2="134.0" class="grid"/><text x="51" y="138.0" class="tick" text-anchor="end">1.00</text><line x1="60" y1="76.0" x2="704" y2="76.0" class="grid"/><text x="51" y="80.0" class="tick" text-anchor="end">1.50</text><line x1="60" y1="18.0" x2="704" y2="18.0" class="grid"/><text x="51" y="22.0" class="tick" text-anchor="end">2.00</text><text x="13" y="134.0" class="axlab" transform="rotate(-90 13 134.0)" text-anchor="middle">单请求延迟 (s)</text><rect x="103.5" y="115.4" width="74.1" height="134.6" rx="2" fill="#6C7A89"/><text x="140.5" y="107.4" class="pt" fill="#6C7A89" text-anchor="middle">1.16</text><text x="140.5" y="271.0" class="tick" text-anchor="middle">BF16</text><rect x="264.5" y="115.8" width="74.1" height="134.2" rx="2" fill="#76b900"/><text x="301.5" y="107.8" class="pt" fill="#76b900" text-anchor="middle">1.16</text><text x="301.5" y="271.0" class="tick" text-anchor="middle">在线 FP8</text><rect x="425.5" y="110.5" width="74.1" height="139.5" rx="2" fill="#1B5A8E"/><text x="462.5" y="102.5" class="pt" fill="#1B5A8E" text-anchor="middle">1.20</text><text x="462.5" y="271.0" class="tick" text-anchor="middle">离线 ModelOpt FP8</text><rect x="586.5" y="44.4" width="74.1" height="205.6" rx="2" fill="#BD3613"/><text x="623.5" y="36.4" class="pt" fill="#BD3613" text-anchor="middle">1.77</text><text x="623.5" y="271.0" class="tick" text-anchor="middle">混合 FP8/NVFP4</text></svg></div>
<figcaption>图 3. 单请求平均延迟，越低越好。量化线性层在单请求时承担反量化开销——混合配置中的 NVFP4 最为显著（+52.8%）；在线 FP8 与 BF16 在噪声范围内持平。</figcaption>
</figure>

<div class="callout">单请求场景低估了量化的价值：单请求时 GEMM 规模小、反量化开销占主导。低精度的算术与带宽优势只有在并发饱和内核时才会兑现——见下一节。</div>

## 并发下的吞吐

下图给出并发 1、4、8 个在途请求时的持续吞吐（images/s），采用批处理服务（`--step-execution --max-num-seqs 8`，每点 16 个请求）。可观察到明显的交叉：低并发下 BF16 最快，但随批次填满，量化格式的权重带宽节省压过其逐次开销。

<figure>
<div class="chartbox"><svg viewBox="0 0 720 350" xmlns="http://www.w3.org/2000/svg" class="chart"><line x1="54" y1="308.0" x2="686" y2="308.0" class="grid"/><text x="45" y="312.0" class="tick" text-anchor="end">0.0</text><line x1="54" y1="250.0" x2="686" y2="250.0" class="grid"/><text x="45" y="254.0" class="tick" text-anchor="end">0.3</text><line x1="54" y1="192.0" x2="686" y2="192.0" class="grid"/><text x="45" y="196.0" class="tick" text-anchor="end">0.6</text><line x1="54" y1="134.0" x2="686" y2="134.0" class="grid"/><text x="45" y="138.0" class="tick" text-anchor="end">0.9</text><line x1="54" y1="76.0" x2="686" y2="76.0" class="grid"/><text x="45" y="80.0" class="tick" text-anchor="end">1.2</text><line x1="54" y1="18.0" x2="686" y2="18.0" class="grid"/><text x="45" y="22.0" class="tick" text-anchor="end">1.5</text><text x="13" y="163.0" class="axlab" transform="rotate(-90 13 163.0)" text-anchor="middle">吞吐 (images/s)</text><text x="54.0" y="333.0" class="tick" text-anchor="middle">1</text><text x="370.0" y="333.0" class="tick" text-anchor="middle">4</text><text x="686.0" y="333.0" class="tick" text-anchor="middle">8</text><text x="370.0" y="345" class="axlab" text-anchor="middle">并发 (同时在途请求数)</text><polyline points="54.0,142.7 370.0,154.9 686.0,88.2" fill="none" stroke="#6C7A89" stroke-width="2.6"/><circle cx="54.0" cy="142.7" r="4" fill="#6C7A89"/><circle cx="370.0" cy="154.9" r="4" fill="#6C7A89"/><circle cx="686.0" cy="88.2" r="4" fill="#6C7A89"/><text x="694.0" y="92.2" class="pt" fill="#6C7A89">1.137</text><polyline points="54.0,142.9 370.0,140.8 686.0,69.6" fill="none" stroke="#76b900" stroke-width="2.6"/><circle cx="54.0" cy="142.9" r="4" fill="#76b900"/><circle cx="370.0" cy="140.8" r="4" fill="#76b900"/><circle cx="686.0" cy="69.6" r="4" fill="#76b900"/><text x="694.0" y="73.6" class="pt" fill="#76b900">1.233</text><polyline points="54.0,140.2 370.0,149.9 686.0,72.5" fill="none" stroke="#1B5A8E" stroke-width="2.6"/><circle cx="54.0" cy="140.2" r="4" fill="#1B5A8E"/><circle cx="370.0" cy="149.9" r="4" fill="#1B5A8E"/><circle cx="686.0" cy="72.5" r="4" fill="#1B5A8E"/><text x="694.0" y="76.5" class="pt" fill="#1B5A8E">1.218</text><polyline points="54.0,196.6 370.0,138.3 686.0,47.6" fill="none" stroke="#BD3613" stroke-width="2.6"/><circle cx="54.0" cy="196.6" r="4" fill="#BD3613"/><circle cx="370.0" cy="138.3" r="4" fill="#BD3613"/><circle cx="686.0" cy="47.6" r="4" fill="#BD3613"/><text x="694.0" y="51.6" class="pt" fill="#BD3613">1.347</text><rect x="528" y="208" width="160" height="96" rx="3" fill="#ffffff" opacity="0.72"/><rect x="536" y="212" width="12" height="12" rx="2" fill="#6C7A89"/><text x="554" y="222" class="leg">BF16</text><rect x="536" y="234" width="12" height="12" rx="2" fill="#76b900"/><text x="554" y="244" class="leg">在线 FP8</text><rect x="536" y="256" width="12" height="12" rx="2" fill="#1B5A8E"/><text x="554" y="266" class="leg">离线 ModelOpt FP8</text><rect x="536" y="278" width="12" height="12" rx="2" fill="#BD3613"/><text x="554" y="288" class="leg">混合 FP8/NVFP4</text></svg></div>
<figcaption>图 4. 吞吐随并发变化，越高越好，标注并发 8 处的端点值。并发 8 时三种量化配置均超过 BF16（1.137）：混合 1.347（+18%）、在线 FP8 1.233（+8%）、离线 FP8 1.218（+7%）。混合配置从并发 1 到 8 扩展 2.34×。</figcaption>
</figure>

该交叉是核心的运行结论：对延迟敏感的单流服务，本硬件上 BF16 仍更优；对吞吐导向的批处理服务，量化配置全面占优。这一“随机制依赖”与上游 Qwen3-Omni W4A4 工作（PR #4025）一致——量化优势同样在并发下、而非单流延迟中显现。

## 结论

对于通过 vLLM-Omni 服务、运行在 NVIDIA B300 上的 Qwen-Image-2512，FP8 与混合 FP8/NVFP4 量化在保持输出保真度的同时回收约 16% 运行显存、缩减 35%–43% checkpoint 体积，并在并发饱和内核后超过 BF16 吞吐。单流延迟敏感场景下 BF16 仍更优；追求吞吐时优先量化，其中混合 FP8/NVFP4 在并发下最快、峰值显存也最低。本文是该系列的第一篇，后续将展开在线量化、内核细节与更多模型上的实践。
