# -*- coding: utf-8 -*-
"""Generate the Stuart markdown blog post (content/blog/modelopt-fp8-nvfp4.md)
from the benchmark JSON, with inline-SVG charts, raw-HTML tables and the image gallery."""
import json, os

SRC = r"C:\Users\lsy\Downloads\modelopt_results"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "content", "blog", "modelopt-fp8-nvfp4.md")

R = json.load(open(os.path.join(SRC, "results.json"), encoding="utf-8"))
O = json.load(open(os.path.join(SRC, "online_quant.json"), encoding="utf-8"))
sr, tp = R["single_request"], R["concurrency_throughput_img_s"]
on, cmp = O["online_fp8"], O["fp8_online_vs_offline_vs_bf16"]
setup = R["setup"]; fam = setup["model_family"]

C = {"bf16": "#6C7A89", "online_fp8": "#76b900", "fp8": "#1B5A8E", "mixed": "#BD3613"}
NAME = {"bf16": "BF16", "online_fp8": "在线 FP8", "fp8": "离线 ModelOpt FP8", "mixed": "混合 FP8/NVFP4"}

def line_chart(series, cats, ymax, w=720, h=350, ylab="吞吐 (images/s)"):
    pl, pr, pt, pb = 54, 34, 18, 42
    pw, ph = w - pl - pr, h - pt - pb
    X = lambda i: pl + pw * i / (len(cats) - 1); Y = lambda v: pt + ph * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for t in range(6):
        v = ymax * t / 5; y = Y(v)
        s.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+pw}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{pl-9}" y="{y+4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    s.append(f'<text x="13" y="{pt+ph/2}" class="axlab" transform="rotate(-90 13 {pt+ph/2})" text-anchor="middle">{ylab}</text>')
    for i, c in enumerate(cats):
        s.append(f'<text x="{X(i):.1f}" y="{pt+ph+25:.1f}" class="tick" text-anchor="middle">{c}</text>')
    s.append(f'<text x="{pl+pw/2:.1f}" y="{h-5}" class="axlab" text-anchor="middle">并发 (同时在途请求数)</text>')
    for k, vals in series:
        col = C[k]
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}"/>')
        s.append(f'<text x="{X(len(vals)-1)+8:.1f}" y="{Y(vals[-1])+4:.1f}" class="pt" fill="{col}">{vals[-1]:.3f}</text>')
    # legend — anchored to the bottom-right corner of the plot
    lw = 150; lx = pl + pw - lw; ly0 = pt + ph - (len(series) * 22) - 8
    s.append(f'<rect x="{lx-8}" y="{ly0-4}" width="{lw+10}" height="{len(series)*22+8}" rx="3" fill="#ffffff" opacity="0.72"/>')
    ly = ly0
    for k, _ in series:
        s.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{C[k]}"/>')
        s.append(f'<text x="{lx+18}" y="{ly+10}" class="leg">{NAME[k]}</text>')
        ly += 22
    s.append("</svg>"); return "".join(s)

def bar_chart(items, ymax, ylab="", fmt="{:.0f}", w=720, h=300):
    pl, pr, pt, pb = 60, 16, 18, 50
    pw, ph = w - pl - pr, h - pt - pb
    n = len(items); bw = pw / n * 0.46; Y = lambda v: pt + ph * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="chart">']
    for t in range(5):
        v = ymax * t / 4; y = Y(v)
        s.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+pw}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{pl-9}" y="{y+4:.1f}" class="tick" text-anchor="end">{fmt.format(v)}</text>')
    if ylab:
        s.append(f'<text x="13" y="{pt+ph/2}" class="axlab" transform="rotate(-90 13 {pt+ph/2})" text-anchor="middle">{ylab}</text>')
    for i, (k, v) in enumerate(items):
        cx = pl + pw * (i + 0.5) / n; x = cx - bw / 2; y = Y(v)
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{pt+ph-y:.1f}" rx="2" fill="{C[k]}"/>')
        s.append(f'<text x="{cx:.1f}" y="{y-8:.1f}" class="pt" fill="{C[k]}" text-anchor="middle">{fmt.format(v)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{pt+ph+21:.1f}" class="tick" text-anchor="middle">{NAME[k]}</text>')
    s.append("</svg>"); return "".join(s)

cats = ["1", "4", "8"]
thr = [
    ("bf16",       [tp["bf16"]["c1"], tp["bf16"]["c4"], tp["bf16"]["c8"]]),
    ("online_fp8", [on["concurrency_throughput_img_s"]["c1"], on["concurrency_throughput_img_s"]["c4"], on["concurrency_throughput_img_s"]["c8"]]),
    ("fp8",        [tp["fp8"]["c1"], tp["fp8"]["c4"], tp["fp8"]["c8"]]),
    ("mixed",      [tp["mixed"]["c1"], tp["mixed"]["c4"], tp["mixed"]["c8"]]),
]
ch_thr  = line_chart(thr, cats, 1.5)
ch_vram = bar_chart([("bf16", sr["bf16"]["peak_vram_mib_2gpu"]/1024), ("online_fp8", on["single_request"]["peak_vram_mib_2gpu"]/1024),
                     ("fp8", sr["fp8"]["peak_vram_mib_2gpu"]/1024), ("mixed", sr["mixed"]["peak_vram_mib_2gpu"]/1024)], 90, "峰值显存, 双卡 (GiB)")
ch_lat  = bar_chart([("bf16", sr["bf16"]["latency_mean_s"]), ("online_fp8", on["single_request"]["latency_mean_s"]),
                     ("fp8", sr["fp8"]["latency_mean_s"]), ("mixed", sr["mixed"]["latency_mean_s"])], 2.0, "单请求延迟 (s)", "{:.2f}")

def fig(svg, cap):
    return f'<figure>\n<div class="chartbox">{svg}</div>\n<figcaption>{cap}</figcaption>\n</figure>'

table1 = f'''<div class="table-caption">表 1. 各配置相对 BF16 基线的单请求特征。</div>
<table>
<thead><tr><th>配置</th><th>延迟 (s)</th><th>峰值显存 (MiB)</th><th>Checkpoint</th><th>SSIM</th><th>PSNR (dB)</th></tr></thead>
<tbody>
<tr><td>BF16</td><td>{sr['bf16']['latency_mean_s']:.3f}</td><td>{sr['bf16']['peak_vram_mib_2gpu']:,}</td><td>{sr['bf16']['ckpt_gb']} GB</td><td>基线</td><td>基线</td></tr>
<tr><td>在线 FP8</td><td>{on['single_request']['latency_mean_s']:.3f}</td><td>{on['single_request']['peak_vram_mib_2gpu']:,}</td><td>复用 BF16</td><td>{on['single_request']['ssim_vs_bf16']}</td><td>{on['single_request']['psnr_db_vs_bf16']}</td></tr>
<tr><td>离线 ModelOpt FP8</td><td>{sr['fp8']['latency_mean_s']:.3f}</td><td>{sr['fp8']['peak_vram_mib_2gpu']:,}</td><td>{sr['fp8']['ckpt_gb']} GB</td><td>{sr['fp8']['ssim']}</td><td>{sr['fp8']['psnr_db']}</td></tr>
<tr><td>混合 FP8/NVFP4</td><td>{sr['mixed']['latency_mean_s']:.3f}</td><td>{sr['mixed']['peak_vram_mib_2gpu']:,}</td><td>{sr['mixed']['ckpt_gb']} GB</td><td>{sr['mixed']['ssim']}</td><td>{sr['mixed']['psnr_db']}</td></tr>
</tbody>
</table>'''

table2 = '''<div class="table-caption">表 2. 2× B300（SM103）上的在线量化支持矩阵。</div>
<table>
<thead><tr><th>方法</th><th>标志</th><th>结果</th><th>原因</th></tr></thead>
<tbody>
<tr><td>FP8</td><td><code>--quantization fp8</code></td><td>可用</td><td>—</td></tr>
<tr><td>INT8 (W8A8)</td><td><code>--quantization int8</code></td><td>加载失败</td><td>CUTLASS scaled-mm 拒绝 SM103：“Int8 not supported on SM103.”</td></tr>
<tr><td>MXFP8 (W8A8)</td><td><code>--quantization mxfp8</code></td><td>加载失败</td><td>NotImplementedError：仅 Ascend (NPU) / Intel (XPU)。</td></tr>
<tr><td>MXFP4 (W4A4)</td><td><code>--quantization mxfp4</code></td><td>加载失败</td><td>NotImplementedError：仅 Ascend；CUDA 尚未实现。</td></tr>
</tbody>
</table>'''

gallery = '''<figure>
<div class="gal">
<div><img src="/images/modelopt/bf162512.png" alt="BF16"><div class="c">BF16（基线）</div></div>
<div><img src="/images/modelopt/online_fp8.png" alt="在线 FP8"><div class="c">在线 FP8</div></div>
<div><img src="/images/modelopt/fp8.png" alt="离线 FP8"><div class="c">离线 ModelOpt FP8</div></div>
<div><img src="/images/modelopt/mixed.png" alt="混合"><div class="c">混合 FP8/NVFP4</div></div>
</div>
<figcaption>图 1. 四种精度下的输出，提示词 “A ceramic teapot on a wooden table”，1024×1024，20 步，种子 42。</figcaption>
</figure>'''

md = f'''---
slug: "/blog/modelopt-fp8-nvfp4"
date: "2026-06-05"
title: "vLLM-Omni 量化推理实践(1)"
description: "Qwen-Image-2512 的 FP8 / 混合 FP8/NVFP4：显存、保真度与并发吞吐。"
---

训练后量化是在不重训的前提下降低大型扩散 Transformer 显存与延迟成本的主要手段。对文生图而言，工程上要回答的问题很具体：能回收多少显存、输出保真度是否保持，以及在何种并发下低精度内核的算术收益开始压过其逐次调用的反量化开销。本文针对通过 **vLLM-Omni** 服务、运行在两块 NVIDIA B300（Blackwell Ultra，计算能力 10.3 / SM103，张量并行为 2）上的扩散模型 **{fam}** 给出实证回答。

评估两类量化。*离线*量化加载预量化的 ModelOpt checkpoint，由 vLLM-Omni 自动识别，测试了全 FP8 与混合 FP8/NVFP4 两个 checkpoint；*在线*量化则在加载时通过 `--quantization <method>` 对 BF16 checkpoint 施加量化。全部测量使用固定负载：{setup["image"]}。

## 实验设置与样例输出

预量化 checkpoint 无需 `--quantization` 标志，由 checkpoint 配置选择路径，且已在启动时核验内核选择、无静默回退到 BF16。

```bash
# BF16 基线
vllm serve Qwen/Qwen-Image-2512 --omni --tensor-parallel-size 2

# 离线 ModelOpt FP8（预量化 checkpoint，自动识别）
vllm serve feizhai123/qwen-image-2512-modelopt-fp8-dynamic-all --omni \\
  --tensor-parallel-size 2 --linear-backend cutlass --force-cutlass-fp8 --trust-remote-code

# 离线 混合 FP8/NVFP4
vllm serve feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy --omni \\
  --tensor-parallel-size 2 --linear-backend cutlass --force-cutlass-fp8 --trust-remote-code

# 在线 FP8（加载时对 BF16 权重做动态量化）
vllm serve Qwen/Qwen-Image-2512 --omni --tensor-parallel-size 2 --quantization fp8
```

三个量化输出与基线、彼此之间的校验和均不同，证实是真实的不同计算，而非缓存或静默回退。各配置的视觉保真度均得到保持，差异仅限于细小的高频细节，且在混合 FP8/NVFP4 配置上最明显——与其更激进的权重压缩一致。

{gallery}

## 单请求：延迟、显存与保真度

下表给出各配置的单请求延迟、双卡峰值显存、磁盘 checkpoint 体积，以及相对 BF16 基线的重建保真度（以 SSIM 与 PSNR 量化，均在匹配的提示词、种子与步数下计算）。

{table1}

峰值显存降低 15.6%（离线 FP8）与 16.9%（混合）；checkpoint 缩减 35% 与 43%。在线 FP8 取得最高保真度（PSNR 39.3 dB，SSIM 0.9984），因为它在加载时对 BF16 权重做动态量化，而非依赖单独标定的产物。

{fig(ch_vram, "图 2. 各配置双卡峰值显存，越低越好。约 16% 的节省主要来自权重存储，而非混合格式层的精度。")}

{fig(ch_lat, "图 3. 单请求平均延迟，越低越好。量化线性层在单请求时承担反量化开销——混合配置中的 NVFP4 最为显著（+52.8%）；在线 FP8 与 BF16 在噪声范围内持平。")}

<div class="callout">单请求场景低估了量化的价值：单请求时 GEMM 规模小、反量化开销占主导。低精度的算术与带宽优势只有在并发饱和内核时才会兑现——见下一节。</div>

## 并发下的吞吐

下图给出并发 1、4、8 个在途请求时的持续吞吐（images/s），采用批处理服务（`--step-execution --max-num-seqs 8`，每点 16 个请求）。可观察到明显的交叉：低并发下 BF16 最快，但随批次填满，量化格式的权重带宽节省压过其逐次开销。

{fig(ch_thr, f"图 4. 吞吐随并发变化，越高越好，标注并发 8 处的端点值。并发 8 时三种量化配置均超过 BF16（{tp['bf16']['c8']}）：混合 {tp['mixed']['c8']}（+18%）、在线 FP8 {on['concurrency_throughput_img_s']['c8']}（+8%）、离线 FP8 {tp['fp8']['c8']}（+7%）。混合配置从并发 1 到 8 扩展 2.34×。")}

该交叉是核心的运行结论：对延迟敏感的单流服务，本硬件上 BF16 仍更优；对吞吐导向的批处理服务，量化配置全面占优。这一“随机制依赖”与上游 Qwen3-Omni W4A4 工作（PR #4025）一致——量化优势同样在并发下、而非单流延迟中显现。

## 结论

对于通过 vLLM-Omni 服务、运行在 NVIDIA B300 上的 {fam}，FP8 与混合 FP8/NVFP4 量化在保持输出保真度的同时回收约 16% 运行显存、缩减 35%–43% checkpoint 体积，并在并发饱和内核后超过 BF16 吞吐。单流延迟敏感场景下 BF16 仍更优；追求吞吐时优先量化，其中混合 FP8/NVFP4 在并发下最快、峰值显存也最低。本文是该系列的第一篇，后续将展开在线量化、内核细节与更多模型上的实践。
'''

open(OUT, "w", encoding="utf-8").write(md)
print("wrote", OUT, len(md), "chars")
