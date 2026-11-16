---
layout: post
title: "vLLM 的 FP8 KV cache 会让小号 Qwen2 系推理模型悄悄“变傻”，一行配置就能自查"
date: 2026-11-13
lang: zh-CN
description: "vLLM 0.33 默认的 FP8 KV cache 让 R1-Distill-Qwen-7B 在 MATH-500 上从 71% 掉到 0.5%，不报错。原因在第 0 层的 key 偏置，第 0 层保持 BF16 就能修好。"
image: /assets/fp8_qwen2_zh.png
---

[English version](/2026/11/13/fp8-kv-cache-qwen2-vllm/)

**一句话总结**
- 用 vLLM 默认的 FP8 KV cache，7B 及以下的 Qwen2 系模型做长推理会崩：R1-Distill-Qwen-7B 在 MATH-500 上从 71% 掉到 0.5%，不报错也没有警告。
- 同样的 R1 蒸馏换成 Llama 底座就没事（67.0% 对 68.4%），14B 的 Qwen 模型也没事。
- 原因在第 0 层：Qwen2 的 key 偏置让这一层的 K 几乎全是均值，FP8 的舍入误差落在注意力真正用到的那一小部分上。一行配置就能自查和修好：`kv_cache_dtype_skip_layers=["0"]`。

在 vLLM 里打开 `kv_cache_dtype="fp8"`，用 DeepSeek-R1-Distill-Qwen-7B 做数学题，正确率从 71% 掉到 0.5%。不报错，也没有警告，模型只是开始不停地重复：

```
To convert convert convert the point point point ...
```

下面是实测结果、原因和修法。

## 数据

vLLM 0.33.0，H200，MATH-500 前 200 题，贪心解码，最多生成 4096 个 token，FP8（E4M3）KV cache，用 vLLM 默认的（未标定的）缩放因子。

| 模型 | BF16 KV | FP8 KV | FP8，第 0 层保持 BF16 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 71.0% | 0.5% | 71.5% |
| Qwen2.5-7B-Instruct | 73.5% | 1.5% | 72.0% |
| DeepSeek-R1-Distill-Qwen-1.5B | 61.0% | 0.0% | 61.0% |
| DeepSeek-R1-Distill-Qwen-14B | 70.0% | 73.0% | – |
| DeepSeek-R1-Distill-Llama-8B（全部 500 题） | 68.4% | 67.0% | 68.8% |

{% include fp8_qwen2_zh.svg %}

换成只让**最后一层**保持 BF16，完全没用（R1-Distill-Qwen-7B，40 题上 0.0%）。14B 模型没事；我离线测困惑度，Qwen2.5-14B 和 Llama-3.1-8B 也不受影响。

最干净的对照是 DeepSeek-R1-Distill-Llama-8B：同样的 R1 蒸馏，底座换成没有 key 偏置的 Llama-3.1。全部层用 FP8 只差 1.4 个点（配对置换检验 p = 0.47，全部 500 题跑一次）。崩溃跟着 Qwen2 底座走，不是推理模型或 FP8 本身的问题。

## 自查

同一个提示，把第 0 层排除在 FP8 之外再跑一次：

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["0"],
)
```

输出恢复正常，就是这个问题。代价是 28 层里有 1 层用 BF16。

注意：用 FlashAttention-3 后端时，有些模型会直接崩（[vLLM #65821](https://github.com/vllm-project/vllm/issues/65821)，修复在 [#65891](https://github.com/vllm-project/vllm/pull/65891)，还没合并）。加上 `attention_backend="FLASHINFER"` 就能绕开。

## 为什么是第 0 层

Qwen2 的 key 投影带偏置。到了第 0 层，偏置占了大头：各通道 K 均值的范数是 K 范数的 98%（Llama-3.1-8B 是 68%）。

注意力真正用到的，是 K 里**随 token 变化**的那一小部分。均值对同一个 query 的所有注意力分数加的是同一个数，softmax 会把它抵消掉。可 FP8 的舍入误差是按整个数值的大小算的，于是全落在了那一小部分上。好比站在山顶上量几个人的身高差，手里的尺子一格一百米。

整张量的缩放因子能防止大的数值溢出，但改变不了 E4M3 的相对精度，所以消不掉落在那一小部分上的舍入误差。何况第 0 层根本没有溢出：在 wikitext 上，它最大的 key 值是 203（R1-Distill-Qwen-7B）和 172（Qwen2.5-7B-Instruct），远低于 E4M3 的上限 448。（这里的正确率都用 vLLM 默认的缩放因子，我没有在 vLLM 里跑过标定后的缩放因子。）

而且第 0 层的误差不会只留在第 0 层。只把第 0 层量化成 FP8，最终输出的 KL 是中位数那一层的约 1300 倍（离线，R1-Distill-Qwen-7B，wikitext）。在长长的思维链里，误差一路累积，最后变成胡言乱语。离线测 Qwen2.5-7B 的 wikitext 困惑度：全部 FP8 从 6.31 涨到 78.2，保留第 0 层是 6.59。

### “可 NVIDIA 说是最后一层”

NVIDIA 的 [TensorRT-Edge-LLM 文档](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/FP8KV.html)指出 Qwen2.5-7B-Instruct 用 FP8 KV 会明显掉点，原因是最后一层的数值很大。两件事都对：最后一层 K 的均值几乎和第 0 层一样大（均值的范数 240 对 274，中间各层约 12）。但数值大不等于敏感：只量化最后一层，KL 是中位数那一层的 8 倍，第 0 层是 1300 倍；只让最后一层保持 BF16，正确率也救不回来。

## 修法

1. **跳过第 0 层**（见上）。简单，现在就能用。
2. **量化前减掉一个固定的 K 中心。** 在 RoPE 之后给所有 key 减去同一个向量，同一个 query 的所有注意力分数只差一个常数，softmax 的结果不变。用在一段文本上测得的各通道 K 均值，R1-Distill-Qwen-7B 上的 FP8 回到 69.5%（离线模拟量化，提示部分用 BF16；同样设置下 BF16 是 73.5%）。
3. **直接用 key 的偏置当这个中心。** 阿里的 rtp-llm 已经这么做了（[alibaba/rtp-llm#1583](https://github.com/alibaba/rtp-llm/pull/1583)）：不用标定，不加算子，只作用于带 QKV 偏置的层。vLLM 里有一个还没合并的 PR（[#59674](https://github.com/vllm-project/vllm/pull/59674)）针对 INT4 cache 走了一条精确的路：缓存前去掉偏置，再在注意力分数里把它随位置变化的那部分加回来。

## 出处

这个机制不是我发现的。Liu 等人的《Quantization Hurts Reasoning?》（[arXiv 2504.04823](https://arxiv.org/abs/2504.04823)）已经在 4 位 KV cache 上指出，R1-Distill-Qwen-1.5B / 7B 的 K 离群通道来自继承自 Qwen2.5 的 key 偏置，图里画的就是第 0 层。rtp-llm 在自己的引擎里修好了 FP8 的情况。vLLM 官方的 [FP8 KV 文章](https://vllm.ai/blog/2026-04-22-fp8-kvcache)（2026 年 4 月）讲的是 Qwen3 / Qwen3.5 和长上下文的累加精度问题，没有涉及 Qwen2 系列。在 vLLM 内部，INT4 cache 上的同一个问题已经有人报告（#59671），内核修复在 [#59674](https://github.com/vllm-project/vllm/pull/59674)，启动时的警告在 [#63081](https://github.com/vllm-project/vllm/pull/63081)；#63081 还测到默认缩放因子的 FP8 cache 让 Qwen2.5-7B 和 1.5B 在检索任务上崩溃，并给出 `--kv-cache-dtype-skip-layers 0` 这个修法。我补充的是推理基准上的实测（MATH-500，R1-Distill 系列）、Llama 底座的对照，以及第 0 层和最后一层的对比。

讨论在 [vLLM #41343](https://github.com/vllm-project/vllm/issues/41343)。如果你在别的模型上也遇到过，欢迎告诉我。

*更正（2026 年 11 月 16 日）：本文第一版里 R1-Distill-Llama-8B 的数字，是在分词器加载出错（[transformers #45488](https://github.com/huggingface/transformers/issues/45488)）的情况下测的：vLLM 里的提示丢了空格和换行。Llama 那一行、图和正文的数字已换成修好分词器后重跑的结果（同样设置，全部 500 题，跑一次）：BF16 68.4%，FP8 67.0%，FP8 且第 0 层用 BF16 68.8%。结论不变：这个模型没有 key 偏置，FP8 下不崩。Qwen 的数字不受影响。*
