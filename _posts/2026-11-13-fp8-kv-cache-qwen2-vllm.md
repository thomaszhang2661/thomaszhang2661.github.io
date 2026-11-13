---
layout: post
title: "FP8 KV cache can silently break small Qwen2-based reasoning models in vLLM. Here's the one-line check."
date: 2026-11-13
description: "In vLLM 0.33 the default FP8 KV cache drops R1-Distill-Qwen-7B from 71% to 0.5% on MATH-500, with no error. The cause is layer 0's key bias; keeping layer 0 in BF16 fixes it."
image: /assets/fp8_qwen2_en.png
---

[中文版](/2026/11/13/fp8-kv-cache-qwen2-vllm-zh/)

**TL;DR**
- With vLLM's default FP8 KV cache, Qwen2-based models at 7B and below fall apart on long reasoning:
  R1-Distill-Qwen-7B goes from 71% to 0.5% on MATH-500, with no error or warning.
- The same R1 distillation on a Llama base is fine (65.6% vs 66.6%). The 14B Qwen model is fine too.
- The cause is layer 0: Qwen2's key bias makes its keys almost all mean, and FP8's rounding error lands on the small
  part attention uses. Check and fix with one line: `kv_cache_dtype_skip_layers=["0"]`.

Turn on `kv_cache_dtype="fp8"` in vLLM, run DeepSeek-R1-Distill-Qwen-7B on a math benchmark, and accuracy drops from
71% to 0.5%. No error, no warning. The model just starts repeating itself:

```
To convert convert convert the point point point ...
```

Here is what I measured, why it happens, and how to fix it.

## The numbers

vLLM 0.33.0, H200, MATH-500 (first 200 problems), greedy decoding, up to 4,096 new tokens, FP8 (E4M3) KV cache with
vLLM's default (uncalibrated) scales.

| Model | BF16 KV | FP8 KV | FP8, layer 0 kept in BF16 |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 71.0% | 0.5% | 71.5% |
| Qwen2.5-7B-Instruct | 73.5% | 1.5% | 72.0% |
| DeepSeek-R1-Distill-Qwen-1.5B | 61.0% | 0.0% | 61.0% |
| DeepSeek-R1-Distill-Qwen-14B | 70.0% | 73.0% | – |
| DeepSeek-R1-Distill-Llama-8B (all 500 problems) | 66.6% | 65.6% | 66.6% |

{% include fp8_qwen2_en.svg %}

Keeping only the **last** layer in BF16 instead does nothing (R1-Distill-Qwen-7B: 0.0% on 40 problems). The 14B model
is fine, and so are Qwen2.5-14B and Llama-3.1-8B in my offline perplexity runs.

The cleanest control is DeepSeek-R1-Distill-Llama-8B: the same R1 distillation, on a Llama-3.1 base, which has no key
bias. FP8 on every layer costs it 1.0 point (paired permutation test p = 0.64, one run over all 500 problems). The
collapse comes with the Qwen2 base, not with reasoning or with FP8 as such.

## Check if you're hit

Run the same prompt with layer 0 excluded from FP8:

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["0"],
)
```

If the output goes back to normal, this is your problem. It costs one layer out of 28 in BF16.

One caveat: with the FlashAttention-3 backend this currently crashes on some models
([vLLM #65821](https://github.com/vllm-project/vllm/issues/65821); a fix is open in
[#65891](https://github.com/vllm-project/vllm/pull/65891)). Adding `attention_backend="FLASHINFER"` works around it.

## Why layer 0

Qwen2's key projection has a bias. In layer 0 that bias dominates: the norm of the per-channel K mean is 98% of the
norm of K (Llama-3.1-8B: 68%).

Attention only cares about the part of K that differs between tokens. The mean adds the same number to every attention
score of a given query, and softmax cancels it. But FP8's rounding error is relative to the whole value, so it lands on
the small part that actually matters. Think of measuring height differences between people standing on a mountain
with a ruler marked every hundred meters.

A per-tensor scale can keep large values from saturating, but it does not change E4M3's relative precision, so it
cannot remove the rounding error on that small token-dependent part. (All numbers here use vLLM's default scales; I
have not tested calibrated ones.)

And layer 0's errors don't stay in layer 0. Quantizing only layer 0 to FP8 gives 1,300× the output KL of the median
layer (offline, R1-Distill-Qwen-7B, wikitext). In a long chain of thought that compounds into nonsense. Offline,
Qwen2.5-7B's wikitext perplexity goes from 6.31 to 78.2 with FP8 on every layer, and to 6.59 with layer 0 kept.

### "But NVIDIA says it's the last layer"

NVIDIA's [TensorRT-Edge-LLM docs](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/FP8KV.html)
flag Qwen2.5-7B-Instruct for FP8 KV degradation and point at the last layer's very large values. Both observations
hold: the last layer's K has a mean almost as large as layer 0's (mean norm 240 vs 274; middle layers ~12). But large
values are not the same as sensitivity. Quantizing only the last layer gives 8× the median layer's KL, against 1,300×
for layer 0, and keeping only the last layer in BF16 does not recover accuracy.

## Fixes

1. **Skip layer 0** (above). Simple, works today.
2. **Subtract a fixed K center before quantizing.** Shifting every key by the same vector after RoPE changes all
   attention scores of a query by one constant, so the softmax output is unchanged. With a per-channel K mean measured
   on some text, FP8 recovers to 69.5% on R1-Distill-Qwen-7B (offline, fake-quantized FP8 with the prompt in BF16;
   BF16 73.5% in that setup).
3. **Use the K bias itself as that center.** [alibaba/rtp-llm#1583](https://github.com/alibaba/rtp-llm/pull/1583)
   does this for FP8: no calibration, no extra kernel, only for layers with a QKV bias. vLLM doesn't have it yet.

## Credit

The mechanism isn't new. Liu et al., *Quantization Hurts Reasoning?*
([arXiv 2504.04823](https://arxiv.org/abs/2504.04823)), traced the extreme K outlier channels in
R1-Distill-Qwen-1.5B / 7B to the key bias inherited from Qwen2.5, shown for layer 0, for 4-bit KV caches. rtp-llm
fixed the FP8 case in its engine. vLLM's own [FP8 KV write-up](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
(April 2026) covers Qwen3 / Qwen3.5 and the long-context accumulation issue, not the Qwen2 family. What I'm adding is
the FP8 picture in vLLM's default path, the layer-0 vs last-layer measurement, and the one-line check.

Discussion is on [vLLM #41343](https://github.com/vllm-project/vllm/issues/41343). If you've seen this on other
models, I'd like to hear about it.
