---
layout: post
title: "One run is not a result: comparing KV-cache dtypes in vLLM"
date: 2026-11-14
description: "Same BF16 config, same 500 MATH-500 problems, only the order shuffled: 66.6%, 65.0%, 63.8%. Why single greedy runs can't compare KV-cache dtypes, and what I do instead."
---

[中文版](/2026/11/14/one-run-is-not-a-result-zh/)

I was comparing a quantized KV cache against BF16 on MATH-500 in vLLM. Same model, same 500 problems, greedy
decoding. The first run said the quantized cache lost 2.2 points.

Then I reran BF16 twice more, changing nothing but the order of the problems:

| Run | Accuracy (BF16 KV) |
|---|---|
| Original order | 66.6% |
| Shuffle 1 | 65.0% |
| Shuffle 2 | 63.8% |

DeepSeek-R1-Distill-Llama-8B, vLLM 0.33, H200, up to 4,096 new tokens. The baseline alone moves by 2.8 points, and
any two runs disagree on about 47 of the 500 problems. Averaged over three runs each, the quantized cache's gap
shrinks from −2.2 to −0.33 points (paired permutation test, p = 0.83).

This model is on the noisy end: about 30% of its answers hit the 4,096-token cap, and a truncated chain of thought is
easy to tip either way. On Qwen2.5-7B-Instruct, five BF16 runs spanned only 71.0–71.6%. But quantized configurations
move more than BF16: on Qwen2.5-7B-Instruct, one quantized configuration scored 67.0% on 200 problems in one job
and 73.5% in another (different batch composition and GPU node).

## Why

Greedy decoding is not deterministic in a serving engine. Requests are batched differently from run to run, many
kernels are not batch-invariant, and a tiny numeric difference flips one token early in a long chain of thought.
Thinking Machines traced this to
[batch invariance](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Horace He, 2025),
and Yuan et al. ([arXiv 2506.09501](https://arxiv.org/abs/2506.09501), first titled "Give Me FP32 or Give Me
Death?") measured up to 9% accuracy swings for R1-Distill-Qwen-7B under BF16 from batch size, GPU count and GPU type
alone.

## What this means for KV-cache comparisons

A quantized KV cache changes the numbers by a small amount, which is about the size of this noise. A single-run
difference of a few points tells you little.

What I do now:

1. Run each configuration at least three times, shuffling the problem order.
2. Use all problems, not a subset.
3. Compare per problem, paired across configurations, and test with a sign-flip permutation test, not by eyeballing
   averages.
4. Report the baseline's own run-to-run spread next to the gap.

It also cost me a claim: from single runs on 200 problems I had reported a 5-point loss for one configuration
(p = 0.011). Over 500 problems and five runs it became −1.9 points (p = 0.17), and I withdrew it.
