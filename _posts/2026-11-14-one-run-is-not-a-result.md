---
layout: post
title: "One run is not a result: comparing KV-cache dtypes in vLLM"
date: 2026-11-14
description: "Same BF16 config, same 500 MATH-500 problems, only the order shuffled: 68.4%, 67.0%, 67.6%. Why single greedy runs can't compare KV-cache dtypes, and what I do instead."
---

[中文版](/2026/11/14/one-run-is-not-a-result-zh/)

I was comparing a quantized KV cache against BF16 on MATH-500 in vLLM. Same model, same 500 problems, greedy
decoding. The first run said the quantized cache lost 1.2 points.

Then I reran BF16 twice more, changing nothing but the order of the problems:

| Run | Accuracy (BF16 KV) |
|---|---|
| Original order | 68.4% |
| Shuffle 1 | 67.0% |
| Shuffle 2 | 67.6% |

DeepSeek-R1-Distill-Llama-8B, vLLM 0.33, H200, up to 4,096 new tokens. The baseline alone moves by 1.4 points, more than the gap, and
any two runs disagree on about 47 of the 500 problems. Averaged over three runs each, the quantized cache's gap
goes from −1.2 to +0.07 points (paired permutation test, p = 1.0).

This model is on the noisy end: about 30% of its answers hit the 4,096-token cap, and a truncated chain of thought is
easy to tip either way. On Qwen2.5-7B-Instruct, five BF16 runs spanned only 71.0–71.6%. But quantized configurations
move more than BF16: on Qwen2.5-7B-Instruct, one quantized configuration scored 67.0% on 200 problems in one job
and 73.5% in another (different batch composition and GPU node).

## Why the same config gives different answers

**1. Greedy decoding should be deterministic.** At each step it picks the most likely token. Same input, same model:
in principle, the same tokens every time.

**2. Floating-point addition depends on order.** A computer keeps a fixed number of digits and rounds after every
addition, so the order of a sum can change the result. With 3 significant digits:

- (1000 + 4) + 4: 1004 rounds to 1000, and 1000 + 4 rounds to 1000 again.
- 1000 + (4 + 4): 1000 + 8 = 1008 rounds to 1010.

Same three numbers, two results. BF16 keeps only about 2–3 significant digits. The FP32 accumulators inside GPU
kernels keep about 7, so the differences there are smaller, but still there.

**3. The order of a sum depends on the batch.** A serving engine computes many requests together. To stay fast, GPU
kernels choose their strategy by batch size: with a small batch, one long sum (over the hidden dimension in a
matmul, over the context in attention) is split into chunks that run in parallel and are then combined; with a large
batch, the work is tiled differently. Different splits, different order of additions. So the same request, batched
with different other requests, gets slightly different numbers, say in the fourth or fifth decimal of a probability.
A kernel is *batch-invariant* if a request's result does not depend on what else is in its batch. Most kernels in
today's engines are not, and Thinking Machines identify this as the main cause.

**4. Shuffling the problems changes the batches.** I changed only the order of the problems. But vLLM batches
continuously: as requests finish, new ones join. A different order means each problem shares its steps with
different problems, in batches of different sizes.

**5. A tiny difference becomes a different answer.** Usually the top token leads by a wide margin, and a difference
in the fifth decimal changes nothing. But now and then two tokens are almost tied (0.4001 vs 0.4000), and the tiny
difference flips the choice. Every later token depends on all the earlier ones, so one flipped token sends the rest
of the generation down another path. A reasoning model writes thousands of tokens, so a near-tie somewhere is
likely; the chain of thought forks, and the final answer may turn right or wrong.

That is how one config scores 68.4%, 67.0% and 67.6%.

**Prior work.** [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Horace He, 2025): the root cause is kernels that are not batch-invariant; with batch-invariant
kernels they got fully deterministic output from vLLM, at some cost in speed. Yuan et al. ([arXiv 2506.09501](https://arxiv.org/abs/2506.09501), first titled
"Give Me FP32 or Give Me Death?"): BF16's few digits make such flips likely, and computing in FP32 makes the
differences small enough that the choice almost never flips. They measured up to 9% accuracy swings for
R1-Distill-Qwen-7B under BF16 from batch size, GPU count and GPU type alone.

## What this means for KV-cache comparisons

A quantized KV cache also changes the numbers slightly, by about as much as this noise, and so triggers the same
forks. If one run shows a 2–3 point gap, you cannot tell whether the quantization caused it or the run happened to
fork differently.

What I do now:

1. Run each configuration at least three times, shuffling the problem order.
2. Use all problems, not a subset.
3. Compare per problem, paired across configurations, and test with a sign-flip permutation test, not by eyeballing
   averages.
4. Report the baseline's own run-to-run spread next to the gap.

It also cost me a claim: from single runs on 200 problems I had reported a 5-point loss for one configuration
(p = 0.011). Over 500 problems and five runs it became −1.9 points (p = 0.17), and I withdrew it.

---

*Correction (Nov 15, 2026): The first version of this post used R1-Distill-Llama-8B numbers measured with a
tokenizer-loading bug ([transformers #45488](https://github.com/huggingface/transformers/issues/45488)) that removed
spaces and newlines from the prompts in vLLM. The table and the numbers in the text now come from reruns with the
correct tokenizer. The BF16 runs now spread over 1.4 points instead of 2.8, while any two runs still disagree on about
47 of the 500 problems, and the quantized cache's single-run gap of −1.2 averages to +0.07 over three runs. The
conclusion is unchanged. The Qwen2.5 numbers were not affected.*
