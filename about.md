---
layout: page
title: About
permalink: /about/
---

I work on LLM inference: KV-cache quantization, attention kernels on NVIDIA Hopper, and serving with vLLM.

- GitHub: [thomaszhang2661](https://github.com/thomaszhang2661)

## Open-source contributions

- vLLM [#67176](https://github.com/vllm-project/vllm/issues/67176): UltraQuant's 4-bit KV scale constant (0.156 → 0.2
  lowers KL by 40–56% on three models at the same bytes).
- vLLM [#41343](https://github.com/vllm-project/vllm/issues/41343): why FP8 KV silently breaks Qwen2-family models (layer 0),
  with measurements and workarounds.
- vLLM [#46461](https://github.com/vllm-project/vllm/pull/46461): a clean reproduction showing the prefix-cache bug in
  TurboQuant also affects UltraQuant.

## Writing

- [FP8 KV cache can silently break small Qwen2-based reasoning models in vLLM](/2026/11/13/fp8-kv-cache-qwen2-vllm/)
  ([中文](/2026/11/13/fp8-kv-cache-qwen2-vllm-zh/))
- [One run is not a result: comparing KV-cache dtypes in vLLM](/2026/11/14/one-run-is-not-a-result/)
  ([中文](/2026/11/14/one-run-is-not-a-result-zh/))
