---
layout: default
title: Jian (Thomas) Zhang
description: ML engineer working on LLM inference and agent systems.
alt_url: /zh/
redirect_from:
  - /about/
---
<div class="home" markdown="1">

{% include profile.html name="Jian (Thomas) Zhang" tagline="ML engineer working on LLM inference and agent systems." %}

I work on LLM inference: KV-cache quantization, attention kernels on NVIDIA Hopper, and serving with vLLM. I also build LLM agent systems.

In 2026 I interned at ByteDance on an agent platform and at TikTok USDS on LLM agents for crash triage. Before that I spent three years as a machine learning engineer at an education-technology company in Beijing, where I took handwriting recognition from 93% to 99.6% word accuracy. I hold a Ph.D. in condensed matter physics from the University of Chinese Academy of Sciences (Institute of Physics), where I worked on numerical methods for strongly correlated electrons.

## Publications

<ul class="pubs">
  <li><strong>Jian Zhang</strong>, Ming-Feng Tian, Guang-Xi Jin, Yuan-Feng Xu, Xi Dai. Implementation of LDA+Gutzwiller with Newton's method. <em>Chinese Physics B</em> 26, 017103 (2017). <a href="https://doi.org/10.1088/1674-1056/26/1/017103">doi</a></li>
</ul>

## Open source

- vLLM [#41343](https://github.com/vllm-project/vllm/issues/41343): why the FP8 KV cache silently breaks Qwen2-family reasoning models (layer 0's key bias), with measurements and workarounds.
- vLLM [#67176](https://github.com/vllm-project/vllm/issues/67176): UltraQuant's 4-bit KV scale constant; changing 0.156 to 0.2 lowers KL divergence by 40–56% on three models at the same memory.
- vLLM [#46461](https://github.com/vllm-project/vllm/pull/46461): a reproduction showing that TurboQuant's prefix-cache bug also affects UltraQuant.

## Experience

<ul class="cv">
  <li><span class="when">2026.05 – 2026.09</span><span class="what"><strong>Machine Learning Engineer Intern</strong>, TikTok USDS Joint Venture, San Jose<small>An LLM-agent pipeline for crash triage and repair (a debugging agent paired with a code-fix agent): 31 root causes fixed across 4 releases, daily crashes down about 90%.</small></span></li>
  <li><span class="when">2026.01 – 2026.04</span><span class="what"><strong>Research Scientist Intern</strong>, ByteDance, San Jose<small>Agent platform: hybrid keyword + vector skill retrieval (about 95% fewer context tokens, accuracy held past 1,000 skills), and a persistent to-do plugin for long multi-agent tasks.</small></span></li>
  <li><span class="when">2021.06 – 2024.08</span><span class="what"><strong>Machine Learning Engineer</strong>, Beijing TianXuewang Education Technology<small>Handwriting recognition from 93% to 99.6% word accuracy, used in more than 100 schools; the product's first Chinese handwriting recognition.</small></span></li>
  <li><span class="when">2013 – 2018</span><span class="what"><strong>Ph.D. researcher</strong>, Institute of Physics, Chinese Academy of Sciences<small>Replaced a solver that cycled without converging with Newton's method and an analytic gradient (90% less compute time); published in Chinese Physics B.</small></span></li>
</ul>

## Education

<ul class="cv">
  <li><span class="when">2024 – 2026</span><span class="what"><strong>M.S., Computer Science</strong>, Northeastern University</span></li>
  <li><span class="when">2010 – 2018</span><span class="what"><strong>Ph.D., Condensed Matter Physics</strong>, University of Chinese Academy of Sciences</span></li>
  <li><span class="when">2006 – 2010</span><span class="what"><strong>B.S., Physics</strong>, Lanzhou University</span></li>
</ul>

## Writing

{% include post-list.html limit=5 %}

[All posts →](/blog/)

</div>
