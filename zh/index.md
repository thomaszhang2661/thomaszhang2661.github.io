---
layout: default
title: 张健 · Jian (Thomas) Zhang
description: 机器学习工程师，做大模型推理和智能体系统。
lang: zh-CN
permalink: /zh/
alt_url: /
---
<div class="home" markdown="1">

{% include profile.html name="张健 · Jian (Thomas) Zhang" tagline="机器学习工程师，做大模型推理和智能体系统。" %}

我做大模型推理：KV cache 量化、NVIDIA Hopper 上的注意力内核，以及用 vLLM 做推理服务；也做大模型智能体系统。

2026 年先后在字节跳动（智能体平台）和 TikTok USDS（用大模型智能体处理崩溃问题）实习。在此之前，我在北京一家教育科技公司做了三年机器学习工程师，把手写识别的词准确率从 93% 提高到 99.6%。博士毕业于中国科学院大学（中科院物理所），专业是凝聚态物理，研究强关联电子体系的数值方法。

## 论文

<ul class="pubs">
  <li><strong>Jian Zhang</strong>, Ming-Feng Tian, Guang-Xi Jin, Yuan-Feng Xu, Xi Dai. Implementation of LDA+Gutzwiller with Newton's method. <em>Chinese Physics B</em> 26, 017103 (2017). <a href="https://doi.org/10.1088/1674-1056/26/1/017103">doi</a></li>
</ul>

## 开源

- vLLM [#41343](https://github.com/vllm-project/vllm/issues/41343)：FP8 KV cache 为什么会让 Qwen2 系推理模型悄悄失效（第 0 层的 key 偏置），附测量数据和绕过办法。
- vLLM [#67176](https://github.com/vllm-project/vllm/issues/67176)：UltraQuant 4 位 KV 的缩放常数；从 0.156 改成 0.2，同样的显存下，三个模型的 KL 散度降低 40–56%。
- vLLM [#46461](https://github.com/vllm-project/vllm/pull/46461)：复现证明 TurboQuant 的前缀缓存 bug 同样影响 UltraQuant。

## 经历

<ul class="cv">
  <li><span class="when">2026.05 – 2026.09</span><span class="what"><strong>机器学习工程师（实习）</strong>，TikTok USDS Joint Venture，圣何塞<small>用大模型智能体自动分析和修复崩溃（调试智能体配合代码修复智能体）：4 个版本里修掉 31 个根因，每日崩溃量下降约 90%。</small></span></li>
  <li><span class="when">2026.01 – 2026.04</span><span class="what"><strong>研究科学家（实习）</strong>，字节跳动，圣何塞<small>智能体平台：关键词加向量的混合技能检索（匹配技能用的上下文 token 减少约 95%，技能数超过 1000 时准确率不降），以及让长时间多智能体任务不再卡住的持久化待办插件。</small></span></li>
  <li><span class="when">2021.06 – 2024.08</span><span class="what"><strong>机器学习工程师</strong>，北京天学网教育科技公司<small>手写识别词准确率从 93% 提高到 99.6%，产品推广到 100 多所学校；做出产品的第一个中文手写识别功能。</small></span></li>
  <li><span class="when">2013 – 2018</span><span class="what"><strong>博士研究</strong>，中国科学院物理研究所<small>原有求解器会陷入循环、无法收敛，我改用牛顿法和解析梯度，计算时间减少 90%；第一作者发表于 Chinese Physics B。</small></span></li>
</ul>

## 教育

<ul class="cv">
  <li><span class="when">2024 – 2026</span><span class="what"><strong>计算机科学硕士</strong>，美国东北大学</span></li>
  <li><span class="when">2010 – 2018</span><span class="what"><strong>凝聚态物理博士</strong>，中国科学院大学</span></li>
  <li><span class="when">2006 – 2010</span><span class="what"><strong>物理学学士</strong>，兰州大学</span></li>
</ul>

## 文章

{% include post-list.html limit=5 %}

[全部文章 →](/zh/blog/)

</div>
