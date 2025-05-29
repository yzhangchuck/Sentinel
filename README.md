# Sentinel: Attention Probing of Proxy Models for LLM Context Compression

> Official repository for the paper:  
> **"Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective"**  
> 📄 [arXiv link](https://arxiv.org/abs/XXXX.XXXXX) — *coming soon or replace with real link*

---

## 📌 Overview

**Sentinel** is a lightweight and interpretable sentence-level context compression framework for Large Language Models (LLMs). Instead of relying on supervised compression models or full-scale generation, Sentinel probes decoder attention from a small off-the-shelf proxy model to identify query-relevant context.

- ✅ No finetuning required  
- ✅ Sentence-level compression  
- ✅ Compatible with any LLM backbone  
- ✅ Model-agnostic and fast

> "We find that query-context understanding is stable across model scales, enabling small models to emulate the attention behaviors of large LLMs."

---

## 🔍 Paper Abstract

> Retrieval-augmented generation (RAG) enhances LLMs with external context, but retrieved passages are often lengthy, noisy, or exceed input limits.  
> We propose **Sentinel**, a lightweight compression method that probes decoder attention from a 0.5B proxy LLM using a probing classifier.  
> On LongBench, Sentinel achieves up to 5× compression while matching the QA performance of 7B-scale systems.

---

## 🚧 Project Status

This repository is currently a placeholder for the paper.  
Code and data will be released soon. Stay tuned!

---

## 📎 Citation

If you find our work helpful, please cite:

```bibtex
@article{zhang2025sentinel,
  title={Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective},
  author={Zhang, Yong and Huang, Yanwen and Cheng, Ning and Guo, Yang and Zhu, Yun and Wang, Yanmeng and Wang, Shaojun and Xiao, Jing},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
