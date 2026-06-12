# Probe 注意力压缩 — 优化概览与运行指南

本文档描述当前 **推荐的最优推理路径**、性能数据、代码结构，以及在本地 / AutoDL 上的完整运行方式。

---

## 1. 最优方案摘要

| 项目 | 推荐配置 | 说明 |
|------|----------|------|
| 主 attention | **SDPA**（`probe_attn_implementation="sdpa"`） | 不 materialize 全 `[S×S]` 矩阵 |
  | Probe 特征 | **Last-row QK + context renorm + sentence mean** | 24 层流式累加，供 LR detector 打分 |
| Forward 次数 | **1 次**（`max_seq_len=None` → 模型上限 32768） | 不再 1024 chunk 循环 |
| Segment 融合 | **Torch**（默认） | cuBLAS matmul + `sent_masks @ ctx` |
| Triton probe | **关**（`use_triton_probe=False`） | block GEMM 版仍比 Torch 慢 ~25% |
| torch.compile | **关** | 实测更慢 |
| flash-attn | **暂不启用** | 需单独编译安装；装好后可试 `flash_attention_2` |
| LongBench 吞吐 | **`compress_batch`，batch_size=4** | 单条延迟不变，吞吐 ~16 samples/s |

**核心代码入口：**

```python
from llmlingua.probe.defaults import create_probe_compressor

compressor = create_probe_compressor(
    detector_path="models/detectors/qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl",
)
result = compressor.compress(context=..., question=..., compression_rate=0.8, context_type="chinese")
```

---

## 2. 性能参考（A800 80GB，Qwen2.5-0.5B）

测试文本：`demo_attention_compression.py` 中的 **姚明维基长文**（~4716 context tokens，~4799 prompt tokens）。

| 阶段 | 原始方案 | 当前最优 |
|------|----------|----------|
| compress 全流程 | ~0.47s（1024×5 chunk + eager） | **~0.061s**（demo 长文） |
| 纯 model forward（含 probe） | — | **~0.052s** |
| 纯 model forward（无 probe） | — | **~0.050s** |
| probe + 前后处理 | — | **~0.009–0.010s** |
| multifieldqa_zh compress @10240 | — | **~0.067s**（200 条均值，tokenizer 优化后） |
| 相对加速 | 1× | **~7.7×** |

**批量吞吐（相同长文 ×4）：**

| 模式 | 摊销延迟 | 吞吐 |
|------|----------|------|
| 单条 `compress()` | ~0.061s/sample | ~16 samples/s |
| `compress_batch(4)` | ~0.063s/sample | ~16 samples/s |

数值对齐：Probe 路径 vs Legacy eager，sentence feature max diff ≈ **2.4e-4**。

### 2.0 Tokenizer 计数策略（2026-05-31 优化）

| 用途 | Tokenizer | 说明 |
|------|-----------|------|
| Forward / `max_seq_len` 是否 chunk | **0.5B** `self.tokenizer` | 与 attention 模型输入一致 |
| 每句 `target_token` 预算 | **7B** `eval_tokenizer` | 与 LongBench 下游 7B 评测对齐 |
| `compressed_length` | **句级 7B 求和** | `sum(sentence_tokens[i] for i in preserved)`，不再对拼接全文再 encode |
| 已删除 | 开头 `eval_tokenizer.encode(context)` | 与 `sum(sentence_tokens)` 重复 |

`chunk_text(..., use_eval_tokenizer=False)` 为默认；仅当 chunk 边界必须按 7B 刻度时传 `True`。

**句级 token 预算与并行（2026-05-31）**

| 参数 | 说明 |
|------|------|
| `sentence_budget_tokenizer` | `"7b"`（默认，对齐 LongBench 评测）或 `"0.5b"`（打分后选句用 0.5B 句长近似） |
| `sentence_tokenize_workers` | `>0` 时多线程逐句 `encode`；实测 **batch 快于 parallel**（Rust fast tokenizer 已批量） |

分段计时（`scripts/benchmark/profile_prep_breakdown.py`，`multifieldqa_zh` n=50，10240）：

| 阶段 | 均值 | 占 est. pipeline |
|------|------|------------------|
| 0.5B prompt tokenize | **~11.7 ms** | 最大 prep 项（forward 必需） |
| split + align | ~2.4 ms | |
| **7B 句级 batch** | **~1.9 ms** | **~3%** |
| 7B 句级 parallel×8 | ~11.2 ms | 更慢，不推荐 |
| 0.5B 句级 batch | ~2.0 ms | 近似预算 |
| forward + probe | ~48.8 ms | **~74%** |
| detector + select | **~1.0 ms** | 算法「打完分+选句」 |

结论：**打分结束在 detector+选句（~1ms）**；与 compress 算法强相关的 GPU 主体是 **forward+probe**。7B 句级 tokenize 在总耗时里占比很小；改用 `0.5b` 预算主要省 **~0ms 级**（句级），不能省 0.5B prompt tokenize。论文里可把 7B 句级计数单列为例行对齐下游，而非 compress 核心算子。

**全流程分段（`scripts/benchmark/profile_full_pipeline.py`，multifieldqa_zh n=50，10240）**

| 阶段 | 均值 | 说明 |
|------|------|------|
| 0.5B encode **offset=False** | **9.0 ms** | 仅需 `input_ids` 的基线 |
| 0.5B tokenize **offset=True** | **9.5 ms** | probe prep 实际调用 |
| **offset 增量 (True−False)** | **~1.0 ms** | 仅占 prep ~7% |
| find_context + split/align | ~3.0 ms | CPU |
| 7B 句级 batch | ~1.9 ms | |
| **prep_full** | **~14.4 ms** | A：prefill/compress 共有 |
| H2D | ~0.1 ms | B |
| **forward+probe** | **~48.1 ms** | C |
| **prefill E2E (A+B+C)** | **~62.6 ms** | 真实 prefill 墙钟 |
| prefill forward-only (C) | ~48.1 ms | 旧 benchmark 口径 |
| detector + select | **~0.9 ms** | D+E，仅 compress |
| **compress E2E (A–E)** | **~63.4 ms** | 分段求和 |
| compress−prefill E2E | **~0.9 ms** | 仅 D+E |
| 旧口径 compress−prefill | ~22 ms | compress 计 prep、prefill 不计 |

复现：`PYTHONPATH=. python scripts/benchmark/profile_full_pipeline.py --target_task multifieldqa_zh --max_samples 50`

### 2.0.1 Compress vs Prefill：计时层级（n=200，@10240，2026-05-31 重跑）

> **易混淆**：§9.1 的 **Prefill 59 ms** ≠「纯模型 prefill」。它是 `tokenize(offset=False)` **8.6 ms** + **forward+probe** **50.5 ms**。**纯 GPU self-attention（无 probe）≈ 46 ms**，需单独计时。

来源：`scripts/benchmark/profile_compress_10240_breakdown.py`（每条样本额外跑一次 plain forward 用于对照；**compress() 墙钟内只跑一次 forward+probe**）  
结果：`experiments/llmlingua2/evaluation/results/compress_10240_breakdown_n200.json`

| 层级 | 阶段 | mean (ms) | 占 compress 墙钟 (~72 ms) | 说明 |
|------|------|-----------|---------------------------|------|
| **L0** | **GPU plain SDPA forward** | **45.8** | **63%** | 0.5B 模型内部 self-attention，**probe 关闭**；即「纯 prefill」 |
| L1 | + probe side-channel + finalize | **+5.3** | +7% | last-row QK、句级累加、`finalize_vectors` |
| **L1 合计** | **forward + probe** | **51.1** | **71%** | compress / §9.1 prefill 的 GPU 主体 |
| L2 | CPU prep（offset tok + 分句 + 7B 句级计数） | **15.2** | 21% | 仅 compress 全计；§9.1 prefill 大多未单独计入 |
| L3 | detector + 选句拼串 | **7.9** | 11% | 仅 compress |
| **墙钟** | **`compress()`** | **72.3** | 100% | 权威端到端 |

```
compress 墙钟 72.3 ms
├─ GPU plain attention (L0)     45.8 ms  ████████████████  ← 纯 prefill
├─ probe 增量 (L1−L0)            5.3 ms  ██
├─ CPU prep (L2)                15.2 ms  █████
└─ detector + 选句 (L3)          7.9 ms  ███
```

**三种「prefill」口径对照（同数据 n=200，prompt mean 4485 tok）**

| 名称 | ms | 含什么 |
|------|-----|--------|
| **纯 GPU attention（L0）** | **45.8** | 仅 `attention_model` SDPA forward，无 probe |
| **GPU forward+probe（L1）** | **51.1** | 打分用 forward；§9.1 `prefill_forward_s` ≈ **50.5** |
| **§9.1 prefill 基准** | **59.0** | L1 + `tokenize(offset=False)` **8.6**（prep 其余未计） |
| **compress 墙钟** | **72.3** | L1 + 全量 prep + detector + 选句 |

**Compress 对比各 prefill 口径的 Δ**

| 对比 | Δ (ms) | 多在哪里 |
|------|--------|----------|
| compress − **纯 GPU attention (45.8)** | **+26.5** | probe **5.3** + prep **15.2** + 后处理 **7.9** |
| compress − **forward+probe (51.1)** | **+21.1** | prep **15.2** + 后处理 **7.9**（GPU 只多跑一次 probe，已在 L1） |
| compress − **§9.1 prefill (59.0)** | **+13.3** | prep 口径差 **~6.6** + 后处理 **7.9**（§9.1 未计全 prep/选句） |

复现：

```bash
PYTHONPATH=. python scripts/benchmark/profile_compress_10240_breakdown.py \
  --target_task multifieldqa_zh --max_samples 200 --max_seq_len 10240 --gpu 0 \
  --save_path experiments/llmlingua2/evaluation/results/compress_10240_breakdown_n200.json
```

### 2.1 显存占用（A800 80GB，同上长文，4799 prompt tokens）

| 模式 | Peak 显存 | vs plain forward | 说明 |
|------|-----------|------------------|------|
| 模型权重（加载后） | **~958 MB** | — | Qwen2.5-0.5B fp16 |
| Plain SDPA forward（无 probe） | **~2415 MB** | — | 单次 4799 token prefill |
| SDPA + probe side-channel | **~2417 MB** | **+2.3 MB** | last-row QK + sentence 累加 |
| probe+SDPA compress（全流程） | **~2417 MB** | **+2.3 MB** | 前后处理不显著增 peak |
| Legacy eager（1024×5 chunk） | **~1941 MB** | -474 MB* | 每 chunk 仅 1024 token，peak 更低 |

\* Legacy peak 更低是因为 **按 1024 分块**，每层不 materialize 全 `[4799×4799]`。若 legacy 对全长做 `output_attentions=True`，每层需存 `[1,14,4799,4799]` fp16 ≈ **644 MB/层**，24 层仅 attention 就 **~15 GB** 量级（不含激活），且极易 OOM。

**结论：**
- Probe 优化 **几乎不增加 peak 显存**（+2.3 MB），主要收益是去掉 `[S×S]` materialization
- 最优路径用 **单次全长 forward**，peak 高于 chunk legacy 是正常现象（用算力换单次延迟）
- 在 80GB 卡上 ~5k tokens 余量充足（peak ~2.4 GB + 权重 ~1 GB）

复现命令：

```bash
PYTHONPATH=. python scripts/benchmark/probe_memory.py
```

### 2.2 HF Prefill-only（0.5B vs 7B，n=200）

与 §2 / §9.1 的 **Sentinel 0.5B probe prefill** 对照。本节为 **HuggingFace transformers 纯 prefill**（第一次 `model(input_ids, use_cache=True)` forward，**无 decode**）。

**测试配置（2026-05-31 重跑，A800 80GB）**

| 项目 | 值 |
|------|-----|
| 数据 | `multifieldqa_zh` **$n{=}200$**（`load_key=prompt`，`max_seq_len=10240`） |
| Prompt 模板 | `probe_filtering`（`Given the following information: … Answer:`） |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`、`Qwen/Qwen2.5-7B-Instruct` |
| 实现 | `scripts/benchmark/benchmark_autoregressive_decode.py --prefill-only` |
| Shell | `scripts/benchmark/run/run_hf_prefill_bench.sh` |
| VRAM | `torch.cuda.max_memory_allocated()`（每条样本前 `reset_peak_memory_stats()`） |
| 结果 JSON | `experiments/llmlingua2/evaluation/results/hf_prefill_mfqa_zh_seq10240_n200_20260531.json` |

**汇总（mean ± stdev）**

| 指标 | **0.5B-Instruct** | **7B-Instruct** | 7B / 0.5B |
|------|-------------------|-----------------|-----------|
| 模型权重（加载后） | **950 MB** | **14,577 MB** | 15.3× |
| Prompt tokens（mean） | **4485** | **4485** | 1× |
| **Prefill** | **54.3 ± 21.6 ms** | **397.8 ± 204.3 ms** | **7.3×** |
| **Peak VRAM**（PyTorch） | **2321 MB** | **16,157 MB** | **7.0×** |
| Peak VRAM（nvidia-smi） | **3518 MB** | **18,257 MB** | 5.2× |
| nvidia after load | 1417 MB | 15,215 MB | — |

**与 Sentinel probe prefill（§9.1，同数据 n=200）对照**

| 阶段 | 0.5B @10240 | 说明 |
|------|-------------|------|
| HF plain prefill（本节） | **54.3 ms** | 无 probe；GPU forward only |
| Sentinel probe prefill（§9.1） | **59.0 ms** | 含 8.6 ms tokenize + forward+probe |
| Sentinel compress（§9.1） | **66.7 ms** | 全流程墙钟 |
| HF 7B prefill（本节） | **397.8 ms** | 下游模型，同 prompt 长度 |

复现（AutoDL）：

```bash
chmod +x scripts/benchmark/run/run_hf_prefill_bench.sh
MAX_SAMPLES=200 MAX_SEQ_LEN=10240 bash scripts/benchmark/run/run_hf_prefill_bench.sh
```

### 2.2b 自回归 Decode（压力测试 + MFQA 实测）

下表 decode 数字来自 **prefill + greedy decode**；**LongBench MFQA 实际 `max_gen=64`**，见「实测校验」。

> **易误解**：32 s 是「生成满 **2000 token**」的耗时，不是 MFQA 每条答案的耗时。按 ~62 tok/s 线性外推，**64 token 仅 ~1 s**。

| 指标 | **0.5B-Instruct** | **7B-Instruct** | 7B / 0.5B |
|------|-------------------|-----------------|-----------|
| **Prefill**（§2.2 重跑） | **54.3 ms** | **397.8 ms** | **7.3×** |
| **Decode 64 tok**（MFQA `max_gen`） | **1005 ms** | **1297 ms** | **1.29×** |
| Decode 吞吐 @64 | **63.7 tok/s** | **49.7 tok/s** | 0.78× |
| **Peak VRAM**（prefill+decode@64） | **2329 MB** | **16,188 MB** | **7.0×** |
| Decode 2000 tok（n=10 压力测试） | **32,159 ms** | **40,274 ms** | 1.25× |

| 阶段 | 0.5B @10240 | 说明 |
|------|-------------|------|
| HF prefill（§2.2） | **54.3 ms** | n=200，纯 forward |
| Decode 2000 tok | **32,159 ms** | n=10 压力测试；约为 prefill 的 **~590×** |

**实测校验（2026-06-10 复核，A800，HF transformers 逐步 decode）**

| 检查项 | 结果 |
|--------|------|
| 实现 bug？ | **无**：`model.generate()` 与手动 `use_cache` 循环一致（2000 tok ≈ 33–35 s） |
| prompt 长度影响？ | **几乎无**：prompt 128 vs 2804 tok，decode 2000 tok 均为 **~33 s**（每步 ~16.7 ms） |
| 线性缩放？ | **是**：0.5B @prompt≈2404 — 32 tok **531 ms**，64 tok **1.04 s**，512 tok **8.2 s**，2000 tok **31.5 s**（≈62 tok/s） |
| MFQA 真实生成长度？ | `eval_longbench.py` 中 `multifieldqa_zh` **`max_gen=64`**（短答案，非 2000） |

| 场景 | 0.5B decode（HF greedy） | 说明 |
|------|--------------------------|------|
| **MFQA 真实（64 tok）** | **~1.0 s** | 与评测配置一致 |
| 压力测试（2000 tok） | **~32 s** | 上表主数字；用于对比长生成上限 |

要点：

- **32 s 不夸张但容易误读**：HF 逐步 decode 在 A800 上 0.5B 约 **62 tok/s**；2000 步自然 ≈ 32 s。评测流水线实际只生成 **≤64 tok（~1 s）**。
- **Compress / prefill（~50–60 ms）仍远快于 eval decode（~1 s）**；只有人为拉满 2000 tok 时 decode 才主导墙钟。
- 7B prefill 比 0.5B 慢 **~7×**，但 decode 2000 tok 仅慢 **~1.25×**（KV-cache 逐步解码，接近 memory-bound）。
- 精度评测走 **vLLM 7B**（`evaluate.sh`）；见 §2.3。

复现 HF decode@64（AutoDL）：

```bash
chmod +x scripts/benchmark/run/run_autoregressive_decode_bench.sh
MAX_SAMPLES=200 DECODE_TOKENS=64 MAX_SEQ_LEN=10240 \
  bash scripts/benchmark/run/run_autoregressive_decode_bench.sh
```

Decode 结果 JSON：`autoregressive_decode_64tok_mfqa_zh_seq10240.json`；2000 tok 压力测试：`autoregressive_decode_2000tok_mfqa_zh.json`（n=10）。

### 2.3 下游 7B：HF vs vLLM 同条件对比（评测路径）

**2026-06-10 云端实测**（AutoDL A800 80GB，`scripts/benchmark/run/run_downstream_7b_compare.sh`）

#### 测量协议（两者一致）

| 项目 | 值 |
|------|-----|
| **Batch** | **1（逐条顺序跑）** — HF 每条独立 `forward+decode` 循环；vLLM 每条 `llm.generate([text])` |
| 数据 | `multifieldqa_zh` **$n{=}200$**，`compressed_prompt`（@10240 Sentinel，`N_target=2000`） |
| Prompt tok (mean) | **1981**（两边相同） |
| Prompt 模板 | chat + MFQA-zh（与 `eval_longbench.py --chat_completion` 一致） |
| Decode | `max_tokens=64`，greedy（HF `use_cache`；vLLM `temperature=0`） |
| vLLM 配置 | `gpu_memory_utilization=0.9`，`enforce_eager`（对齐 `eval_multifieldqa_probe_vllm.sh`） |

> **注意**：`eval_longbench.py` 默认 `batch_size=16` 异步并发；本表为 **单条延迟** 基准，便于与 Sentinel compress 墙钟对齐。吞吐评测需另报 batch 模式。

#### 同条件结果（compressed + eval 模板）

| 指标 | **HF transformers** | **vLLM in-process** | vLLM / HF |
|------|----------------------|----------------------|-----------|
| Prefill (ms) | **178.5 ± 17.9** | —（含在 wall） | — |
| Decode@64 (ms) | **1237.4 ± 19.4** | —（含在 wall） | — |
| **E2E wall (ms)** | **1416 ± 27** | **346 ± 246** (median 246) | **~4.1× 加速** |
| Decode 吞吐 | 51.7 tok/s | — | — |
| **PyTorch peak (MB)** | **15,294** | — | — |
| **nvidia-smi peak (MB)** | **16,257** | **73,737** | **~4.5×** |
| nvidia after load | 15,123 | 73,733 | — |
| 实际 out tok (mean) | 64（强制满步） | **~17**（EOS 早停） | — |

结果 JSON：
- HF：`experiments/llmlingua2/evaluation/results/hf7b_decode_64tok_eval_compressed_n200.json`
- vLLM：`experiments/llmlingua2/evaluation/results/vllm7b_decode_64tok_eval_compressed_n200.json`

要点：

- **两边都是一条一条跑**（batch=1），可比的是 per-sample 墙钟，不是 vLLM 连续 batch 吞吐。
- **同 prompt、同模板** 下 vLLM E2E **~0.35 s**，HF **~1.42 s**；Sentinel compress（§9.1 @10240 **~75 ms**）仍远快于任一下游 decode。
- vLLM 显存高是因为 `gpu_memory_utilization=0.9` **预分配 KV pool**（nvidia-smi ~74 GB）；HF PyTorch 只计实际 tensor（~15 GB）。对比显存应优先看 **nvidia-smi 同口径**：HF ~16 GB vs vLLM ~74 GB。
- HF 强制 decode 满 64 tok；vLLM 多数样本 EOS 早停（mean ~17 tok），故 vLLM 墙钟优势略被放大。

#### 历史参考（不同条件，勿直接对比）

| 后端 | Prompt | 模板 | E2E (ms) | 显存 |
|------|--------|------|----------|------|
| HF 7B | raw @10240 | probe_filtering | 1701 (404+1297) | PyTorch 16.2 GB |
| vLLM 7B | raw longbench | eval_mfqa_zh | 436 ± 319 | nvidia 76.6 GB |

#### 复现（AutoDL）

```bash
chmod +x scripts/benchmark/run/run_downstream_7b_compare.sh
MAX_SAMPLES=200 DECODE_TOKENS=64 \
  COMPRESSED_JSON=results/multifieldqa_zh_probe_chunk10240_target2000.json \
  bash scripts/benchmark/run/run_downstream_7b_compare.sh
```

---

## 3. 架构简述

```
compress()
  └─ _detector_based_filtering_impl()
       ├─ 0.5B tokenize + 分句 + 7B 句级 token 计数（可缓存）
       ├─ attention_model(...)          ← SDPA 主路径（正常 self-attention）
       │    └─ 每层 Qwen2Attention patch:
       │         fused_probe_layer()    ← side-channel：last-row QK → sentence 特征
       ├─ ProbeState.finalize_vectors() ← [num_sents, 24×14]
       └─ torch LR detector → 选句 → 压缩文本
```

- **SDPA 主路径**：负责模型正常 forward，内部 fused kernel，不输出 `[S×S]`。
- **Probe side-channel**：只算 last query row 对全序列的 attention，slice context 并重归一化，再按句聚合。
- **Legacy 路径**（`use_probe_attention=False`）：`output_attentions=True` + chunk，仅作对比/debug。

---

## 4. 环境准备

### 4.1 Conda 环境（AutoDL / 本地 GPU）

```bash
# 激活已有环境（AutoDL 上通常为 sentinel）
source /root/miniconda3/etc/profile.d/conda.sh   # Linux AutoDL
conda activate sentinel

# 若从零创建（参考）
# conda create -n sentinel python=3.10 -y
# conda activate sentinel
# pip install torch transformers tiktoken joblib nltk spacy tqdm
# python -m spacy download zh_core_web_sm   # 可选；默认用 fast chinese split 可不装
```

**已验证版本（AutoDL）：** Python 3.10，PyTorch 2.6.0+cu124，CUDA 12.4，A800 80GB。

### 4.2 HuggingFace 环境变量

国内 / AutoDL 建议：

```bash
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME='/root/autodl-tmp/.cache/huggingface'   # AutoDL 大磁盘
# 本地可改为：
# export HF_HOME="$HOME/.cache/huggingface"
```

可选：

```bash
export CUDA_VISIBLE_DEVICES=0          # 指定 GPU
export TRANSFORMERS_OFFLINE=0          # 需要拉模型时保持 0
```

### 4.3 项目路径与 PYTHONPATH

```bash
cd /root/autodl-tmp/LLMLingua          # AutoDL 项目根目录
export PYTHONPATH=.                    # 必须，否则找不到 llmlingua / attention_compressor_clean
```

### 4.4 模型与 Detector

| 资源 | 路径 / 名称 |
|------|-------------|
| Attention 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（HF 自动下载） |
| Eval tokenizer | `Qwen/Qwen2.5-7B-Instruct` |
| Detector | `models/detectors/qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl` |

确认 detector 存在：

```bash
ls models/detectors/qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl
```

---

## 5. 运行方式

### 5.1 一键脚本（推荐）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sentinel
cd /root/autodl-tmp/LLMLingua
export PYTHONPATH=.
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME='/root/autodl-tmp/.cache/huggingface'

chmod +x scripts/run_probe.sh

# 基准测试（长文 ~4700 tokens，5 次 + batch×4 吞吐）
./scripts/run_probe.sh bench

# 单条压缩 demo（默认 demo 长文，输出压缩结果）
./scripts/run_probe.sh demo

# LongBench 批量压缩（中文任务示例，对齐 compress.sh 的 target_token=2000）
./scripts/run_probe.sh longbench \
  experiments/llmlingua2/evaluation/data/LongBench/longbench_formatted.json \
  results/probe_multifieldqa_zh_2000tok.json \
  models/detectors/xxx.pkl \
  --target_task multifieldqa_zh \
  --target_token 2000 \
  --gpu 0
```

环境变量可覆盖默认值：

```bash
export DETECTOR=models/detectors/xxx.pkl
export BATCH_SIZE=8
export ATTENTION_MODEL=Qwen/Qwen2.5-0.5B-Instruct
./scripts/run_probe.sh bench
```

### 5.2 Python 统一 CLI

```bash
PYTHONPATH=. python scripts/probe_compress.py bench --runs 5 --batch-size 4

PYTHONPATH=. python scripts/probe_compress.py run \
  --context "你的文本..." \
  --question "问题？" \
  --context-type chinese \
  --compression-rate 0.8 \
  --show-text

# 不传 --context → 自动使用 demo 长文（姚明维基 ~4700 tokens）
PYTHONPATH=. python scripts/probe_compress.py run --show-text

PYTHONPATH=. python scripts/probe_compress.py longbench \
  --load_origin_from experiments/llmlingua2/evaluation/data/LongBench/longbench_formatted.json \
  --save_path results/longbench_probe.json \
  --batch_size 4 \
  --target_token 2000 \
  --target_task multifieldqa_zh \
  --gpu 0
```

**LongBench 中文精度测试（probe 路径，最小改动）：**

- 已支持：`multifieldqa_zh`、`dureader`、`vcsum`、`lsht`、`passage_retrieval_zh`（自动 `context_type=chinese`）
- 已对齐：`--target_token 2000`、`--target_task`、`compressed_prompt` 存为 **单元素 list**（与 `compress.py` + `--use_attention_filter` 一致）
- 评估：压缩完成后用现有 `evaluate.sh`，`--load_prompt_from <save_path>`、`--load_key compressed_prompt`
- **与 `compress.sh` 仍可能不一致**（精度对比需注意）：无 `--chunk` / `--strict_sentence_chunks` / `--force_tokens`；prompt 为 plain 模板而非 `use_chat_template`；单次全长 forward 而非分 chunk。若要做严格 apples-to-apples，需后续在 `compress.py` 接 probe 或把上述选项补进 probe 路径。

### 5.3 兼容旧入口

```bash
# 等价于 probe_compress bench
PYTHONPATH=. python time_probe_inference.py --runs 5

# 等价于 probe_compress longbench
PYTHONPATH=. python experiments/llmlingua2/evaluation/compress_probe_batch.py \
  --load_origin_from ... --save_path ... --detector_path models/detectors/xxx.pkl
```

### 5.4 原始 Demo

```bash
PYTHONPATH=. python demo_attention_compression.py
```

`demo_attention_compression.py` 仍保留多 scheme 对比；日常 benchmark 建议用 `scripts/probe_compress.py`。

---

## 6. Benchmark 测试文本

单条 latency 测试默认加载 **demo 长文**（与历史 benchmark 一致）：

| 字段 | 来源 |
|------|------|
| context | `demo_attention_compression.py` → `chinese_context`（维基姚明长文） |
| question | `"姚明受过几次伤"` |
| context_type | `"chinese"` |

代码：`llmlingua.probe.defaults.load_bench_text()`

预期输出规模：**~4716 context tokens**，compress **~0.06s**（A800）。

---

## 7. 代码结构

**开源精简版**（姚明 demo，无 benchmark）：[`opensource/`](../opensource/) — `attention_compressor.py` + `demo_attention_compression.py` + `README.md`

```
llmlingua/probe/
  defaults.py           # OPTIMAL_PROBE_KWARGS, create_probe_compressor(), load_bench_text()
  state.py              # ProbeState（单条 + batch）
  qwen2_probe.py        # Qwen2Attention patch（SDPA + probe）
  kernels/fused_probe.py # Torch 融合 / Triton 实验（默认不用 Triton）

attention_compressor_clean.py   # AttentionCompressor.compress / compress_batch

scripts/
  probe_compress.py     # 统一 CLI：run | bench | longbench
  run_probe.sh          # Shell 包装
  benchmark/            # ★ 评测与 profiling 套件（开源见 benchmark/README.md）
    profile_compress_10240_breakdown.py  # §2.0.1
    benchmark_autoregressive_decode.py   # §2.2 / §2.2b
    run/run_hf_prefill_bench.sh
    run/run_probe_tokenizer_bench.sh     # §9.1

docs/
  PROBE_OPTIMIZATION.md   # 本文档
```

---

## 8. 不推荐启用的选项

| 选项 | 原因 |
|------|------|
| `use_triton_probe=True` | A800 ~5k tokens 上 forward 0.068s vs Torch 0.052s |
| `use_torch_compile=True` | 实测更慢 |
| `use_probe_attention=False` | 回退 eager + 可能 chunk，~0.47s |
| `max_seq_len=1024` | 除非超长文本必须截断；probe 路径默认 None 单次 forward |

---

## 9. 数据集效率复现（multifieldqa_zh）

一键（AutoDL 项目根目录）：

```bash
chmod +x scripts/benchmark/run/run_probe_tokenizer_bench.sh
./scripts/benchmark/run/run_probe_tokenizer_bench.sh
```

或分步：

```bash
export PYTHONPATH=.
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME='/root/autodl-tmp/.cache/huggingface'

DET=train/detector/qwen2.5-0.5b-instruct-2000_all_layer_last_token_20260413_152942_model.pkl
DATA=experiments/llmlingua2/evaluation/data/LongBench/longbench_formatted.json

# 单条 demo 基准
python scripts/probe_compress.py bench --runs 5 --gpu 0

# 200 条：prefill + compress + 显存（10240 / 1024 各跑一遍）
python scripts/benchmark/benchmark_dataset_prefill.py \
  --load_origin_from "$DATA" --target_task multifieldqa_zh \
  --max_seq_len 10240 --target_token 2000 --gpu 0 \
  --save_path experiments/llmlingua2/evaluation/results/multifieldqa_zh_stats_10240.json

python scripts/benchmark/benchmark_dataset_prefill.py \
  --max_seq_len 1024 --target_token 2000 --gpu 0 \
  --save_path experiments/llmlingua2/evaluation/results/multifieldqa_zh_stats_1024.json

python scripts/benchmark/summarize_probe_chunk_comparison.py \
  --stats-10240 experiments/llmlingua2/evaluation/results/multifieldqa_zh_stats_10240.json \
  --stats-1024 experiments/llmlingua2/evaluation/results/multifieldqa_zh_stats_1024.json \
  --out experiments/llmlingua2/evaluation/results/probe_chunk_comparison_summary.json
```

关注指标：`prefill_time_s`、`compress_time_s`、`peak_prefill_mb`、`n_multi_chunk`。优化后 `compress_time_s` 应更接近 `prefill_time_s`（少 1～2 次全长 7B encode）。

### 9.1 论文效率对比表（multifieldqa_zh，200 条，2026-05-31 重跑）

完整 LaTeX 见 `experiments/llmlingua2/evaluation/results/PAPER_EFFICIENCY_TABLE.md`。

| Chunk cap | Multi-chunk | **Prefill** (ms) | 其中 tokenize† | **Compress** (ms) | Δ (ms) | Peak VRAM plain | Peak VRAM probe | Peak VRAM compress | Probe−Plain |
|-----------|-------------|------------------|----------------|-------------------|--------|-----------------|-----------------|--------------------|--------------|
| **10240** | 0 / 200 | **59.0 ± 27.2** | 8.6 | **66.7 ± 26.6** | **+7.7** | **2321 MB** | **2323 MB** | **2323 MB** | **+1.7 MB** |
| **1024** | 194 / 200 | 127.8 ± 56.6 | 9.8 | 172.3 ± 81.9 | +44.5 | **1268 MB** | **1276 MB** | **1276 MB** | **+8.1 MB** |

† 0.5B tokenize，`offset_mapping=False`（标准 prefill 编码口径）

**VRAM 口径**：`torch.cuda.max_memory_allocated()` 峰值（PyTorch tensor 分配），每次 forward/compress 前 `reset_peak_memory_stats()`；**不含** vLLM 7B eval、CUDA context 缓存、`nvidia-smi` 进程总占用。compress peak ≈ probe peak（同一 0.5B forward 主导）。

- **Prefill（本表 59 ms）**：`tokenize(offset=False)` **8.6 ms** + **forward+probe ~50.5 ms**；**不是**纯 GPU attention（纯 SDPA ≈ **45.8 ms**，见 §2.0.1）。
- **Compress（本表 66.7 ms）**：完整 `compress()` 墙钟；§2.0.1 细分重跑 **72.3 ms**（同脚本、含选句拼串 ~7.8 ms）。
- **Δ（59 vs 67，旧表）~8 ms**：主要是 §9.1 prefill **未计全 prep** + **未计选句**；按 §2.0.1 对 **forward+probe** 基准，compress 多 **~21 ms**（prep + 后处理）。
- **10240 vs 1024**：prefill **2.2×**、compress **2.6×** 更快。

## 10. 后续可选优化

1. **flash-attn 2**：安装成功后设置 `probe_attn_implementation="flash_attention_2"`，主路径 SDPA 可能再快一点；probe side-channel 不变。
   ```bash
   # AutoDL 终端直接跑，避免 SSH 超时
   nohup pip install flash-attn --no-build-isolation > /tmp/flash_attn_install.log 2>&1 &
   tail -f /tmp/flash_attn_install.log
   ```
2. **Triton block GEMM**：代码在 `kernels/fused_probe.py`，需进一步融合 / 减少 24 层 kernel launch 才有收益。
3. **LongBench batch_size**：按显存调 `4~8`；长度差异大时可按 token 长度分桶 batch。

---

## 11. 常见问题

**Q: `ModuleNotFoundError: llmlingua.probe`**

```bash
export PYTHONPATH=.    # 在项目根目录执行
```

**Q: bench 只有 68 tokens / 0.02s**

未加载 demo 长文。确认 `demo_attention_compression.py` 存在，或使用 `load_bench_text()` / 不传 `--context` 的 `bench` 子命令。

**Q: CUDA OOM on batch**

减小 `--batch_size` 或 `--batch_size 2`；单条 `compress()` 不受影响。

**Q: 与训练特征是否一致**

Probe 约定：full-sequence softmax（last query row）→ slice context → context 内 renorm → sentence mean；与训练 last-token 路径对齐（max diff ≈ 2e-4）。

---

*最后更新：2026-06-10（§2.3 HF vs vLLM 同条件 batch=1 对比）· 验证环境 AutoDL A800 + conda `sentinel` / `revllm`*
