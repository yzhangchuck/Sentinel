#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Batch attention-based text compression (姚明维基长文, 4 questions).

Usage:
    python demo_attention_compression_batch.py

Requires:
    - Detector: models/detectors/qwen2.5-0.5b-instruct-3000_all_layer_last_token_qwen2_20250507_212739_model.pkl
    - GPU recommended (CUDA)
"""

import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attention_compressor import AttentionCompressor
from demo_attention_compression import (
    COMPRESSION_RATE,
    CONTEXT,
    DETECTOR_DIR,
    DETECTOR_FILE,
)

BATCH_QUESTIONS = [
    "姚明受过几次伤",
    "姚明生涯拿过几个mvp",
    "姚明哪年入选名人堂",
    "姚明在哪些球队工作过",
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_path = os.path.join(DETECTOR_DIR, DETECTOR_FILE)

    print("=== Sentinel Batch Compression Demo ===")
    print(f"Device: {device}")
    print(f"Compression rate: {COMPRESSION_RATE:.0%}")
    print(f"Detector: {detector_path}")
    print(f"Samples: {len(BATCH_QUESTIONS)} (same context, different questions)")

    if not os.path.exists(detector_path):
        print("\nDetector not found. Place the .pkl file under models/detectors/")
        print("See README.md for download instructions.")
        sys.exit(1)

    compressor = AttentionCompressor(
        attention_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        detector_path=detector_path,
        max_seq_len=None,
        device=device,
        probe_attn_implementation="sdpa",
        use_pure_gpu=True,
        use_fast_chinese_split=True,
        print_sentence_scores=False,
    )

    samples = [
        {"context": CONTEXT, "question": q, "context_type": "chinese"}
        for q in BATCH_QUESTIONS
    ]

    t0 = time.perf_counter()
    results = compressor.compress_batch(
        samples,
        compression_rate=COMPRESSION_RATE,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nBatch processing time: {elapsed:.3f}s ({elapsed / len(samples):.3f}s per sample)")
    print("\nOriginal (first 200 chars):")
    print(CONTEXT[:200] + "...")

    for i, (question, result) in enumerate(zip(BATCH_QUESTIONS, results), 1):
        print(f"\n--- [{i}] {question} ---")
        print(f"Original length:   {result['original_length']} tokens")
        print(f"Compressed length: {result['compressed_length']} tokens")
        print(f"Compression ratio: {result['compression_ratio']:.2%}")
        print("Compressed text:")
        print(result["compressed_text"])


if __name__ == "__main__":
    main()
