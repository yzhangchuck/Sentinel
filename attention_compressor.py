#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-based Text Compressor (opensource).

Main flow: last-row probe (SDPA) + torch detector + sentence selection.
Public API: compress(), compress_batch().
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Tuple, Union, Dict, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
import nltk
import gc

from probe import ProbeState, patch_qwen2_attention_for_probe


class AttentionCompressor:
    """
    Slim text compressor using attention + detector for sentence-level importance.
    Context-only patch (Scheme 4+), detector-based filtering only.
    """

    def __init__(
        self,
        attention_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        detector_path: Optional[str] = None,
        eval_tokenizer_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_seq_len: Optional[int] = None,
        device: str = "cuda",
        use_threshold_by_default: bool = False,
        default_threshold: float = 0.5,
        min_word_length: int = 1,
        print_sentence_scores: bool = True,
        use_pure_gpu: bool = False,
        use_fast_chinese_split: bool = False,
        use_probe_attention: bool = True,
        probe_attn_implementation: str = "sdpa",
        use_torch_compile: bool = False,
        use_triton_probe: bool = False,
        sentence_budget_tokenizer: Literal["7b", "0.5b"] = "7b",
        sentence_tokenize_workers: int = 0,
        batch_prep_workers: int = 0,
        use_prep_pipeline: bool = True,
        disable_chunking: bool = False,
        **kwargs,  # Accept extra params for demo/profile compatibility
    ):
        self.attention_model_path = attention_model_path
        self.detector_path = detector_path
        self._max_seq_len_config = max_seq_len
        self.max_seq_len = max_seq_len or 32768
        self.use_threshold_by_default = use_threshold_by_default
        self.default_threshold = default_threshold
        self.min_word_length = min_word_length
        self.print_sentence_scores = print_sentence_scores
        self.use_pure_gpu = use_pure_gpu
        self.use_fast_chinese_split = use_fast_chinese_split
        if not use_probe_attention:
            raise ValueError("opensource AttentionCompressor requires use_probe_attention=True")
        self.use_probe_attention = True
        self.probe_attn_implementation = probe_attn_implementation
        self.use_torch_compile = use_torch_compile
        self.use_triton_probe = use_triton_probe
        self._probe_state = None
        self._filtering_cache: Dict[Tuple, dict] = {}
        self._filtering_cache_max = 32
        self._doc_prep_cache: Dict[Tuple[str, str], Tuple[List[str], Optional[List[int]]]] = {}
        self._ctx_budget_cache: Dict[Tuple[int, str], int] = {}
        self.sentence_budget_tokenizer = sentence_budget_tokenizer
        self.sentence_tokenize_workers = max(0, int(sentence_tokenize_workers))
        self.batch_prep_workers = max(0, int(batch_prep_workers))
        self.use_prep_pipeline = bool(use_prep_pipeline)
        self.disable_chunking = bool(disable_chunking)
        self._prep_cache_lock = threading.Lock()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._load_attention_model()
        self._load_eval_tokenizer(eval_tokenizer_path)
        self._load_detector()
        self._setup_text_processing()

        print(f"AttentionCompressor initialized:")
        print(f"  - Model: {attention_model_path}")
        print(f"  - Mode: last-row probe + SDPA detector filtering")
        print(f"  - Max_seq_len: {self.max_seq_len}" + (
            " (model max, single forward)" if self._max_seq_len_config is None else ""
        ))
        print(f"  - Device: {self.device}")
        if self.use_pure_gpu:
            print(f"  - Pure GPU: detector on GPU")
        if self.use_fast_chinese_split:
            print(f"  - Fast Chinese split: rule-based (no spacy)")
        if self.use_torch_compile:
            print(f"  - torch.compile: enabled")
        if self.use_triton_probe:
            print(f"  - Triton fused probe: enabled")
        if self.sentence_budget_tokenizer != "7b":
            print(f"  - Sentence budget tokenizer: {self.sentence_budget_tokenizer}")
        if self.sentence_tokenize_workers > 0:
            print(f"  - Parallel sentence tokenize workers: {self.sentence_tokenize_workers}")
        if self.batch_prep_workers > 0:
            print(f"  - Batch prep workers: {self.batch_prep_workers}")
        if self.use_prep_pipeline:
            print(f"  - Prep/forward pipeline: enabled (overlap CPU prep with GPU)")
        if self.disable_chunking:
            print(f"  - Chunking: disabled (single forward, no split/gate)")

    def _apply_torch_compile(self):
        try:
            self.attention_model = torch.compile(
                self.attention_model,
                mode="reduce-overhead",
                fullgraph=False,
            )
            print("  - torch.compile: applied to attention_model")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            self.use_torch_compile = False

    def _load_attention_model(self):
        """Load the attention model and tokenizer"""
        print(f"Loading attention model from: {self.attention_model_path}")
        if "0.5" in self.attention_model_path or "0.5B" in self.attention_model_path:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        attn_impl = self.probe_attn_implementation
        try:
            self.attention_model = AutoModelForCausalLM.from_pretrained(
                self.attention_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            ).eval()
        except (ValueError, ImportError) as e:
            if attn_impl != "sdpa":
                print(f"⚠️  attn_implementation={attn_impl} unavailable ({e}), fallback to sdpa")
                attn_impl = "sdpa"
                self.probe_attn_implementation = attn_impl
                self.attention_model = AutoModelForCausalLM.from_pretrained(
                    self.attention_model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                ).eval()
            else:
                raise
        self.attention_model.to(self.device)
        self.model_dtype = torch_dtype
        self._attn_implementation = attn_impl

        self.tokenizer = AutoTokenizer.from_pretrained(self.attention_model_path)
        config = self.attention_model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        model_max = int(getattr(config, "max_position_embeddings", 32768))
        if self._max_seq_len_config is None:
            self.max_seq_len = model_max
        else:
            self.max_seq_len = min(self._max_seq_len_config, model_max)

        self._setup_attention_capture()

    def _load_eval_tokenizer(self, eval_tokenizer_path: str):
        """Load evaluation tokenizer for token counting."""
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            eval_tokenizer_path, use_fast=True
        )

    def _load_detector(self):
        """Load trained detector for classification-based filtering"""
        if self.detector_path:
            print(f"Loading detector from: {self.detector_path}")
            self.detector = joblib.load(self.detector_path)
            self._build_torch_detector_from_sklearn(self.detector)
        else:
            self.detector = None
            self.torch_detector = None
            self.detector_scaler = None

    def _build_torch_detector_from_sklearn(self, model):
        """Convert sklearn LR (and optional StandardScaler) to torch for GPU inference."""
        self.torch_detector = None
        self.detector_scaler = None

        clf = model
        scaler = None

        if hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "coef_"):
                    clf = step
                elif hasattr(step, "mean_") and hasattr(step, "scale_"):
                    scaler = step

        if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
            print("ℹ️  Torch detector conversion skipped, using sklearn predict_proba.")
            return

        out_features, in_features = clf.coef_.shape
        linear = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
            linear.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
        linear.requires_grad_(False)
        linear.to(self.device)
        linear.eval()
        self.torch_detector = linear

        if scaler is not None:
            self.detector_scaler = {
                "mean": torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device),
                "scale": torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device).clamp(min=1e-8),
            }
        else:
            self.detector_scaler = None

        print(f"✅ Torch detector ready: in={in_features}, out={out_features}")

    def _setup_text_processing(self):
        """Setup text processing. Spacy lazy-loaded when use_fast_chinese_split=False."""
        self.zh_sent_tokenize = None  # Lazy load when needed

    def _ensure_spacy_loaded(self):
        """Lazy-load spacy only when use_fast_chinese_split=False and context_type=chinese."""
        if self.zh_sent_tokenize is not None:
            return
        if self.use_fast_chinese_split:
            return
        try:
            import spacy
            self.zh_sent_tokenize = spacy.load(
                "zh_core_web_sm",
                disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
            )
            self.zh_sent_tokenize.add_pipe('sentencizer')
        except OSError:
            print("⚠️  zh_core_web_sm not found, Chinese will use simple split")

    def _setup_attention_capture(self):
        """Setup last-row probe (SDPA + side-channel)."""
        self._probe_state = ProbeState(
            self.device, self.model_dtype, self.num_layers, self.num_heads,
            use_triton=self.use_triton_probe,
        )
        n = patch_qwen2_attention_for_probe(self.attention_model, self._probe_state)
        dummy_inputs = self.tokenizer("hello world", return_tensors="pt").to(self.device)
        seq_len = dummy_inputs["input_ids"].shape[1]
        self._probe_state.begin(
            [(0, min(2, seq_len - 1))],
            0,
            min(2, seq_len - 1) if seq_len > 0 else 0,
        )
        with torch.no_grad():
            _ = self.attention_model(**dummy_inputs, output_attentions=False, return_dict=True)
        self._probe_state.clear()
        print(f"  - Last-row probe patch: enabled ({n} attention layers, {self._attn_implementation})")
        if self.use_torch_compile:
            self._apply_torch_compile()
            with torch.no_grad():
                _ = self.attention_model(**dummy_inputs, output_attentions=False, return_dict=True)

    def _compress_with_chunking(
        self, context: str, question: str, context_type: str
    ) -> Tuple[List, List[str], List[int]]:
        """Question-aware chunk gate + optional multi-forward."""
        max_ctx_tokens = self._max_context_tokens_for_forward(question)
        doc_sentences = self._split_context_sentences(context, context_type)
        if not doc_sentences:
            return [], [], []
        attn_toks = self._batch_count_tokens(doc_sentences, self.tokenizer)
        if sum(attn_toks) <= max_ctx_tokens:
            return self._get_sentence_scores(
                context,
                question,
                context_type,
                preset_sentences=doc_sentences,
                preset_sentence_tokens=self._count_sentence_tokens(doc_sentences),
            )
        chunk_specs = self._build_attention_chunk_specs(
            doc_sentences, attn_toks, max_ctx_tokens, context_type
        )
        all_sentence_scores: List = []
        all_sentences: List[str] = []
        all_sentence_tokens: List[int] = []
        for spec in chunk_specs:
            chunk_sents = spec.get("sentences")
            chunk_budget_toks = (
                self._count_sentence_tokens(chunk_sents) if chunk_sents else None
            )
            chunk_scores, chunk_sentences, chunk_tokens = self._get_sentence_scores(
                spec["text"],
                question,
                context_type,
                preset_sentences=chunk_sents,
                preset_sentence_tokens=chunk_budget_toks,
            )
            all_sentence_scores.extend(chunk_scores)
            all_sentences.extend(chunk_sentences)
            all_sentence_tokens.extend(chunk_tokens)
        return all_sentence_scores, all_sentences, all_sentence_tokens

    def compress(
        self,
        context: str,
        question: str = "",
        target_token: int = -1,
        compression_rate: float = 0.5,
        context_type: str = "english",
        use_threshold_filtering: bool = False,
        threshold: float = 0.5,
    ) -> Dict[str, Union[str, List, Dict]]:
        """
        Compress text using attention-based filtering.

        Returns:
            Dict: compressed_text, original_length, compressed_length, compression_ratio,
                  sentence_scores, sentences, preserved_indices, processing_time
        """
        start_time = time.time()

        if self.disable_chunking:
            sentence_scores, sentences, sentence_tokens = self._get_sentence_scores(
                context, question, context_type
            )
        else:
            sentence_scores, sentences, sentence_tokens = self._compress_with_chunking(
                context, question, context_type
            )

        total_tokens = sum(sentence_tokens)

        if self.use_threshold_by_default and not use_threshold_filtering:
            use_threshold_filtering = True
            threshold = self.default_threshold

        if use_threshold_filtering:
            compressed_text, preserved_indices = self._select_sentences_by_threshold(
                sentences, sentence_scores, sentence_tokens, threshold, context_type
            )
            target_tokens = sum(sentence_tokens[i] for i in preserved_indices)
            actual_compression_rate = 1.0 - (target_tokens / total_tokens) if total_tokens > 0 else 0.0
        else:
            if target_token > 0:
                target_tokens = min(target_token, total_tokens)
                actual_compression_rate = 1.0 - (target_tokens / total_tokens)
            else:
                target_tokens = int(total_tokens * (1 - compression_rate))
                actual_compression_rate = compression_rate

            compressed_text, preserved_indices, joined_token_len = self._select_sentences(
                sentences, sentence_scores, sentence_tokens, target_tokens, context_type
            )

        if use_threshold_filtering:
            joined_token_len = None
        compressed_tokens = (
            joined_token_len
            if joined_token_len is not None
            else self._compressed_length_from_text(compressed_text)
        )

        processing_time = time.time() - start_time

        return {
            'compressed_text': compressed_text,
            'original_length': total_tokens,
            'compressed_length': compressed_tokens,
            'compression_ratio': actual_compression_rate,
            'sentence_scores': sentence_scores,
            'sentences': sentences,
            'preserved_indices': preserved_indices,
            'processing_time': processing_time
        }

    def compress_batch(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 4,
        target_token: int = -1,
        compression_rate: float = 0.5,
        use_threshold_filtering: bool = False,
        threshold: float = 0.5,
        length_bucket: bool = True,
        use_prep_pipeline: Optional[bool] = None,
    ) -> List[Dict[str, Union[str, List, Dict]]]:
        """
        Batch compress for throughput (single-sample latency unchanged).

        Each sample dict: context, question (optional), context_type (optional).

        length_bucket: sort by length before chunking (ordering only when pipeline on).

        use_prep_pipeline: overlap CPU tokenize/split for sample i+1 with GPU
        forward on sample i. Recommended for heterogeneous LongBench workloads.
        Set False to use batched GPU forward (best when prompts are similar length).
        """
        if not samples:
            return []
        pipeline = (
            self.use_prep_pipeline if use_prep_pipeline is None else use_prep_pipeline
        )
        chunk_kwargs = dict(
            target_token=target_token,
            compression_rate=compression_rate,
            use_threshold_filtering=use_threshold_filtering,
            threshold=threshold,
        )
        if pipeline and len(samples) > 1:
            if length_bucket and batch_size > 1:
                indexed = list(enumerate(samples))
                indexed.sort(
                    key=lambda item: len(item[1].get("context", ""))
                    + len(item[1].get("question", ""))
                )
                results: List[Optional[Dict]] = [None] * len(samples)
                for start in range(0, len(indexed), batch_size):
                    bucket = indexed[start : start + batch_size]
                    chunk = [sample for _, sample in bucket]
                    chunk_results = self._compress_sequential_pipelined(
                        chunk, **chunk_kwargs
                    )
                    for (orig_idx, _), result in zip(bucket, chunk_results):
                        results[orig_idx] = result
                return results  # type: ignore[return-value]
            return self._compress_sequential_pipelined(samples, **chunk_kwargs)

        if batch_size <= 1 or len(samples) <= 1 or not length_bucket:
            results: List[Dict] = []
            for start in range(0, len(samples), batch_size):
                chunk = samples[start : start + batch_size]
                results.extend(
                    self._compress_batch_chunk(
                        chunk,
                        target_token=target_token,
                        compression_rate=compression_rate,
                        use_threshold_filtering=use_threshold_filtering,
                        threshold=threshold,
                    )
                )
            return results

        indexed = list(enumerate(samples))
        indexed.sort(
            key=lambda item: len(item[1].get("context", ""))
            + len(item[1].get("question", ""))
        )
        results: List[Optional[Dict]] = [None] * len(samples)
        for start in range(0, len(indexed), batch_size):
            bucket = indexed[start : start + batch_size]
            chunk = [sample for _, sample in bucket]
            chunk_results = self._compress_batch_chunk(
                chunk,
                target_token=target_token,
                compression_rate=compression_rate,
                use_threshold_filtering=use_threshold_filtering,
                threshold=threshold,
            )
            for (orig_idx, _), result in zip(bucket, chunk_results):
                results[orig_idx] = result
        return results  # type: ignore[return-value]

    def _prepare_sample_package(self, sample: Dict[str, str]) -> dict:
        """CPU-only prep for one sample (thread-safe)."""
        context = sample["context"]
        question = sample.get("question", "")
        context_type = sample.get("context_type", "english")
        if self._sample_needs_chunking(context, question, context_type):
            return {"needs_chunking": True, "sample": sample}
        with self._prep_cache_lock:
            doc_sentences, preset_tokens = self._doc_sentences_and_tokens(
                context, context_type
            )
            prep = self._prepare_filtering_inputs(
                context,
                question,
                context_type,
                preset_sentences=doc_sentences if doc_sentences else None,
                preset_sentence_tokens=preset_tokens,
            )
        return {
            "needs_chunking": False,
            "context": context,
            "context_type": context_type,
            "prep": prep,
        }

    def _forward_scores_from_prep(
        self, prep: dict, context_type: str
    ) -> Tuple[Union[List[float], torch.Tensor], List[str], List[int]]:
        """GPU forward + detector on ready prep (main thread only)."""
        if self.torch_detector is None:
            raise ValueError("Torch detector not loaded. Detector required for clean mode.")

        _tf32_restore = None
        if self.use_pure_gpu and torch.cuda.is_available():
            _tf32_restore = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            with torch.inference_mode():
                self._probe_state.begin(
                    prep["sent_positions"], prep["context_start"], prep["context_end"]
                )
                _ = self.attention_model(
                    input_ids=prep["inputs"]["input_ids"],
                    attention_mask=prep["inputs"]["attention_mask"],
                    output_attentions=False,
                    return_dict=True,
                )
                vectors = self._probe_state.finalize_vectors()
            if vectors is None:
                raise ValueError("Attention probe processing failed to produce features.")
            return self._vectors_to_sentence_scores(
                vectors,
                prep["sentences"],
                prep["sentence_tokens"],
                context_type,
            )
        finally:
            if _tf32_restore is not None:
                torch.backends.cuda.matmul.allow_tf32 = _tf32_restore

    def _compress_from_prep_package(
        self,
        package: dict,
        target_token: int,
        compression_rate: float,
        use_threshold_filtering: bool,
        threshold: float,
    ) -> Dict[str, Union[str, List, Dict]]:
        if package.get("needs_chunking"):
            sample = package["sample"]
            return self.compress(
                context=sample["context"],
                question=sample.get("question", ""),
                target_token=target_token,
                compression_rate=compression_rate,
                context_type=sample.get("context_type", "english"),
                use_threshold_filtering=use_threshold_filtering,
                threshold=threshold,
            )

        start_time = time.time()
        prep = package["prep"]
        context_type = package["context_type"]
        sentence_scores, sentences, sentence_tokens = self._forward_scores_from_prep(
            prep, context_type
        )
        return self._finalize_compress_result(
            package["context"],
            sentences,
            sentence_scores,
            sentence_tokens,
            context_type,
            target_token,
            compression_rate,
            use_threshold_filtering,
            threshold,
            time.time() - start_time,
        )

    def _compress_sequential_pipelined(
        self,
        samples: List[Dict[str, str]],
        target_token: int,
        compression_rate: float,
        use_threshold_filtering: bool,
        threshold: float,
    ) -> List[Dict[str, Union[str, List, Dict]]]:
        """Overlap CPU prep for next sample with GPU forward on current sample."""
        if len(samples) == 1:
            return [
                self._compress_from_prep_package(
                    self._prepare_sample_package(samples[0]),
                    target_token,
                    compression_rate,
                    use_threshold_filtering,
                    threshold,
                )
            ]

        queue: Queue = Queue(maxsize=2)
        errors: List[BaseException] = []

        def _prep_worker() -> None:
            for idx, sample in enumerate(samples):
                try:
                    queue.put((idx, self._prepare_sample_package(sample)))
                except BaseException as exc:
                    errors.append(exc)
                    return

        worker = threading.Thread(target=_prep_worker, daemon=True)
        worker.start()

        results: List[Optional[Dict]] = [None] * len(samples)
        for _ in range(len(samples)):
            if errors:
                worker.join(timeout=0.1)
                raise errors[0]
            idx, package = queue.get()
            results[idx] = self._compress_from_prep_package(
                package,
                target_token,
                compression_rate,
                use_threshold_filtering,
                threshold,
            )
        worker.join()
        if errors:
            raise errors[0]
        return results  # type: ignore[return-value]

    def _sample_needs_chunking(
        self, context: str, question: str, context_type: str
    ) -> bool:
        if self.disable_chunking:
            return False
        doc_sentences = self._split_context_sentences(context, context_type)
        if not doc_sentences:
            return False
        attn_toks = self._batch_count_tokens(doc_sentences, self.tokenizer)
        return sum(attn_toks) > self._max_context_tokens_for_forward(question)

    def _compress_batch_chunk(
        self,
        samples: List[Dict[str, str]],
        target_token: int,
        compression_rate: float,
        use_threshold_filtering: bool,
        threshold: float,
    ) -> List[Dict]:
        if len(samples) == 1:
            s = samples[0]
            return [
                self.compress(
                    context=s["context"],
                    question=s.get("question", ""),
                    target_token=target_token,
                    compression_rate=compression_rate,
                    context_type=s.get("context_type", "english"),
                    use_threshold_filtering=use_threshold_filtering,
                    threshold=threshold,
                )
            ]

        if any(
            self._sample_needs_chunking(
                s["context"], s.get("question", ""), s.get("context_type", "english")
            )
            for s in samples
        ):
            return [
                self.compress(
                    context=s["context"],
                    question=s.get("question", ""),
                    target_token=target_token,
                    compression_rate=compression_rate,
                    context_type=s.get("context_type", "english"),
                    use_threshold_filtering=use_threshold_filtering,
                    threshold=threshold,
                )
                for s in samples
            ]

        start_time = time.time()
        batch_prep = self._prepare_filtering_batch(samples)
        per_sample = batch_prep["per_sample"]
        inputs = batch_prep["inputs"]
        batch_meta = batch_prep["batch_meta"]
        n_valid_list = [len(m["sent_positions"]) for m in batch_meta]

        with torch.inference_mode():
            self._probe_state.begin_batch(batch_meta)
            _ = self.attention_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=False,
                return_dict=True,
            )
            vectors = self._probe_state.finalize_batch_vectors()

        if vectors is None:
            raise ValueError("Batch probe failed to produce features.")

        flat_chunks = [vectors[b, : n_valid_list[b]] for b in range(len(samples))]
        flat_vectors = torch.cat(flat_chunks, dim=0)
        flat_probs = self._detector_probs_from_vectors(flat_vectors)

        elapsed = time.time() - start_time
        per_time = elapsed / max(len(samples), 1)
        identical_prep = self._batch_samples_share_prep(per_sample)

        chunk_results = []
        offset = 0
        for b, prep in enumerate(per_sample):
            n_valid = n_valid_list[b]
            sentence_probs = flat_probs[offset : offset + n_valid]
            offset += n_valid
            if self.use_pure_gpu:
                sentence_scores = sentence_probs
            else:
                sentence_scores = sentence_probs.detach().cpu().tolist()
            sentence_scores = self._apply_context_type_score_adjustments(
                sentence_scores,
                prep["sentences"],
                prep["context_type"],
            )
            if identical_prep and b > 0:
                chunk_results.append({**chunk_results[0], "processing_time": per_time})
                continue
            chunk_results.append(
                self._finalize_compress_result(
                    prep["context"],
                    prep["sentences"],
                    sentence_scores,
                    prep["sentence_tokens"],
                    prep["context_type"],
                    target_token,
                    compression_rate,
                    use_threshold_filtering,
                    threshold,
                    per_time,
                )
            )
        return chunk_results

    def _build_filtering_prompt(self, context: str, question: str) -> str:
        return (
            f"Given the following information: {context}\n"
            f"Answer the following question based on the given information with one or few words: {question}\n"
            f"Answer:"
        )

    @staticmethod
    def _filtering_cache_key(
        context: str,
        question: str,
        context_type: str,
        preset_sentences: Optional[List[str]] = None,
    ) -> Tuple:
        return (
            context,
            question,
            context_type,
            tuple(preset_sentences) if preset_sentences is not None else None,
        )

    def _filtering_cache_get(self, key: Tuple) -> Optional[dict]:
        return self._filtering_cache.get(key)

    def _filtering_cache_put(self, key: Tuple, value: dict) -> None:
        self._filtering_cache[key] = value
        if len(self._filtering_cache) > self._filtering_cache_max:
            self._filtering_cache.pop(next(iter(self._filtering_cache)))

    @staticmethod
    def _batch_samples_share_prep(per_sample: List[dict]) -> bool:
        if len(per_sample) <= 1:
            return False
        first = per_sample[0]
        return all(
            p["sentences"] == first["sentences"]
            and p["sentence_tokens"] == first["sentence_tokens"]
            and p["context_type"] == first["context_type"]
            for p in per_sample[1:]
        )

    def _align_filtering_row(
        self,
        context: str,
        question: str,
        context_type: str,
        prompt: str,
        offset_mapping: Union[np.ndarray, torch.Tensor],
        preset_sentences: Optional[List[str]] = None,
        preset_sentence_tokens: Optional[List[int]] = None,
    ) -> dict:
        """Map one tokenized prompt row to sentence metadata (CPU)."""
        context_start, context_end = self._find_context_position(
            offset_mapping, prompt, context
        )
        if preset_sentences is not None:
            sent_positions, sentences, sentence_tokens = self._map_sentences_to_offsets(
                offset_mapping,
                prompt,
                context,
                preset_sentences,
                preset_sentence_tokens,
            )
        else:
            sent_positions, sentences, sentence_tokens = self._split_into_sentences(
                offset_mapping, prompt, context, context_start, context_type
            )
        return {
            "context_start": context_start,
            "context_end": context_end,
            "sent_positions": sent_positions,
            "sentences": sentences,
            "sentence_tokens": sentence_tokens,
        }

    def _doc_sentences_and_tokens(
        self, context: str, context_type: str
    ) -> Tuple[List[str], Optional[List[int]]]:
        """Split + 7B token count once per (context, context_type)."""
        doc_key = (context, context_type)
        cached = self._doc_prep_cache.get(doc_key)
        if cached is not None:
            return cached
        doc_sentences = self._split_context_sentences(context, context_type)
        preset_tokens = (
            self._count_sentence_tokens(doc_sentences) if doc_sentences else None
        )
        self._doc_prep_cache[doc_key] = (doc_sentences, preset_tokens)
        return doc_sentences, preset_tokens

    def _prepare_filtering_row_meta(
        self, sample: Dict[str, str], prompt: str, offset_mapping
    ) -> Tuple[dict, dict]:
        context = sample["context"]
        question = sample.get("question", "")
        context_type = sample.get("context_type", "english")
        doc_sentences, preset_tokens = self._doc_sentences_and_tokens(
            context, context_type
        )
        cache_key = self._filtering_cache_key(
            context, question, context_type, doc_sentences if doc_sentences else None
        )
        cached = self._filtering_cache_get(cache_key)
        if cached is not None:
            row = {
                "context_start": cached["context_start"],
                "context_end": cached["context_end"],
                "sent_positions": cached["sent_positions"],
                "sentences": cached["sentences"],
                "sentence_tokens": cached["sentence_tokens"],
            }
        else:
            row = self._align_filtering_row(
                context,
                question,
                context_type,
                prompt,
                offset_mapping,
                preset_sentences=doc_sentences if doc_sentences else None,
                preset_sentence_tokens=preset_tokens,
            )
            self._filtering_cache_put(
                cache_key,
                {
                    "context_start": row["context_start"],
                    "context_end": row["context_end"],
                    "sent_positions": row["sent_positions"],
                    "sentences": row["sentences"],
                    "sentence_tokens": row["sentence_tokens"],
                },
            )
        per_sample = {
            "context": context,
            "context_type": context_type,
            "sentences": row["sentences"],
            "sentence_tokens": row["sentence_tokens"],
        }
        batch_meta = {
            "sent_positions": row["sent_positions"],
            "context_start": row["context_start"],
            "context_end": row["context_end"],
        }
        return per_sample, batch_meta

    def _prepare_filtering_batch(self, samples: List[Dict[str, str]]) -> dict:
        """One batched tokenize + parallel CPU align (same semantics as compress())."""
        prompts = [
            self._build_filtering_prompt(s["context"], s.get("question", ""))
            for s in samples
        ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        offset_mappings = encoded.pop("offset_mapping")
        inputs = {
            k: v.to(self.device, non_blocking=(self.device.type == "cuda"))
            for k, v in encoded.items()
        }

        per_sample: List[dict] = []
        batch_meta: List[dict] = []
        # Warm doc-level cache once per unique context before parallel align.
        for sample in samples:
            self._doc_sentences_and_tokens(
                sample["context"], sample.get("context_type", "english")
            )
        unique_samples = len(
            {
                (s["context"], s.get("question", ""), s.get("context_type", "english"))
                for s in samples
            }
        )
        workers = self.sentence_tokenize_workers
        if workers <= 0 and unique_samples == len(samples) and len(samples) > 1:
            workers = self.batch_prep_workers or min(4, len(samples))
        if workers > 1 and unique_samples > 1:

            def _row(i: int):
                offset_raw = offset_mappings[i]
                if self.use_pure_gpu and self.device.type == "cuda":
                    om = offset_raw.to(self.device)
                else:
                    om = offset_raw.cpu().numpy()
                return self._prepare_filtering_row_meta(samples[i], prompts[i], om)

            with ThreadPoolExecutor(max_workers=min(workers, len(samples))) as pool:
                rows = list(pool.map(_row, range(len(samples))))
        else:
            rows = []
            for i, sample in enumerate(samples):
                offset_raw = offset_mappings[i]
                if self.use_pure_gpu and self.device.type == "cuda":
                    om = offset_raw.to(self.device)
                else:
                    om = offset_raw.cpu().numpy()
                rows.append(self._prepare_filtering_row_meta(sample, prompts[i], om))

        for ps, bm in rows:
            per_sample.append(ps)
            batch_meta.append(bm)

        return {"inputs": inputs, "per_sample": per_sample, "batch_meta": batch_meta}

    def _detector_probs_from_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Run torch detector on [N, D] vectors; returns [N] probs on device."""
        expected_dim = self.torch_detector.in_features
        if vectors.shape[-1] != expected_dim:
            if (
                vectors.shape[-1] == self.num_heads
                and expected_dim == self.num_layers * self.num_heads
            ):
                vectors = vectors.repeat(1, self.num_layers)
            else:
                raise ValueError(
                    f"Vector dim {vectors.shape[-1]} != detector expected {expected_dim}"
                )
        if vectors.dtype != torch.float32:
            vectors = vectors.to(dtype=torch.float32)
        if self.detector_scaler:
            vectors = (vectors - self.detector_scaler["mean"]) / self.detector_scaler["scale"]
        logits = self.torch_detector(vectors)
        return torch.sigmoid(logits).squeeze(-1)

    def _vectors_to_sentence_scores(
        self,
        vectors: torch.Tensor,
        sentences: List[str],
        sentence_tokens: List[int],
        context_type: str,
    ) -> Tuple[Union[List[float], torch.Tensor], List[str], List[int]]:
        sentence_probs = self._detector_probs_from_vectors(vectors)
        if self.use_pure_gpu:
            sentence_scores = sentence_probs
        else:
            sentence_scores = sentence_probs.detach().cpu().tolist()

        sentence_scores = self._apply_context_type_score_adjustments(
            sentence_scores, sentences, context_type
        )
        return sentence_scores, sentences, sentence_tokens

    @staticmethod
    def _sentences_contain_any(sentences: List[str], markers: Tuple[str, ...]) -> bool:
        return any(marker in sent for sent in sentences for marker in markers)

    def _mandatory_chinese_indices(self, sentences: List[str]) -> List[int]:
        """lsht-style mandatory sentences; skip scan when markers absent."""
        markers = ('新闻内容：', '类别：')
        if not self._sentences_contain_any(sentences, markers):
            return []
        return [i for i, sent in enumerate(sentences) if any(m in sent for m in markers)]

    def _apply_context_type_score_adjustments(
        self,
        sentence_scores: Union[List[float], torch.Tensor],
        sentences: List[str],
        context_type: str,
    ):
        if context_type == 'fewshot':
            fewshot_markers = ('Passage:', 'Question:', 'Answer:')
            if self._sentences_contain_any(sentences, fewshot_markers):
                if isinstance(sentence_scores, torch.Tensor):
                    out = sentence_scores.clone()
                    for i, sent in enumerate(sentences):
                        if any(marker in sent for marker in fewshot_markers):
                            out[i] = 1.0
                    return out
                for i, sent in enumerate(sentences):
                    if any(marker in sent for marker in fewshot_markers):
                        sentence_scores[i] = 1.0

        if context_type == 'chinese':
            chinese_markers = ('新闻内容', '类别')
            if self._sentences_contain_any(sentences, chinese_markers):
                if isinstance(sentence_scores, torch.Tensor):
                    mask = torch.tensor(
                        [any(marker in sent for marker in chinese_markers) for sent in sentences],
                        device=sentence_scores.device, dtype=torch.bool,
                    )
                    return torch.where(
                        mask, torch.clamp(sentence_scores, min=0.95), sentence_scores
                    )
                for i, sent in enumerate(sentences):
                    if any(marker in sent for marker in chinese_markers):
                        sentence_scores[i] = max(
                            sentence_scores[i] if sentence_scores else 0.0, 0.95
                        )
        return sentence_scores

    @staticmethod
    def _join_separator(context_type: str, sentences: List[str]) -> str:
        if context_type == "code" or context_type == "fewshot":
            return "\n\n"
        if context_type == "chinese" and any("新闻内容：" in s for s in sentences):
            return "\n\n"
        return " "

    @staticmethod
    def _join_compressed_sentences(
        sentences: List[str],
        preserved_indices: List[int],
        context_type: str,
    ) -> str:
        compressed_sentences = [sentences[i] for i in preserved_indices]
        sep = AttentionCompressor._join_separator(context_type, sentences)
        return sep.join(compressed_sentences)

    def _join_separator_token_cost(
        self, context_type: str, sentences: List[str]
    ) -> int:
        sep = self._join_separator(context_type, sentences)
        cache = getattr(self, "_join_sep_token_cache", None)
        if cache is None:
            cache = {}
            self._join_sep_token_cache = cache
        if sep not in cache:
            cache[sep] = self._encode_length(self._budget_tokenizer(), sep)
        return cache[sep]

    @staticmethod
    def _estimate_joined_token_count(
        indices: List[int],
        sentence_tokens: List[int],
        sep_cost: int,
    ) -> int:
        n = len(indices)
        if n == 0:
            return 0
        return sum(sentence_tokens[i] for i in indices) + max(0, n - 1) * sep_cost

    def _finalize_compress_result(
        self,
        context: str,
        sentences: List[str],
        sentence_scores,
        sentence_tokens: List[int],
        context_type: str,
        target_token: int,
        compression_rate: float,
        use_threshold_filtering: bool,
        threshold: float,
        processing_time: float,
    ) -> Dict:
        total_tokens = sum(sentence_tokens)

        if self.use_threshold_by_default and not use_threshold_filtering:
            use_threshold_filtering = True
            threshold = self.default_threshold

        if use_threshold_filtering:
            compressed_text, preserved_indices = self._select_sentences_by_threshold(
                sentences, sentence_scores, sentence_tokens, threshold, context_type
            )
            target_tokens = sum(sentence_tokens[i] for i in preserved_indices)
            actual_compression_rate = 1.0 - (target_tokens / total_tokens) if total_tokens > 0 else 0.0
        else:
            if target_token > 0:
                target_tokens = min(target_token, total_tokens)
                actual_compression_rate = 1.0 - (target_tokens / total_tokens)
            else:
                target_tokens = int(total_tokens * (1 - compression_rate))
                actual_compression_rate = compression_rate

            compressed_text, preserved_indices, joined_token_len = self._select_sentences(
                sentences, sentence_scores, sentence_tokens, target_tokens, context_type
            )

        if use_threshold_filtering:
            joined_token_len = None
        compressed_tokens = (
            joined_token_len
            if joined_token_len is not None
            else self._compressed_length_from_text(compressed_text)
        )

        return {
            'compressed_text': compressed_text,
            'original_length': total_tokens,
            'compressed_length': compressed_tokens,
            'compression_ratio': actual_compression_rate,
            'sentence_scores': sentence_scores,
            'sentences': sentences,
            'preserved_indices': preserved_indices,
            'processing_time': processing_time,
        }

    def _get_sentence_scores(
        self,
        context: str,
        question: str,
        context_type: str,
        preset_sentences: Optional[List[str]] = None,
        preset_sentence_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[str], List[int]]:
        """Get importance scores for each sentence using detector-based filtering."""
        return self._detector_based_filtering(
            context,
            question,
            context_type,
            preset_sentences=preset_sentences,
            preset_sentence_tokens=preset_sentence_tokens,
        )

    def _detector_based_filtering(
        self,
        context: str,
        question: str,
        context_type: str,
        preset_sentences: Optional[List[str]] = None,
        preset_sentence_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[str], List[int]]:
        """Use trained detector to score sentence importance."""
        if not self.detector:
            raise ValueError("Detector not loaded. Cannot perform detector-based filtering.")

        _tf32_restore = None
        if self.use_pure_gpu and torch.cuda.is_available():
            _tf32_restore = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return self._detector_based_filtering_impl(
                context,
                question,
                context_type,
                preset_sentences=preset_sentences,
                preset_sentence_tokens=preset_sentence_tokens,
            )
        finally:
            if _tf32_restore is not None:
                torch.backends.cuda.matmul.allow_tf32 = _tf32_restore

    def _prepare_filtering_inputs(
        self,
        context: str,
        question: str,
        context_type: str,
        preset_sentences: Optional[List[str]] = None,
        preset_sentence_tokens: Optional[List[int]] = None,
    ) -> dict:
        """Tokenize + split sentences; cache when same (context, question, context_type)."""
        cache_key = self._filtering_cache_key(
            context, question, context_type, preset_sentences
        )
        cached = self._filtering_cache_get(cache_key)
        if cached is not None and "inputs" in cached:
            return cached

        prompt = self._build_filtering_prompt(context, question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        offset_raw = inputs.pop("offset_mapping")[0]
        if self.use_pure_gpu and self.device.type == "cuda":
            offset_mapping = offset_raw.to(self.device)
        else:
            offset_mapping = offset_raw.cpu().numpy()
        inputs = {
            k: v.to(self.device, non_blocking=(self.device.type == "cuda"))
            for k, v in inputs.items()
        }
        row = self._align_filtering_row(
            context,
            question,
            context_type,
            prompt,
            offset_mapping,
            preset_sentences=preset_sentences,
            preset_sentence_tokens=preset_sentence_tokens,
        )
        context_num_tokens = int(row["context_end"] - row["context_start"] + 1)
        prep = {
            "inputs": inputs,
            "context_start": row["context_start"],
            "context_end": row["context_end"],
            "context_num_tokens": context_num_tokens,
            "sent_positions": row["sent_positions"],
            "sentences": row["sentences"],
            "sentence_tokens": row["sentence_tokens"],
        }
        self._filtering_cache_put(cache_key, prep)
        return prep

    def _detector_based_filtering_impl(
        self,
        context: str,
        question: str,
        context_type: str,
        preset_sentences: Optional[List[str]] = None,
        preset_sentence_tokens: Optional[List[int]] = None,
    ) -> Tuple[Union[List[float], torch.Tensor], List[str], List[int]]:
        """Probe forward + torch detector scoring."""
        prep = self._prepare_filtering_inputs(
            context,
            question,
            context_type,
            preset_sentences=preset_sentences,
            preset_sentence_tokens=preset_sentence_tokens,
        )
        inputs = prep["inputs"]
        context_start = prep["context_start"]
        context_end = prep["context_end"]
        sent_positions = prep["sent_positions"]
        sentences = prep["sentences"]
        sentence_tokens = prep["sentence_tokens"]

        if self.torch_detector is None:
            raise ValueError("Torch detector not loaded. Detector required for clean mode.")

        with torch.inference_mode():
            self._probe_state.begin(sent_positions, context_start, context_end)
            _ = self.attention_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=False,
                return_dict=True,
            )
            vectors = self._probe_state.finalize_vectors()
            if vectors is None:
                raise ValueError("Attention probe processing failed to produce features.")

        expected_dim = self.torch_detector.in_features
        if vectors.shape[1] != expected_dim:
            if vectors.shape[1] == self.num_heads and expected_dim == self.num_layers * self.num_heads:
                vectors = vectors.repeat(1, self.num_layers)
            else:
                raise ValueError(f"Vector dim {vectors.shape[1]} != detector expected {expected_dim}")

        # Detector needs float32; vectors from _stream_feat_sum already on device
        if vectors.dtype != torch.float32:
            vectors = vectors.to(dtype=torch.float32)

        if self.detector_scaler:
            vectors = (vectors - self.detector_scaler["mean"]) / self.detector_scaler["scale"]

        logits = self.torch_detector(vectors)
        sentence_probs = torch.sigmoid(logits).squeeze(-1)
        if self.use_pure_gpu:
            sentence_scores = sentence_probs
        else:
            sentence_scores = sentence_probs.detach().cpu().tolist()

        sentence_scores = self._apply_context_type_score_adjustments(
            sentence_scores, sentences, context_type
        )
        return sentence_scores, sentences, sentence_tokens

    def _find_context_position(
        self, offset_mapping: Union[np.ndarray, torch.Tensor], prompt: str, context: str
    ) -> Tuple[int, int]:
        """Find context position in token sequence."""
        start_char = prompt.find(context)
        end_char = start_char + len(context)
        token_starts = offset_mapping[:, 0]
        token_ends = offset_mapping[:, 1]
        valid_tokens = (token_ends > start_char) & (token_starts < end_char)
        if isinstance(offset_mapping, torch.Tensor):
            token_indices = valid_tokens.nonzero(as_tuple=True)[0]
            vals = token_indices[[0, -1]].cpu().tolist()
            return int(vals[0]), int(vals[1])
        token_indices = np.nonzero(valid_tokens)[0]
        return int(token_indices[0]), int(token_indices[-1])

    def _split_context_sentences(self, context: str, context_type: str) -> List[str]:
        """Split context into sentences (text only; shared by chunking and offset alignment)."""
        if context_type == 'english':
            raw_sentences = nltk.sent_tokenize(context)
            raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
            sentences = self._sync_sentence(raw_sentences, context)
        elif context_type == 'code':
            sentences = self._code_sentence_split_fallback(context)
        elif context_type == 'chinese':
            if self.use_fast_chinese_split:
                sentences = self._split_chinese_simple(context)
            else:
                self._ensure_spacy_loaded()
                if self.zh_sent_tokenize:
                    doc = self.zh_sent_tokenize(context)
                    raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                    sentences = self._sync_sentence(raw_sentences, context)
                else:
                    sentences = context.split('。')
                    sentences = [s.strip() + '。' for s in sentences if s.strip()]
        else:
            sentences = context.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]

        filtered_sentences = []
        for sent in sentences:
            cleaned_sent = sent.strip()
            if not cleaned_sent:
                continue
            if self._count_words_multilingual(cleaned_sent) < self.min_word_length:
                continue
            filtered_sentences.append(sent)
        return filtered_sentences

    @staticmethod
    def _context_type_has_chinese(context_type: str, sentences: List[str]) -> bool:
        if context_type == "chinese":
            return True
        return any('\u4e00' <= char <= '\u9fff' for char in ''.join(sentences[:3]))

    def _join_context_sentences(self, sentences: List[str], context_type: str) -> str:
        """Join sentences the same way chunk assembly packs them."""
        if not sentences:
            return ""
        if context_type in ("code", "fewshot"):
            return '\n\n'.join(sentences)
        if context_type == "chinese":
            if any("新闻内容：" in s for s in sentences):
                return '\n\n'.join(sentences)
            return ''.join(sentences)
        return ' '.join(sentences)

    def _map_sentences_to_offsets(
        self,
        offset_mapping: Union[np.ndarray, torch.Tensor],
        prompt: str,
        context: str,
        sentences: List[str],
        sentence_tokens: Optional[List[int]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[str], List[int]]:
        """Map pre-split sentences to token positions in an offset-tokenized prompt."""
        if not sentences:
            return [], [], []

        use_torch = isinstance(offset_mapping, torch.Tensor)

        if use_torch:
            num_sents = len(sentences)
            sent_lengths = torch.tensor(
                [len(sent) for sent in sentences],
                dtype=torch.long,
                device=offset_mapping.device
            )
            cumulative_lengths = torch.cumsum(sent_lengths, dim=0)
            start_chars = torch.zeros_like(cumulative_lengths)
            start_chars[1:] = cumulative_lengths[:-1]
            offset = prompt.find(sentences[0])
            start_chars = start_chars + offset
            end_chars = start_chars + sent_lengths
            token_starts = offset_mapping[:, 0:1]
            token_ends = offset_mapping[:, 1:2]
            start_chars_2d = start_chars.unsqueeze(0)
            end_chars_2d = end_chars.unsqueeze(0)
            valid_tokens = (token_ends > start_chars_2d) & (token_starts < end_chars_2d)
            token_sentence_pairs = valid_tokens.nonzero(as_tuple=False)
            if token_sentence_pairs.numel() > 0:
                sent_idx = token_sentence_pairs[:, 1]
                token_idx = token_sentence_pairs[:, 0]
                try:
                    sent_min = torch.full((num_sents,), 999999, device=offset_mapping.device, dtype=torch.long)
                    sent_max = torch.full((num_sents,), -1, device=offset_mapping.device, dtype=torch.long)
                    sent_min.scatter_reduce_(0, sent_idx, token_idx, reduce='amin')
                    sent_max.scatter_reduce_(0, sent_idx, token_idx, reduce='amax')
                    valid = sent_max >= 0
                    # Batch sync once (no per-element .item())
                    sent_min_list = sent_min.cpu().tolist()
                    sent_max_list = sent_max.cpu().tolist()
                    valid_list = valid.cpu().tolist()
                    sent_positions = [
                        (sent_min_list[i], sent_max_list[i]) if valid_list[i] else (0, 0)
                        for i in range(num_sents)
                    ]
                except (AttributeError, TypeError):
                    # Fallback: batch gather, one .tolist() sync (no per-element .item())
                    dev = offset_mapping.device
                    mins, maxs = [], []
                    for i in range(num_sents):
                        mask = token_sentence_pairs[:, 1] == i
                        st = token_sentence_pairs[mask, 0]
                        if st.numel() > 0:
                            mins.append(st.min().unsqueeze(0))
                            maxs.append(st.max().unsqueeze(0))
                        else:
                            mins.append(torch.tensor([0], device=dev, dtype=torch.long))
                            maxs.append(torch.tensor([0], device=dev, dtype=torch.long))
                    m = torch.cat(mins).cpu().tolist()
                    x = torch.cat(maxs).cpu().tolist()
                    sent_positions = list(zip(m, x))
            else:
                sent_positions = [(0, 0)] * num_sents
        else:
            sent_lengths = np.array([len(sent) for sent in sentences])
            cumulative_lengths = np.cumsum(sent_lengths)
            start_chars = np.zeros_like(cumulative_lengths)
            start_chars[1:] = cumulative_lengths[:-1]
            offset = prompt.find(sentences[0])
            start_chars = start_chars + offset
            end_chars = start_chars + sent_lengths
            token_starts = offset_mapping[:, 0][:, np.newaxis]
            token_ends = offset_mapping[:, 1][:, np.newaxis]
            valid_tokens = (token_ends > start_chars) & (token_starts < end_chars)
            token_sentence_pairs = np.argwhere(valid_tokens)
            sent_positions = []
            for sent_idx in range(len(sentences)):
                sent_tokens = token_sentence_pairs[token_sentence_pairs[:, 1] == sent_idx, 0]
                if len(sent_tokens) == 0:
                    sent_positions.append((0, 0))
                else:
                    sent_positions.append((
                        int(sent_tokens[0]),
                        int(sent_tokens[-1]),
                    ))

        if sentence_tokens is None:
            token_counts = self._count_sentence_tokens([s.strip() for s in sentences])
        else:
            token_counts = sentence_tokens
        valid_sentences = []
        valid_positions = []
        valid_tokens = []
        for i, count in enumerate(token_counts):
            if count > 0:
                valid_sentences.append(sentences[i])
                valid_positions.append(sent_positions[i])
                valid_tokens.append(count)
        return valid_positions, valid_sentences, valid_tokens

    def _split_into_sentences(
        self,
        offset_mapping: Union[np.ndarray, torch.Tensor],
        prompt: str,
        context: str,
        context_start: int,
        context_type: str
    ) -> Tuple[List[Tuple[int, int]], List[str], List[int]]:
        """Split context into sentences and get their token positions."""
        sentences = self._split_context_sentences(context, context_type)
        return self._map_sentences_to_offsets(offset_mapping, prompt, context, sentences)

    def _sync_sentence(self, sentences: List[str], text: str) -> List[str]:
        """Ensure split sentences match original text exactly."""
        if not sentences:
            return []
        seen_text = 0
        sentence_num = len(sentences)
        new_sentences = []
        for i, s in enumerate(sentences):
            if not s.strip():
                continue
            if i == sentence_num - 1:
                remaining_text = text[seen_text:]
                if remaining_text.strip():
                    new_sentences.append(remaining_text)
                break
            next_sentence = sentences[i + 1] if i + 1 < len(sentences) else ""
            if next_sentence:
                search_len = min(10, len(next_sentence))
                search_text = next_sentence[:search_len]
                next_sentence_start = text.find(search_text, seen_text + len(s.strip()))
                if next_sentence_start > seen_text:
                    sentence_text = text[seen_text:next_sentence_start]
                    if sentence_text.strip():
                        new_sentences.append(sentence_text)
                    seen_text = next_sentence_start
                else:
                    remaining_text = text[seen_text:]
                    if remaining_text.strip():
                        new_sentences.append(remaining_text)
                    break
        return new_sentences

    def _split_chinese_simple(self, context: str) -> List[str]:
        """Fast rule-based Chinese sentence splitting. Splits by 。？！； and newlines."""
        raw = re.split(r'[。？！；\n]+', context)
        sentences = [s.strip() for s in raw if s.strip()]
        return self._sync_sentence(sentences, context)

    def _code_sentence_split_fallback(self, code: str) -> List[str]:
        """Simple code splitting: split by newlines."""
        lines = code.split('\n')
        sentences = []
        current_block = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_block:
                    block_text = '\n'.join(current_block)
                    if self._count_words_multilingual(block_text) >= self.min_word_length:
                        sentences.append(block_text)
                    current_block = []
                continue
            current_block.append(line)
            if any(stripped.endswith(p) for p in ['{', '}', ';', ':', '"""', "'''"]):
                if len(current_block) >= 2:
                    block_text = '\n'.join(current_block)
                    if self._count_words_multilingual(block_text) >= self.min_word_length:
                        sentences.append(block_text)
                    current_block = []
        if current_block:
            block_text = '\n'.join(current_block)
            if self._count_words_multilingual(block_text) >= self.min_word_length:
                sentences.append(block_text)
        return sentences if sentences else [code]

    def _count_words_multilingual(self, text: str) -> int:
        """Count words in text, handling both English and Chinese."""
        text = text.strip()
        if not text:
            return 0
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        if has_chinese:
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            english_words = len([word for word in text.split() if any(char.isalpha() for char in word)])
            return chinese_chars + english_words
        else:
            words = text.split()
            return len([word for word in words if any(char.isalpha() for char in word)])

    def _select_sentences(
        self,
        sentences: List[str],
        scores: Union[List[float], torch.Tensor],
        sentence_tokens: List[int],
        target_tokens: int,
        context_type: str = "english"
    ) -> Tuple[str, List[int], Optional[int]]:
        """Select sentences based on importance scores and token budget."""
        if isinstance(scores, torch.Tensor):
            sorted_indices = torch.argsort(scores, descending=True).tolist()
        else:
            sorted_indices = sorted(
                range(len(sentences)), key=lambda i: scores[i], reverse=True
            )

        mandatory_indices = (
            self._mandatory_chinese_indices(sentences) if context_type == 'chinese' else []
        )
        sep_cost = self._join_separator_token_cost(context_type, sentences)

        preserved_set = set()
        current_tokens = 0
        for idx in mandatory_indices:
            tok = sentence_tokens[idx]
            extra = sep_cost if preserved_set else 0
            if current_tokens + tok + extra <= target_tokens:
                preserved_set.add(idx)
                current_tokens += tok + extra

        for idx in sorted_indices:
            if idx in preserved_set:
                continue
            extra = sep_cost if preserved_set else 0
            next_tokens = current_tokens + sentence_tokens[idx] + extra
            if next_tokens > target_tokens:
                break
            preserved_set.add(idx)
            current_tokens = next_tokens

        if not preserved_set:
            preserved_set.add(sorted_indices[0])

        if self.print_sentence_scores:
            print(f"📊 Token budget filtering: {len(preserved_set)}/{len(sentences)} sentences selected (target: {target_tokens} tokens, actual: {current_tokens} tokens)")
            print("🔥 Selected sentences ranked by score:")
            preserved_indices_dbg = sorted(preserved_set)
            if isinstance(scores, torch.Tensor):
                preserved_t = torch.tensor(
                    preserved_indices_dbg, device=scores.device, dtype=torch.long
                )
                selected_scores = scores[preserved_t].cpu().tolist()
                selected_data = list(zip(
                    preserved_indices_dbg, selected_scores,
                    [sentences[i] for i in preserved_indices_dbg],
                    [sentence_tokens[i] for i in preserved_indices_dbg],
                ))
            else:
                selected_data = [
                    (i, scores[i], sentences[i], sentence_tokens[i])
                    for i in preserved_indices_dbg
                ]
            selected_data.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, score, sentence, tokens) in enumerate(selected_data, 1):
                print(f"  {rank:2d}. [Index: {idx:2d}] [Score: {score:.3f}] [Tokens: {tokens:3d}] {sentence}")
            print()

        preserved_indices = sorted(preserved_set)
        return self._finalize_joined_selection(
            sentences,
            preserved_indices,
            scores,
            target_tokens,
            context_type,
            sentence_tokens=sentence_tokens,
        )

    def _finalize_joined_selection(
        self,
        sentences: List[str],
        preserved_indices: List[int],
        scores: Union[List[float], torch.Tensor],
        target_tokens: int,
        context_type: str,
        sentence_tokens: Optional[List[int]] = None,
    ) -> Tuple[str, List[int], Optional[int]]:
        """Join selected sentences and enforce budget on the actual joined 7B encode."""
        indices = sorted(preserved_indices)
        compressed_text = self._join_compressed_sentences(
            sentences, indices, context_type
        )
        if target_tokens <= 0 or len(indices) <= 1:
            return compressed_text, indices, None

        budget_tok = self._budget_tokenizer()
        actual = self._encode_length(budget_tok, compressed_text)
        if actual <= target_tokens:
            return compressed_text, indices, actual

        if isinstance(scores, torch.Tensor):
            score_list = scores.detach().cpu().tolist()
        else:
            score_list = list(scores)
        mandatory = set(
            self._mandatory_chinese_indices(sentences)
            if context_type == "chinese"
            else []
        )
        removable = sorted(
            (i for i in indices if i not in mandatory),
            key=lambda i: score_list[i],
        )
        sep_cost = (
            self._join_separator_token_cost(context_type, sentences)
            if sentence_tokens is not None
            else 0
        )

        drop_ptr = 0
        while actual > target_tokens and len(indices) > 1:
            while drop_ptr < len(removable) and removable[drop_ptr] not in indices:
                drop_ptr += 1
            if drop_ptr < len(removable):
                drop = removable[drop_ptr]
                drop_ptr += 1
            else:
                drop = min(indices, key=lambda i: score_list[i])
            if drop not in indices:
                break
            indices.remove(drop)
            compressed_text = self._join_compressed_sentences(
                sentences, indices, context_type
            )
            if (
                sentence_tokens is not None
                and self._estimate_joined_token_count(
                    indices, sentence_tokens, sep_cost
                )
                > target_tokens
            ):
                continue
            actual = self._encode_length(budget_tok, compressed_text)

        return compressed_text, sorted(indices), actual

    def _select_sentences_by_threshold(
        self,
        sentences: List[str],
        scores: Union[List[float], torch.Tensor],
        sentence_tokens: List[int],
        threshold: float,
        context_type: str = "english"
    ) -> Tuple[str, List[int]]:
        """Select sentences based on threshold classification."""
        mandatory_indices = (
            self._mandatory_chinese_indices(sentences) if context_type == 'chinese' else []
        )

        if isinstance(scores, torch.Tensor):
            preserved_set = set(
                (scores >= threshold).nonzero(as_tuple=True)[0].tolist()
            )
        else:
            preserved_set = {i for i, score in enumerate(scores) if score >= threshold}
        preserved_set.update(mandatory_indices)
        preserved_indices = sorted(preserved_set)
        if not preserved_indices:
            if isinstance(scores, torch.Tensor):
                # One batch sync (no .item())
                best_idx_t = scores.argmax()
                best_score_t = scores[best_idx_t]
                synced = torch.stack([best_idx_t.float(), best_score_t]).cpu().tolist()
                best_idx = int(synced[0])
                best_score = float(synced[1])
            else:
                best_idx = max(range(len(scores)), key=lambda x: scores[x])
                best_score = scores[best_idx]
            preserved_indices = [best_idx]
            print(f"⚠️  No sentences met threshold {threshold:.2f}, selecting best sentence (score: {best_score:.3f})")

        if self.print_sentence_scores:
            print(f"📊 Threshold filtering: {len(preserved_indices)}/{len(sentences)} sentences selected (threshold: {threshold})")
            print("🔥 Selected sentences ranked by score:")
            if isinstance(scores, torch.Tensor):
                preserved_t = torch.tensor(preserved_indices, device=scores.device, dtype=torch.long)
                selected_scores = scores[preserved_t].cpu().tolist()
                selected_data = list(zip(preserved_indices, selected_scores, [sentences[i] for i in preserved_indices], [sentence_tokens[i] for i in preserved_indices]))
            else:
                selected_data = [(i, scores[i], sentences[i], sentence_tokens[i]) for i in preserved_indices]
            selected_data.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, score, sentence, tokens) in enumerate(selected_data, 1):
                display_sentence = sentence[:100] + "..." if len(sentence) > 100 else sentence
                print(f"  {rank:2d}. [Index: {idx:2d}] [Score: {score:.3f}] [Tokens: {tokens:3d}] {display_sentence}")
            print()

        preserved_indices = sorted(preserved_indices)
        compressed_text = self._join_compressed_sentences(
            sentences, preserved_indices, context_type
        )
        return compressed_text, preserved_indices

    def _compressed_length_from_text(self, compressed_text: str) -> int:
        """7B token count on joined compressed text (matches LongBench eval budget)."""
        return self._encode_length(self._budget_tokenizer(), compressed_text)

    def _budget_tokenizer(self):
        """Tokenizer for per-sentence token budget (target_token selection)."""
        if self.sentence_budget_tokenizer == "0.5b":
            return self.tokenizer
        return self.eval_tokenizer

    def _count_sentence_tokens(self, sentences: List[str]) -> List[int]:
        """Count tokens per sentence for selection budget."""
        return self._batch_count_tokens(
            sentences, self._budget_tokenizer(), workers=self.sentence_tokenize_workers
        )

    @staticmethod
    def _encode_length(tokenizer, text: str) -> int:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _batch_count_tokens(
        self, sentences: List[str], tokenizer, workers: int = 0
    ) -> List[int]:
        """Token counting: single HF batch (Rust fast tokenizer) or parallel encode."""
        if not sentences:
            return []
        if workers > 1:
            return self._batch_count_tokens_parallel(sentences, tokenizer, workers)

        try:
            if hasattr(tokenizer, "__call__"):
                batch_result = tokenizer(
                    sentences,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )
                if "length" in batch_result:
                    return batch_result["length"]
                return [len(ids) for ids in batch_result["input_ids"]]
            return [self._encode_length(tokenizer, s) for s in sentences]
        except Exception:
            return [self._encode_length(tokenizer, s) for s in sentences]

    def _batch_count_tokens_parallel(
        self, sentences: List[str], tokenizer, workers: int
    ) -> List[int]:
        """Parallel per-sentence encode (HF fast/Rust tokenizer releases GIL)."""
        n_workers = min(workers, len(sentences))

        def _one(text: str) -> int:
            return self._encode_length(tokenizer, text)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(
                pool.map(
                    _one,
                    sentences,
                    chunksize=max(1, len(sentences) // (n_workers * 4)),
                )
            )

    _PROMPT_TOKEN_MARGIN = 10

    def _attention_prompt_overhead_tokens(self, question: str) -> int:
        """Template + question tokens (empty context), 0.5B attention tokenizer."""
        return self._encode_length(
            self.tokenizer, self._build_filtering_prompt("", question)
        )

    def _max_context_tokens_for_forward(
        self, question: str, ctx_budget: Optional[int] = None
    ) -> int:
        """Max context tokens per forward so prompt fits in max_seq_len."""
        if ctx_budget is not None:
            return ctx_budget
        cache_key = (int(self.max_seq_len), question)
        cached = self._ctx_budget_cache.get(cache_key)
        if cached is not None:
            return cached
        overhead = self._attention_prompt_overhead_tokens(question)
        budget = max(int(self.max_seq_len) - overhead - self._PROMPT_TOKEN_MARGIN, 1)
        self._ctx_budget_cache[cache_key] = budget
        return budget

    def _split_text_by_token_budget(
        self, text: str, token_budget: int, tokenizer=None
    ) -> List[str]:
        """Fallback when one sentence exceeds token_budget (token-window sub-chunks)."""
        tok = tokenizer or self.tokenizer
        if hasattr(tok, "encode"):
            ids = tok.encode(text, add_special_tokens=False)
        else:
            ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= token_budget:
            return [text]
        chunks = []
        for start in range(0, len(ids), token_budget):
            piece = ids[start : start + token_budget]
            if hasattr(tok, "decode"):
                chunks.append(tok.decode(piece, skip_special_tokens=True))
            else:
                chunks.append(tok.decode(piece))
        return chunks

    def _build_attention_chunk_specs(
        self,
        sentences: List[str],
        sentence_tokens: List[int],
        chunk_size: int,
        context_type: str,
    ) -> List[Dict[str, Union[str, List[str], List[int]]]]:
        """Pack pre-split sentences into forward chunks (reuses sentence lists per chunk)."""
        if not sentences:
            return []
        has_chinese = self._context_type_has_chinese(context_type, sentences)
        specs: List[Dict[str, Union[str, List[str], List[int]]]] = []
        range_start: Optional[int] = None
        current_tokens = 0

        def _append_range(start: int, end: int) -> None:
            if end <= start:
                return
            chunk_sents = sentences[start:end]
            chunk_toks = sentence_tokens[start:end]
            specs.append({
                "text": self._join_context_sentences(chunk_sents, context_type),
                "sentences": chunk_sents,
                "sentence_tokens": chunk_toks,
            })

        for idx, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
            if range_start is not None:
                if has_chinese:
                    potential_tokens = current_tokens + tokens
                else:
                    potential_tokens = current_tokens + 1 + tokens
            else:
                potential_tokens = tokens

            if potential_tokens <= chunk_size:
                if range_start is None:
                    range_start = idx
                current_tokens = potential_tokens
                continue

            if range_start is not None:
                _append_range(range_start, idx)

            if tokens > chunk_size:
                for piece in self._split_text_by_token_budget(sentence, chunk_size):
                    specs.append({"text": piece})
                range_start = None
                current_tokens = 0
            else:
                range_start = idx
                current_tokens = tokens

        if range_start is not None:
            _append_range(range_start, len(sentences))
        return specs

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """Get information about the loaded model and configuration."""
        return {
            'attention_model_path': self.attention_model_path,
            'detector_path': self.detector_path,
            'max_seq_len': self.max_seq_len,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'device': str(self.device),
            'use_threshold_by_default': self.use_threshold_by_default,
            'default_threshold': self.default_threshold,
            'min_word_length': self.min_word_length,
            'print_sentence_scores': self.print_sentence_scores,
            'disable_chunking': self.disable_chunking,
        }

    def clear_cache(self):
        """Clear GPU cache and perform garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()