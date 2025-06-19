#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-based Text Compressor - Clean Implementation

This module provides a specialized, clean implementation of attention-based text compression,
extracted from the original PromptCompressor for focused inference tasks.

Key Features:
- Raw attention filtering without external dependencies
- Trained detector-based compression
- Multilingual support (English, Chinese, Code)
- Optimized for inference performance
- Minimal dependencies and clean architecture

Author: Extracted from LLMLingua project
"""

import re
import json
import time
from typing import List, Tuple, Union, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
except ImportError:
    # Fallback for older transformers versions
    from transformers import AutoModelForCausalLM
    Qwen2DecoderLayer = None
import tiktoken
import joblib
import spacy
import nltk
import gc


class AttentionCompressor:
    """
    A specialized text compressor using attention mechanisms for sentence-level importance scoring.
    
    This class provides two compression modes:
    1. Raw attention filtering: Direct attention weight analysis
    2. Detector-based filtering: Uses trained classifiers on attention features
    
    Args:
        attention_model_path (str): Path to the attention model (default: Qwen2.5-0.5B-Instruct)
        detector_path (str, optional): Path to trained detector model for classification-based filtering
        use_raw_attention (bool): Whether to use raw attention weights (default: False)
        use_last_layer_only (bool): Whether to use only the last layer attention (default: False)
        use_all_queries (bool): Whether to use all query positions (default: False)
        eval_tokenizer_path (str): Tokenizer for evaluation metrics (default: Qwen2.5-7B-Instruct)
        max_seq_len (int): Maximum sequence length for chunking (default: 1024)
        device (str): Computing device (default: "cuda")
        do_selected_feature_idx (bool): Whether to use selected attention heads (default: False)
        use_threshold_by_default (bool): Whether to use threshold filtering by default (default: False)
        default_threshold (float): Default threshold value for classification filtering (default: 0.5)
        min_word_length (int): Minimum word/character count for sentence filtering (default: 5)
        print_sentence_scores (bool): Whether to print sentence scores and selection details (default: False)
    """

    def __init__(
        self,
        attention_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        detector_path: Optional[str] = None,
        use_raw_attention: bool = False,
        use_last_layer_only: bool = False,
        use_all_queries: bool = False,
        eval_tokenizer_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_seq_len: int = 1024,
        device: str = "cuda",
        do_selected_feature_idx: bool = False,
        use_threshold_by_default: bool = False,
        default_threshold: float = 0.5,
        min_word_length: int = 5,
        print_sentence_scores: bool = True,
    ):
        self.attention_model_path = attention_model_path
        self.detector_path = detector_path
        self.use_raw_attention = use_raw_attention
        self.use_last_layer_only = use_last_layer_only
        self.use_all_queries = use_all_queries
        self.max_seq_len = max_seq_len
        self.do_selected_feature_idx = do_selected_feature_idx
        self.use_threshold_by_default = use_threshold_by_default
        self.default_threshold = default_threshold
        self.min_word_length = min_word_length
        self.print_sentence_scores = print_sentence_scores
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load models and tokenizers
        self._load_attention_model()
        self._load_eval_tokenizer(eval_tokenizer_path)
        self._load_detector()
        self._setup_text_processing()
        
        print(f"AttentionCompressor initialized:")
        print(f"  - Model: {attention_model_path}")
        print(f"  - Mode: {'Raw attention' if use_raw_attention else 'Detector-based'}")
        print(f"  - Layers: {'Last only' if use_last_layer_only else 'All layers'}")
        print(f"  - Queries: {'All positions' if use_all_queries else 'Last token only'}")
        print(f"  - Max_seq_len: {self.max_seq_len}")
        print(f"  - Min word length: {self.min_word_length}")
        print(f"  - Device: {self.device}")
        if self.use_threshold_by_default:
            print(f"  - Threshold filtering enabled by default: {self.default_threshold}")
        if self.print_sentence_scores:
            print(f"  - Sentence score printing: enabled")

    def _load_attention_model(self):
        """Load the attention model and tokenizer"""
        print(f"Loading attention model from: {self.attention_model_path}")
        
        # Determine model precision based on model size
        if "0.5" in self.attention_model_path or "0.5B" in self.attention_model_path:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        # Load model
        self.attention_model = AutoModelForCausalLM.from_pretrained(
            self.attention_model_path,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        ).eval()
        self.attention_model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.attention_model_path)
        
        # Get model config
        config = self.attention_model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        
        # Setup attention capturing if needed
        if not self.use_raw_attention:
            self._setup_attention_capture()

    def _load_eval_tokenizer(self, eval_tokenizer_path: str):
        """Load evaluation tokenizer for token counting"""
        if eval_tokenizer_path == 'use_gpt':
            self.eval_tokenizer = tiktoken.get_encoding('cl100k_base')
        else:
            self.eval_tokenizer = AutoTokenizer.from_pretrained(
                eval_tokenizer_path, use_fast=True
            )

    def _load_detector(self):
        """Load trained detector for classification-based filtering"""
        if not self.use_raw_attention and self.detector_path:
            print(f"Loading detector from: {self.detector_path}")
            self.detector = joblib.load(self.detector_path)
        else:
            self.detector = None

    def _setup_text_processing(self):
        """Setup text processing tools for different languages"""
        # Chinese sentence tokenizer
        try:
            self.zh_sent_tokenize = spacy.load(
                "zh_core_web_sm", 
                disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
            )
        except OSError:
            print("⚠️  zh_core_web_sm not found, Chinese text processing may be limited")
            self.zh_sent_tokenize = None
        
        if self.zh_sent_tokenize:
            self.zh_sent_tokenize.add_pipe('sentencizer')

    def _count_words_multilingual(self, text: str) -> int:
        """
        Count words in text, handling both English and Chinese
        
        Args:
            text: Input text to count words
            
        Returns:
            Number of words/characters in the text
        """
        text = text.strip()
        if not text:
            return 0
        
        # Check if text contains Chinese characters
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        
        if has_chinese:
            # For Chinese text, count characters (excluding punctuation and spaces)
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            # Also count English words in mixed text
            english_words = len([word for word in text.split() if any(char.isalpha() for char in word)])
            return chinese_chars + english_words
        else:
            # For English text, count words
            words = text.split()
            return len([word for word in words if any(char.isalpha() for char in word)])

    def _setup_attention_capture(self):
        """Setup attention weight capturing for detector-based filtering"""
        self.attn_tap = []
        
        if self.do_selected_feature_idx:
            # Load selected heads configuration
            json_path = self.detector_path.replace('.pkl', '.json')
            with open(json_path, 'r') as f:
                selected_heads = json.load(f)
            selected_heads = {int(k): v for k, v in selected_heads.items()}
            self._patch_model_for_selected_heads(selected_heads)
        else:
            # Patch for all layers, last token attention
            self._patch_model_for_last_token()
            
        # Test the patching with dummy input
        dummy_inputs = self.tokenizer("hello world", return_tensors="pt").to(self.device)
        self.attn_tap.clear()
        with torch.no_grad():
            _ = self.attention_model(**dummy_inputs, output_attentions=True, return_dict=True)
        print(f"Attention capture setup complete. Captured: {len(self.attn_tap)} attention tensors")

    def _patch_model_for_selected_heads(self, selected_heads: Dict[int, List[int]]):
        """Patch model to capture attention from selected heads only"""
        layer_counter = 0
        
        for module in self.attention_model.modules():
            if isinstance(module, Qwen2DecoderLayer):
                original_forward = module.forward

                def make_forward(original_forward, layer_idx, compressor_ref):
                    def patched_forward(self, *args, **kwargs):
                        outputs = original_forward(*args, **kwargs)
                        
                        if len(outputs) > 1 and outputs[1] is not None:
                            attn = outputs[1]
                            if hasattr(attn, 'shape') and layer_idx in selected_heads:
                                for h in selected_heads[layer_idx]:
                                    if h < attn.shape[1]:
                                        compressor_ref.attn_tap.append((layer_idx, h, attn[:, h, -1, :]))
                            
                            # Clear attention to save memory
                            if isinstance(outputs, tuple):
                                outputs = (outputs[0], None) + outputs[2:]
                            elif isinstance(outputs, list):
                                outputs[1] = None
                        return outputs
                    return patched_forward

                module.forward = make_forward(original_forward, layer_counter, self).__get__(module, module.__class__)
                layer_counter += 1

    def _patch_model_for_last_token(self):
        """Patch model to capture last token attention from all layers"""
        layer_counter = 0
        
        for module in self.attention_model.modules():
            if isinstance(module, Qwen2DecoderLayer):
                original_forward = module.forward

                def make_forward(original_forward, layer_idx, compressor_ref):
                    def patched_forward(self, *args, **kwargs):
                        outputs = original_forward(*args, **kwargs)
                        
                        if len(outputs) > 1 and outputs[1] is not None:
                            attn = outputs[1]  # [B, H, S, S]
                            if hasattr(attn, 'shape'):
                                # Extract last token attention: [B, H, S]
                                last_token_attn = attn[:, :, -1, :]
                                compressor_ref.attn_tap.append((layer_idx, last_token_attn))
                            
                            # Clear attention to save memory
                            if isinstance(outputs, tuple):
                                outputs = (outputs[0], None) + outputs[2:]
                            elif isinstance(outputs, list):
                                outputs[1] = None
                        return outputs
                    return patched_forward

                module.forward = make_forward(original_forward, layer_counter, self).__get__(module, module.__class__)
                layer_counter += 1

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
        Compress text using attention-based filtering
        
        Args:
            context (str): Input text to compress
            question (str): Question for context-aware compression
            target_token (int): Target number of tokens (-1 to use compression_rate, default: -1)
            compression_rate (float): Compression rate (0.0-1.0, ignored if target_token > 0)
            context_type (str): Type of context ("english", "chinese", or "code")
            use_threshold_filtering (bool): Whether to use threshold-based filtering instead of rate/token-based
            threshold (float): Threshold for classification (default: 0.5)
            
        Returns:
            Dict containing:
                - compressed_text: The compressed text
                - original_length: Original text length in tokens
                - compressed_length: Compressed text length in tokens
                - compression_ratio: Actual compression ratio achieved
                - sentence_scores: Importance scores for each sentence
                - processing_time: Time taken for compression
        """
        start_time = time.time()
        
        # Get sentence scores using attention filtering
        sentence_scores, sentences, sentence_tokens = self._get_sentence_scores(
            context, question, context_type
        )
        
        # Calculate target tokens or use threshold filtering
        total_tokens = sum(sentence_tokens)
        
        # Check if we should use threshold filtering by default
        if self.use_threshold_by_default and not use_threshold_filtering:
            use_threshold_filtering = True
            threshold = self.default_threshold
            
        if use_threshold_filtering:
            # Use threshold-based filtering
            compressed_text, preserved_indices = self._select_sentences_by_threshold(
                sentences, sentence_scores, sentence_tokens, threshold
            )
            target_tokens = sum(sentence_tokens[i] for i in preserved_indices)
            actual_compression_rate = 1.0 - (target_tokens / total_tokens) if total_tokens > 0 else 0.0
        else:
            # Use token budget-based filtering
            if target_token > 0:
                target_tokens = min(target_token, total_tokens)
                actual_compression_rate = 1.0 - (target_tokens / total_tokens)
            else:
                target_tokens = int(total_tokens * (1 - compression_rate))
                actual_compression_rate = compression_rate
                
            # Select sentences based on importance scores
            compressed_text, preserved_indices = self._select_sentences(
                sentences, sentence_scores, sentence_tokens, target_tokens
            )
        
        # Calculate final metrics
        if hasattr(self.eval_tokenizer, 'encode'):
            compressed_tokens = len(self.eval_tokenizer.encode(compressed_text))
        else:
            compressed_tokens = len(self.eval_tokenizer(compressed_text)['input_ids'])
            
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

    def _get_sentence_scores(
        self, 
        context: str, 
        question: str, 
        context_type: str
    ) -> Tuple[List[float], List[str], List[int]]:
        """Get importance scores for each sentence using attention filtering"""
        
        if self.use_raw_attention:
            return self._raw_attention_filtering(context, question, context_type)
        else:
            return self._detector_based_filtering(context, question, context_type)

    def _raw_attention_filtering(
        self, 
        context: str, 
        question: str, 
        context_type: str
    ) -> Tuple[List[float], List[str], List[int]]:
        """Use raw attention weights to score sentence importance"""
        
        # Create prompt
        prompt = f"Given the following information: {context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
        
        # Find context position in tokens
        context_start, context_end = self._find_context_position(offset_mapping, prompt, context)
        
        # Split context into sentences
        sent_positions, sentences, sentence_tokens = self._split_into_sentences(
            offset_mapping, prompt, context, context_start, context_type
        )
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.attention_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )
            
            # Process attention weights
            if self.use_last_layer_only:
                token_attentions = [outputs.attentions[-1]]
            else:
                token_attentions = outputs.attentions
                
            stacked_attentions = torch.stack(token_attentions)
            
            # Extract relevant attention
            if self.use_all_queries:
                # Find question start position
                if question:
                    question_start, _ = self._find_context_position(offset_mapping, prompt, question)
                else:
                    question_start = context_end + 1
                    
                valid_positions = torch.arange(question_start, len(inputs['input_ids'][0]), device=self.device)
                all_layers_attentions = stacked_attentions[:, 0, :, valid_positions, context_start:context_end+1]
                all_layers_attentions = all_layers_attentions.mean(dim=2)
            else:
                all_layers_attentions = stacked_attentions[:, 0, :, -1, context_start:context_end+1]
            
            # Compute attention ratios
            head_attentions = all_layers_attentions.view(-1, context_end - context_start + 1)
            attention_sums = torch.sum(head_attentions, dim=1, keepdim=True).clamp_(min=1e-8)
            attention_ratio = head_attentions.div_(attention_sums).transpose(0, 1)
            attention_ratio = torch.nan_to_num(attention_ratio, 0.0)
            
            # Calculate sentence scores
            sentence_scores = []
            for start, end in sent_positions:
                sent_attention = torch.nanmean(attention_ratio[start:end + 1], dim=0)
                sent_attention = torch.nan_to_num(sent_attention, 0.0)
                score = torch.mean(sent_attention).item()
                sentence_scores.append(score)
        
        # Normalize scores
        if sentence_scores:
            max_score, min_score = max(sentence_scores), min(sentence_scores)
            score_range = max_score - min_score
            if score_range > 0:
                sentence_scores = [(score - min_score) / score_range for score in sentence_scores]
        
        return sentence_scores, sentences, sentence_tokens

    def _detector_based_filtering(
        self, 
        context: str, 
        question: str, 
        context_type: str
    ) -> Tuple[List[float], List[str], List[int]]:
        """Use trained detector to score sentence importance"""
        
        if not self.detector:
            raise ValueError("Detector not loaded. Cannot perform detector-based filtering.")
        
        # Create prompt  
        prompt = f"Given the following information: {context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
        
        # Find context position
        context_start, context_end = self._find_context_position(offset_mapping, prompt, context)
        
        # Split context into sentences
        sent_positions, sentences, sentence_tokens = self._split_into_sentences(
            offset_mapping, prompt, context, context_start, context_type
        )
        
        # Extract attention features
        with torch.no_grad():
            if self.do_selected_feature_idx and hasattr(self, "attn_tap"):
                # Use selected heads
                self.attn_tap.clear()
                _ = self.attention_model(**inputs, output_attentions=True, return_dict=True)
                
                # Process captured attention
                self.attn_tap.sort(key=lambda x: (x[0], x[1]))
                attn_tensor = torch.stack([x[2] for x in self.attn_tap], dim=0)
                attn_tensor = attn_tensor.mean(dim=1)
                attn_sums = attn_tensor.sum(dim=1, keepdim=True).clamp(min=1e-8)
                attn_ratio = (attn_tensor / attn_sums).transpose(0, 1)
                attn_ratio = torch.nan_to_num(attn_ratio, 0.0)
                
                # Extract sentence features
                indices = torch.cat([
                    torch.arange(start, end + 1, device=attn_ratio.device) 
                    for start, end in sent_positions
                ])
                sent_lengths = [end - start + 1 for start, end in sent_positions]
                all_features = attn_ratio[indices]
                features_split = torch.split(all_features, sent_lengths)
                vectors = torch.vstack([torch.nanmean(feat, dim=0) for feat in features_split])
                
            else:
                # Use standard attention extraction
                self.attn_tap.clear()
                _ = self.attention_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True,
                )
                
                # Process attention tensors
                attn_tensors = [x[1] for x in sorted(self.attn_tap, key=lambda x: x[0])]
                attn_tensors = [a.unsqueeze(2) for a in attn_tensors]
                
                if self.use_last_layer_only:
                    stacked_attentions = attn_tensors[-1].unsqueeze(0)
                else:
                    stacked_attentions = torch.stack(attn_tensors)
                
                # Extract query-specific attention
                if self.use_all_queries:
                    if question:
                        question_start, _ = self._find_context_position(offset_mapping, prompt, question)
                    else:
                        question_start = context_end + 1
                    valid_positions = torch.arange(question_start, len(inputs['input_ids'][0]), device=stacked_attentions.device)
                    all_layers_attentions = stacked_attentions[:, 0, :, valid_positions, context_start:context_end+1]
                    all_layers_attentions = all_layers_attentions.mean(dim=2)
                else:
                    all_layers_attentions = stacked_attentions[:, 0, :, -1, context_start:context_end+1]
                
                head_attentions = all_layers_attentions.view(-1, context_end - context_start + 1)
                attention_sums = torch.sum(head_attentions, dim=1, keepdim=True).clamp_(min=1e-8)
                attention_ratio = head_attentions.div_(attention_sums).transpose(0, 1)
                attention_ratio = torch.nan_to_num(attention_ratio, 0.0)
                
                # Extract sentence features
                indices = torch.cat([
                    torch.arange(start, end + 1, device=stacked_attentions.device) 
                    for start, end in sent_positions
                ])
                sent_lengths = [end - start + 1 for start, end in sent_positions]
                all_features = attention_ratio[indices]
                features_split = torch.split(all_features, sent_lengths)
                vectors = torch.vstack([torch.nanmean(feat, dim=0) for feat in features_split])
        
        # Get detector predictions
        feature_vectors = vectors.cpu().numpy()
        sentence_probs = self.detector.predict_proba(feature_vectors)[:, 1]
        
        return sentence_probs.tolist(), sentences, sentence_tokens

    def _find_context_position(self, offset_mapping: np.ndarray, prompt: str, context: str) -> Tuple[int, int]:
        """Find context position in token sequence"""
        start_char = prompt.find(context)
        end_char = start_char + len(context)
        
        token_starts = offset_mapping[:, 0]
        token_ends = offset_mapping[:, 1]
        
        valid_tokens = (token_ends > start_char) & (token_starts < end_char)
        token_indices = np.nonzero(valid_tokens)[0]
        
        return token_indices[0], token_indices[-1]

    def _split_into_sentences(
        self, 
        offset_mapping: np.ndarray, 
        prompt: str, 
        context: str, 
        context_start: int, 
        context_type: str
    ) -> Tuple[List[Tuple[int, int]], List[str], List[int]]:
        """Split context into sentences and get their token positions"""
        
        # Split into sentences based on language
        if context_type == 'english' or context_type == 'code':
            # Filter empty sentences from nltk tokenization
            raw_sentences = nltk.sent_tokenize(context)
            raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
            sentences = self._sync_sentence(raw_sentences, context)
        elif context_type == 'chinese':
            if self.zh_sent_tokenize:
                doc = self.zh_sent_tokenize(context)
                raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                sentences = self._sync_sentence(raw_sentences, context)
            else:
                # Fallback to simple splitting if spacy model not available
                sentences = context.split('。')  # Split by Chinese period
                sentences = [s.strip() + '。' for s in sentences if s.strip()]
        else:
            # Default to simple splitting for unknown types
            sentences = context.split('\n\n')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out empty, whitespace-only, or too short sentences
        filtered_sentences = []
        for sent in sentences:
            # Strip whitespace and check if sentence has meaningful content
            cleaned_sent = sent.strip()
            if not cleaned_sent:
                continue
                
            # Check word/character length (minimum threshold words/characters)
            word_count = self._count_words_multilingual(cleaned_sent)
            if word_count < self.min_word_length:
                continue
                
            filtered_sentences.append(sent)  # Keep original format (with potential leading/trailing spaces)
        
        sentences = filtered_sentences
        
        # Early return if no valid sentences found
        if not sentences:
            return [], [], []
        
        # Calculate sentence positions in prompt
        sent_lengths = np.array([len(sent) for sent in sentences])
        cumulative_lengths = np.cumsum(sent_lengths)
        start_chars = np.zeros_like(cumulative_lengths)
        start_chars[1:] = cumulative_lengths[:-1]
        
        # Offset by context start in prompt
        offset = prompt.find(sentences[0])
        start_chars = start_chars + offset
        end_chars = start_chars + sent_lengths
        
        # Map to token positions
        token_starts = offset_mapping[:, 0][:, np.newaxis]
        token_ends = offset_mapping[:, 1][:, np.newaxis]
        
        valid_tokens = (token_ends > start_chars) & (token_starts < end_chars)
        token_sentence_pairs = np.argwhere(valid_tokens)
        
        # Get sentence token ranges
        sent_positions = []
        for sent_idx in range(len(sentences)):
            sent_tokens = token_sentence_pairs[token_sentence_pairs[:, 1] == sent_idx, 0]
            sent_positions.append((
                sent_tokens[0] - context_start,
                sent_tokens[-1] - context_start
            ))
        
        # Count tokens per sentence and filter out sentences with no tokens
        sentence_tokens = []
        valid_sentences = []
        valid_positions = []
        
        for i, sent in enumerate(sentences):
            if hasattr(self.eval_tokenizer, 'encode'):
                token_count = len(self.eval_tokenizer.encode(sent.strip()))
            else:
                token_count = len(self.eval_tokenizer(sent.strip())['input_ids'])
            
            # Only keep sentences that have at least 1 token
            if token_count > 0:
                sentence_tokens.append(token_count)
                valid_sentences.append(sent)
                valid_positions.append(sent_positions[i])
        
        # Update the lists to only include valid sentences
        sentences = valid_sentences
        sent_positions = valid_positions
        
        return sent_positions, sentences, sentence_tokens

    def _sync_sentence(self, sentences: List[str], text: str) -> List[str]:
        """Ensure split sentences match original text exactly"""
        if not sentences:
            return []
            
        seen_text = 0
        sentence_num = len(sentences)
        new_sentences = []
        
        for i, s in enumerate(sentences):
            # Skip if input sentence is empty
            if not s.strip():
                continue
                
            if i == sentence_num - 1:
                remaining_text = text[seen_text:]
                if remaining_text.strip():  # Only add if not empty
                    new_sentences.append(remaining_text)
                break
            
            # Find next sentence, handle cases where sentence might be modified
            next_sentence = sentences[i + 1] if i + 1 < len(sentences) else ""
            if next_sentence:
                # Use first few chars to find position, handle short sentences
                search_len = min(10, len(next_sentence))
                search_text = next_sentence[:search_len]
                next_sentence_start = text.find(search_text, seen_text + len(s.strip()))
                
                if next_sentence_start > seen_text:  # Ensure we have valid content
                    sentence_text = text[seen_text:next_sentence_start]
                    if sentence_text.strip():  # Only add if not empty
                        new_sentences.append(sentence_text)
                    seen_text = next_sentence_start
                else:
                    # Fallback: if can't find next sentence, take remaining text
                    remaining_text = text[seen_text:]
                    if remaining_text.strip():
                        new_sentences.append(remaining_text)
                    break
            
        return new_sentences

    def _select_sentences(
        self, 
        sentences: List[str], 
        scores: List[float], 
        sentence_tokens: List[int], 
        target_tokens: int
    ) -> Tuple[str, List[int]]:
        """Select sentences based on importance scores and token budget"""
        
        # Sort sentences by importance score
        sent_indices = list(range(len(sentences)))
        sorted_indices = sorted(sent_indices, key=lambda x: scores[x], reverse=True)
        
        # Greedily add sentences until target token count
        preserved_indices = []
        current_tokens = 0
        
        for idx in sorted_indices:
            next_tokens = current_tokens + sentence_tokens[idx]
            if next_tokens > target_tokens:
                break
            preserved_indices.append(idx)
            current_tokens = next_tokens
        
        # Ensure at least one sentence is preserved
        if not preserved_indices:
            preserved_indices = [sorted_indices[0]]
        
        # Print selected sentences sorted by score (high to low) if enabled
        if self.print_sentence_scores:
            print(f"📊 Token budget filtering: {len(preserved_indices)}/{len(sentences)} sentences selected (target: {target_tokens} tokens, actual: {current_tokens} tokens)")
            print("🔥 Selected sentences ranked by score:")
            
            # Create list of (index, score, sentence, tokens) for selected sentences
            selected_data = [(i, scores[i], sentences[i], sentence_tokens[i]) for i in preserved_indices]
            # Sort by score (descending)
            selected_data.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (idx, score, sentence, tokens) in enumerate(selected_data, 1):
                # Truncate sentence if too long
                display_sentence = sentence
                print(f"  {rank:2d}. [Index: {idx:2d}] [Score: {score:.3f}] [Tokens: {tokens:3d}] {display_sentence}")
            
            print()  # Add blank line for readability
        
        # Sort preserved indices to maintain original order
        preserved_indices = sorted(preserved_indices)
        
        # Create compressed text
        compressed_sentences = [sentences[i] for i in preserved_indices]
        compressed_text = ' '.join(compressed_sentences)
        
        return compressed_text, preserved_indices

    def _select_sentences_by_threshold(
        self, 
        sentences: List[str], 
        scores: List[float], 
        sentence_tokens: List[int], 
        threshold: float
    ) -> Tuple[str, List[int]]:
        """Select sentences based on threshold classification"""
        
        # Select sentences with scores >= threshold
        preserved_indices = []
        for i, score in enumerate(scores):
            if score >= threshold:
                preserved_indices.append(i)
        
        # Ensure at least one sentence is preserved if none meet threshold
        if not preserved_indices:
            # Select the sentence with highest score
            best_idx = max(range(len(scores)), key=lambda x: scores[x])
            preserved_indices = [best_idx]
            print(f"⚠️  No sentences met threshold {threshold:.2f}, selecting best sentence (score: {scores[best_idx]:.3f})")
        
        # Print selected sentences sorted by score (high to low) if enabled
        if self.print_sentence_scores:
            print(f"📊 Threshold filtering: {len(preserved_indices)}/{len(sentences)} sentences selected (threshold: {threshold})")
            print("🔥 Selected sentences ranked by score:")
            
            # Create list of (index, score, sentence, tokens) for selected sentences
            selected_data = [(i, scores[i], sentences[i], sentence_tokens[i]) for i in preserved_indices]
            # Sort by score (descending)
            selected_data.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (idx, score, sentence, tokens) in enumerate(selected_data, 1):
                # Truncate sentence if too long
                display_sentence = sentence[:100] + "..." if len(sentence) > 100 else sentence
                print(f"  {rank:2d}. [Index: {idx:2d}] [Score: {score:.3f}] [Tokens: {tokens:3d}] {display_sentence}")
            
            print()  # Add blank line for readability
        
        # Sort preserved indices to maintain original order
        preserved_indices = sorted(preserved_indices)
        
        # Create compressed text
        compressed_sentences = [sentences[i] for i in preserved_indices]
        compressed_text = ' '.join(compressed_sentences)
        
        return compressed_text, preserved_indices

    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        """Get information about the loaded model and configuration"""
        return {
            'attention_model_path': self.attention_model_path,
            'detector_path': self.detector_path,
            'use_raw_attention': self.use_raw_attention,
            'use_last_layer_only': self.use_last_layer_only,
            'use_all_queries': self.use_all_queries,
            'max_seq_len': self.max_seq_len,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'device': str(self.device),
            'do_selected_feature_idx': self.do_selected_feature_idx,
            'use_threshold_by_default': self.use_threshold_by_default,
            'default_threshold': self.default_threshold,
            'min_word_length': self.min_word_length,
            'print_sentence_scores': self.print_sentence_scores,
        }

    def clear_cache(self):
        """Clear GPU cache and perform garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
    """
    Split long text into overlapping chunks for processing
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence boundary
        chunk = text[start:end]
        last_period = chunk.rfind('.')
        last_newline = chunk.rfind('\n')
        break_point = max(last_period, last_newline)
        
        if break_point > start + chunk_size // 2:
            chunks.append(text[start:break_point + 1])
            start = break_point + 1 - overlap
        else:
            chunks.append(chunk)
            start = end - overlap
    
    return chunks


