#!/usr/bin/env python
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel
import hashlib
from adversarial_utils.models import generate_text, generate_text_no_retry
from adversarial_utils.utils import extract_code_from_response, validate_completion, extract_rating
import torch
import traceback


class LLMEvaluator:
    def __init__(
        self,
        logger,
        judge_model: PreTrainedModel,
        judge_tokenizer: PreTrainedTokenizer,
        completion_model: PreTrainedModel,
        completion_tokenizer: PreTrainedTokenizer,
        completion_temperature: float = 0,
        judgement_temperature: float = 0,
        completion_model_name: str = "",
        max_batch_size: int = 8,
        evalution_scale: int = 5
    ) -> None:
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.completion_model = completion_model
        self.completion_tokenizer = completion_tokenizer
        self.completion_temperature = completion_temperature
        self.judgement_temperature = judgement_temperature
        self.completion_cache: Dict[str, str] = {}
        self.judgement_cache: Dict[str, Optional[int]] = {}
        self.completion_model_name = completion_model_name
        self.max_batch_size = max_batch_size
        self.logger = logger
        self.evaluation_scale = evalution_scale
    
    def _prepare_completion_prompts(self, incomplete_codes: List[str], summarizations: List[str]) -> List[str]:
        """Prepare completion prompts with error handling."""
        prompts = []
        for incomplete_code, summarization in zip(incomplete_codes, summarizations):
            messages = [
                {"role": "system", "content": (
                    "You are a code completion assistant. You MUST implement all 'pass' statements or "
                    "incomplete functions with real code. Return ONLY the complete implementation, "
                    "no explanations or additional commentary."
                )},
                {"role": "user", "content": (
                    f"Summarization: {summarization}\n\n"
                    f"Complete the following code:\n```python\n{incomplete_code}\n```"
                )}
            ]
            
            try:
                # Apply chat template with error handling
                prompt = self.completion_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append(prompt)
            except Exception as e:
                self.logger.error(f"Error applying chat template: {e}")
                # Fallback to simple format
                fallback_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
                prompts.append(fallback_prompt)
                
        return prompts

    def _prepare_judgement_prompts(self, source_codes: List[str], summarizations: List[str], completed_codes: List[str]) -> List[str]:
        """Prepare judgement prompts."""
        prompts = []
        for source_code, summarization, completed_code in zip(source_codes, summarizations, completed_codes):
            if self.evaluation_scale == 5:
                messages = [
                    {"role": "system", "content": "You are a code evaluator."},
                    {"role": "user", "content": (
                        "Below, you are provided with the complete (ground truth) code, the human summarization "
                        "and the generated code completion candidate.\n\n"
                        "Complete code:\n"
                        f"{source_code}\n\n"
                        "Human summarization:\n"
                        f"{summarization}\n\n"
                        "Generated candidate:\n"
                        f"{completed_code}\n\n"
                        "Please evaluate the quality of the generated code completion candidate according to the following metric: "
                        "1 - Very poor: The generated code completely fails to meet the requirements. "
                        "2 - Poor: The code shows some attempt to address the requirements but has major issues. "
                        "3 - Average: The candidate meets the basic requirements. "
                        "4 - Good: The generated code is high-quality—it mostly meets the requirements and follows best practices. "
                        "5 - Excellent: The code fulfills all the requirements and is functionally equivalent to the ground truth.\n"
                        "Return your answer as a single line that starts with 'RATING:' followed by a number between 1 and 5, and nothing else."
                    )}
                ]
            elif self.evaluation_scale == 10:
                messages = [
                    {"role": "system", "content": "You are a code evaluator."},
                    {"role": "user", "content": (
                            "Below, you are provided with the complete (ground truth) code, the human summarization "
                            "and the generated code completion candidate.\n\n"
                            "Complete code:\n"
                            f"{source_code}\n\n"
                            "Human summarization:\n"
                            f"{summarization}\n\n"
                            "Generated candidate:\n"
                            f"{completed_code}\n\n"
                            "Please evaluate the quality of the generated code completion candidate according to the following metric: "
                            "1 - Extremely poor: The code is completely incorrect and does not address the requirements at all. "
                            "2 - Very poor: The code shows minimal attempt but is almost entirely wrong. "
                            "3 - Poor: The code attempts to address the requirements but has major flaws. "
                            "4 - Subpar: The code partially addresses the requirements but is missing key elements. "
                            "5 - Below average: The code meets some requirements but has significant issues. "
                            "6 - Average: The code meets the basic requirements but lacks polish or has minor issues. "
                            "7 - Above average: The code mostly meets the requirements and is generally correct. "
                            "8 - Good: The code is high-quality, meets requirements, and follows best practices. "
                            "9 - Very good: The code is nearly perfect, with only very minor issues. "
                            "10 - Excellent: The code fulfills all requirements, is functionally equivalent to the ground truth, and is exemplary.\n"
                            "Return your answer as a single line that starts with 'RATING:' followed by a number between 1 and 10, and nothing else."
                        )
                    }
                ]
            prompts.append(self.judge_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return prompts

    def generate_completions_batch(self, incomplete_codes: List[str], summarizations: List[str]) -> List[str]:
        """Generate completions with better error handling."""
        final_completions = [""] * len(incomplete_codes)
        to_generate = []
        
        # Check cache
        for i, (inc_code, summ) in enumerate(zip(incomplete_codes, summarizations)):
            cache_key = hashlib.md5((inc_code + summ + self.completion_model_name).encode()).hexdigest()
            if cache_key in self.completion_cache:
                self.logger.debug(f"Using cached completion for item {i + 1}")
                final_completions[i] = self.completion_cache[cache_key]
            else:
                to_generate.append((i, inc_code, summ, cache_key))
        
        # Generate new completions
        if to_generate:
            self.logger.info(f"Generating {len(to_generate)} completions in batches of up to {self.max_batch_size}")
            
            for batch_start in range(0, len(to_generate), self.max_batch_size):
                batch = to_generate[batch_start:batch_start + self.max_batch_size]
                batch_incomplete = [item[1] for item in batch]
                batch_summ = [item[2] for item in batch]
                
                try:
                    prompts = self._prepare_completion_prompts(batch_incomplete, batch_summ)
                    self.logger.debug(f"Prepared {len(prompts)} prompts for completion")
                    self.logger.debug(f"First prompt preview: {prompts[0][:300]}...")
                    
                    # Try generation without retry first to see actual error
                    try:
                        responses = generate_text_no_retry(
                            self.completion_model, 
                            self.completion_tokenizer,
                            prompts=prompts, 
                            temperature=self.completion_temperature, 
                            max_new_tokens=2048,
                            logger_instance=self.logger  # Pass logger instance
                        )
                        self.logger.debug(f"Got {len(responses)} responses")
                    except Exception as e:
                        self.logger.error(f"Direct generation error: {type(e).__name__}: {str(e)}")
                        # Now try with retry
                        responses = generate_text(
                            self.completion_model, 
                            self.completion_tokenizer,
                            prompts=prompts, 
                            temperature=self.completion_temperature, 
                            max_new_tokens=2048,
                            logger_instance=self.logger  # Pass logger instance
                        )
                    
                    # Process responses
                    for (idx, inc_code, _, cache_key), response in zip(batch, responses):
                        completed_code = extract_code_from_response(response)
                        is_valid = validate_completion(completed_code, inc_code)
                        
                        if is_valid:
                            final_completions[idx] = completed_code
                            self.completion_cache[cache_key] = completed_code
                            self.logger.debug(f"Valid completion for item {idx + 1}")
                        else:
                            self.logger.warning(f"Invalid completion for item {idx + 1}")
                            self.logger.warning(f"Response: {response[:300]}...")
                            self.completion_cache[cache_key] = ""
                            
                except Exception as e:
                    self.logger.error(f"Batch completion generation failed: {e}")
                    self.logger.error(f"Error type: {type(e).__name__}")
                    self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    
                    # Cache failures
                    for idx, _, _, cache_key in batch:
                        self.completion_cache[cache_key] = ""
                        
        return final_completions

    def judge_completions_batch(self, source_codes: List[str], summarizations: List[str], completed_codes: List[str]) -> List[Optional[int]]:
        """Judge completions with adaptive batch sizing for OOM handling."""
        final_ratings: List[Optional[int]] = [None] * len(source_codes)
        items_to_process = []
        
        # Check cache and prepare items
        for i, (src_code, summ, comp_code) in enumerate(zip(source_codes, summarizations, completed_codes)):
            if not comp_code:
                continue
                
            cache_key = hashlib.md5((src_code + summ + comp_code).encode()).hexdigest()
            if cache_key in self.judgement_cache:
                self.logger.debug(f"Using cached judgement for item {i + 1}")
                final_ratings[i] = self.judgement_cache[cache_key]
            else:
                items_to_process.append({
                    "idx": i, "src": src_code, "summ": summ, 
                    "comp": comp_code, "key": cache_key
                })
        
        if not items_to_process:
            return final_ratings
        
        # Process with adaptive batch sizing
        current_batch_size = self.max_batch_size
        processed = set()
        
        while len(processed) < len(items_to_process):
            # Get unprocessed items
            remaining = [item for item in items_to_process if item["idx"] not in processed]
            if not remaining:
                break
                
            batch = remaining[:current_batch_size]
            prompts = self._prepare_judgement_prompts(
                [item["src"] for item in batch],
                [item["summ"] for item in batch],
                [item["comp"] for item in batch]
            )
            
            try:
                responses = generate_text(
                    self.judge_model, 
                    self.judge_tokenizer,
                    prompts=prompts, 
                    temperature=self.judgement_temperature, 
                    max_new_tokens=64,
                    logger_instance=self.logger  # Pass logger instance
                )
                
                # Process responses
                for item, response in zip(batch, responses):
                    rating = extract_rating(response)
                    if rating:
                        rating_int = int(rating)
                        final_ratings[item["idx"]] = rating_int
                        self.judgement_cache[item["key"]] = rating_int
                    else:
                        self.judgement_cache[item["key"]] = None
                    processed.add(item["idx"])
                
                # Success - try to increase batch size
                if current_batch_size < self.max_batch_size:
                    current_batch_size = min(current_batch_size * 2, self.max_batch_size)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    self.logger.warning(f"OOM with batch size {current_batch_size}")
                    torch.cuda.empty_cache()
                    
                    if current_batch_size == 1:
                        # Mark problematic items as failed
                        for item in batch:
                            self.judgement_cache[item["key"]] = None
                            processed.add(item["idx"])
                    else:
                        # Reduce batch size
                        current_batch_size = max(1, current_batch_size // 2)
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Error in judgement batch: {e}")
                for item in batch:
                    self.judgement_cache[item["key"]] = None
                    processed.add(item["idx"])
        
        return final_ratings

    def evaluate_batch(self, incomplete_codes: List[str], source_codes: List[str], summarizations: List[str]) -> List[Optional[int]]:
        """Evaluate a batch of codes."""
        if not incomplete_codes:
            return []
        
        self.logger.info(f"Starting batch completion for {len(incomplete_codes)} items.")
        completed_codes = self.generate_completions_batch(incomplete_codes, summarizations)
        
        valid_indices = [i for i, code in enumerate(completed_codes) if code]
        if not valid_indices:
            self.logger.warning("No valid completions generated in the batch.")
            return [None] * len(incomplete_codes)
        
        # Judge only valid completions
        source_codes_for_judgement = [source_codes[i] for i in valid_indices]
        summarizations_for_judgement = [summarizations[i] for i in valid_indices]
        completed_codes_for_judgement = [completed_codes[i] for i in valid_indices]
        
        self.logger.info(f"Starting batch judgement for {len(completed_codes_for_judgement)} validly completed items.")
        ratings_for_valid = self.judge_completions_batch(
            source_codes_for_judgement, summarizations_for_judgement, completed_codes_for_judgement
        )
        
        # Map back to original indices
        final_ratings: List[Optional[int]] = [None] * len(incomplete_codes)
        for i, rating in enumerate(ratings_for_valid):
            original_idx = valid_indices[i]
            final_ratings[original_idx] = rating
            if rating is not None:
                self.logger.info(f"Final evaluation rating for item {original_idx + 1}: {rating}")
            else:
                self.logger.warning(f"Judgement failed for item {original_idx + 1}")
        
        return final_ratings

    def evaluate(self, incomplete_code: str, source_code: str, summarization: str) -> Optional[int]:
        """Evaluate a single code sample."""
        results = self.evaluate_batch([incomplete_code], [source_code], [summarization])
        return results[0] if results else None