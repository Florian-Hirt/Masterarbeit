#!/usr/bin/env python
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import json
import argparse
import time
import torch
import logging
import warnings
import ast
import gast
import networkx as nx
from networkx import DiGraph
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, Gemma3ForCausalLM, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re
import hashlib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
from accelerate.utils import find_executable_batch_size
import torch._dynamo


# Import existing modules
# Import from perturbation.py
from perturbation_backup2 import (
    non_deterministic_topological_sort, neighborhood_topological_sort,
    graph_to_ast, remove_next_syntax_edges_until_first_function_call, remove_last_reads,
    ASTOrder, remove_cfg_next_edges_between_functions, add_import_dependencies,
    add_control_block_dependencies)


from code_completion_generation import incomplete_code_transform

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch._dynamo.config.disable = True
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Type definitions for the custom graph structure from the perturbation module
class Edge:
    id1: str
    id2: str

class Graph:
    nodes: Dict[str, Any]
    edges: List[Edge]

NodeID = str
TopoSort = List[NodeID]

class AdversarialReorderingSearch:
    def __init__(
        self,
        evaluation_function_batch: Callable[[List[str], List[str], List[str]], List[Optional[int]]],
        initial_samples: int = 10,
        neighborhood_samples: int = 5,
        iterations: int = 3,
        change_probability: float = 0.3,
    ) -> None:
        self.evaluation_function_batch = evaluation_function_batch
        self.initial_samples = initial_samples
        self.neighborhood_samples = neighborhood_samples
        self.iterations = iterations
        self.change_probability = change_probability
        self.results_cache: Dict[str, Tuple[Optional[float], str]] = {}

    def reset_cache(self) -> None:
        self.results_cache = {}

    def get_reordered_code(
        self,
        original_code_for_fallback: str,
        topo_sort: Optional[TopoSort] = None,
        base_graph: Optional[Graph] = None,
        base_ast_tree: Optional[ast.AST] = None
    ) -> Tuple[str, Optional[Graph], Optional[ast.AST], Optional[DiGraph]]:
        if base_graph is None or base_ast_tree is None:
            try:
                from digraph_transformer import dataflow_parser
                logger.debug(f"Generating base graph/AST from: {original_code_for_fallback[:100]}...")
                current_graph, current_ast = dataflow_parser.get_program_graph(original_code_for_fallback)

                if current_graph is None or current_ast is None:
                    logger.warning("Failed to generate base graph/AST.")
                    return original_code_for_fallback, None, None, None

                remove_next_syntax_edges_until_first_function_call(current_graph, current_ast)
                remove_last_reads(current_graph, current_ast)
                ast_order = ASTOrder(current_graph)
                ast_order.visit(current_ast)
                ast_order.reorder_graph()
                remove_cfg_next_edges_between_functions(current_graph)
                add_import_dependencies(current_graph, current_ast)
                add_control_block_dependencies(current_graph)
                
                base_graph = current_graph
                base_ast_tree = current_ast
            except Exception as e_init:
                logger.error(f"Error during initial graph/AST creation: {e_init}")
                logger.error(traceback.format_exc())
                return original_code_for_fallback, None, None, None

        try:
            nx_graph = nx.DiGraph()
            if base_graph and base_graph.nodes:
                nx_graph.add_nodes_from(base_graph.nodes.keys())
            if base_graph and base_graph.edges:
                edges = [(edge.id1, edge.id2) for edge in base_graph.edges]
                nx_graph.add_edges_from(edges)

            if not list(nx_graph.nodes):
                logger.warning("NX graph (from base_graph) is empty.")
                return original_code_for_fallback, base_graph, base_ast_tree, None

            if not nx.is_directed_acyclic_graph(nx_graph):
                logger.warning("Graph (base) contains cycles.")
                return original_code_for_fallback, base_graph, base_ast_tree, nx_graph

            if topo_sort is None:
                topo_sort = non_deterministic_topological_sort(nx_graph)

            reconstructed_gast_ast, _ = graph_to_ast(base_graph, topo_sort, show_reorderings=False)
            


            try:
                ast_for_unparse = gast.gast_to_ast(reconstructed_gast_ast)
                
                import astunparse
                reordered_code_candidate = astunparse.unparse(ast_for_unparse)
                
                
                try:
                    ast.parse(reordered_code_candidate) # This will raise SyntaxError if invalid
                except SyntaxError as e_syn_debug:
                    logger.error(f"DEBUG: Syntax error for sample being processed.")
                    logger.error(f"DEBUG: Original code (first 300 chars): {original_code_for_fallback[:300]}")
                    logger.error(f"DEBUG: Topological sort (first 10 elements): {str(topo_sort)[:200]}") # Log part of the topo_sort
                    logger.error(f"DEBUG: Problematic reordered_code_candidate that failed ast.parse:\n<<<<<<<<<< CODE START >>>>>>>>>>\n{reordered_code_candidate}\n<<<<<<<<<<< CODE END >>>>>>>>>>>>")
                    logger.error(f"DEBUG: GAST tree that was unparsed (root type): {type(reconstructed_gast_ast)}")
                    # You could even try to pretty-print parts of 'reconstructed_gast_ast_reordered_modules' or 'ast_for_unparse'
                    # using ast.dump(ast_for_unparse, indent=2) but this can be VERY verbose.
                    raise e_syn_debug # Re-raise the original exception

                return reordered_code_candidate, base_graph, base_ast_tree, nx_graph
            
            except KeyError as e_key:
                logger.error(f"astunparse KeyError: {e_key.args[0]}")
                if hasattr(e_key.args[0], '__name__') and e_key.args[0].__name__ == 'Or':
                    logger.error(f"KeyError was for ast.Or.")
                logger.error(traceback.format_exc())
                return original_code_for_fallback, base_graph, base_ast_tree, nx_graph
            except SyntaxError as e_syn:
                logger.warning(f"Syntax error in astunparse'd reordered code (topo_sort: {str(topo_sort)[:50]}...): {e_syn}")
                return original_code_for_fallback, base_graph, base_ast_tree, nx_graph
            except Exception as e_reorder:
                logger.error(f"General error during reordering/unparsing: {e_reorder}")
                logger.error(traceback.format_exc())
                return original_code_for_fallback, base_graph, base_ast_tree, nx_graph
        except Exception as e:
            logger.error(e)


    def search(
        self,
        original_code_param: str,
        source_code_for_eval: str,
        summarization_for_eval: str
    ) -> Tuple[str, float, List[int], Dict[str, int]]:
        self.reset_cache()

        best_score: float = float('inf')
        best_code: str = original_code_param
        best_topo_sort: Optional[TopoSort] = None
        all_scores_valid: List[int] = []
        stats: Dict[str, Union[int, float]] = {
            "total_reordering_attempts": 0,
            "unique_reorderings_generated": 0,
            "failed_reorderings": 0,
            "transform_to_incomplete_failed": 0,
            "evaluation_attempts_for_new_reorderings": 0,
            "successful_llm_evaluations": 0,
            "failed_llm_evaluations": 0,
            "cache_hits_reordering_score": 0,
            "cache_hits_reordering_failed_eval": 0,
            "total_topo_sorts_generated": 0,
            "unique_perturbations": 0
        }

        logger.info("Initializing graph and AST for the original code...")
        _, base_graph, base_ast_tree, base_nx_graph = self.get_reordered_code(original_code_param) # Initial call to set up base structures
        if base_graph is None or base_ast_tree is None or base_nx_graph is None:
            logger.warning("Failed to initialize base graph and AST from original_code for search.")
            stats["unique_perturbations"] = 0 # Ensure key exists
            return original_code_param, best_score, all_scores_valid, stats

        try:
            ast.parse(original_code_param)
        except SyntaxError:
            logger.error("The original_code_param itself has a syntax error. Cannot proceed.")
            return original_code_param, float('inf'), [], stats
        
        unique_reorderings_generated_hashes = set()

        # --- Phase 1: Initial random sampling (Batched) ---
        logger.info("Phase 1: Initial random sampling (batched)...")
        topo_sorts_phase1 = []
        for _ in range(self.initial_samples):
            stats["total_topo_sorts_generated"] += 1
            topo_sorts_phase1.append(non_deterministic_topological_sort(base_nx_graph))

        incomplete_codes_batch: List[str] = []
        reordered_codes_batch_for_llm: List[str] = []
        topo_sorts_for_llm_batch_phase1: List[TopoSort] = []

        for i, topo_sort in enumerate(topo_sorts_phase1):
            stats["total_reordering_attempts"] += 1
            reordered_code, _, _, _ = self.get_reordered_code(original_code_param, topo_sort, base_graph, base_ast_tree)
            
            if reordered_code is None or reordered_code == original_code_param: # Also check against original_code_param
                stats["failed_reorderings"] +=1
                logger.warning(f"Initial sample {i+1} reordering failed or produced original code, skipping.")
                continue

            reordered_code_hash = hashlib.md5(reordered_code.encode()).hexdigest()[:16]
            
            if reordered_code_hash not in unique_reorderings_generated_hashes:
                 unique_reorderings_generated_hashes.add(reordered_code_hash)
                 stats["unique_reorderings_generated"] +=1
            
            if reordered_code_hash in self.results_cache:
                logger.info(f"Cache hit for reordering {reordered_code_hash[:8]} in Phase 1 prep")
                cached_score_tuple = self.results_cache.get(reordered_code_hash)
                if cached_score_tuple and cached_score_tuple[0] is not None:
                    stats["cache_hits_reordering_score"] +=1
                    score_int = int(cached_score_tuple[0])
                    all_scores_valid.append(score_int)
                    if score_int < best_score:
                        best_score = score_int
                        best_code = cached_score_tuple[1]
                        best_topo_sort = topo_sort
                else:
                    stats["cache_hits_reordering_failed_eval"] +=1
                continue

            incomplete_reordered = incomplete_code_transform(reordered_code)
            if incomplete_reordered is None:
                logger.warning(f"Transformation to incomplete code failed for initial sample {i+1}.")
                stats["transform_to_incomplete_failed"] +=1
                self.results_cache[reordered_code_hash] = (None, reordered_code)
                continue
            
            incomplete_codes_batch.append(incomplete_reordered)
            reordered_codes_batch_for_llm.append(reordered_code)
            topo_sorts_for_llm_batch_phase1.append(topo_sort)

        if incomplete_codes_batch:
            stats["evaluation_attempts_for_new_reorderings"] += len(incomplete_codes_batch)
            logger.info(f"Evaluating {len(incomplete_codes_batch)} initial samples in a batch.")
            source_codes_eval_batch = [source_code_for_eval] * len(incomplete_codes_batch)
            summarizations_eval_batch = [summarization_for_eval] * len(incomplete_codes_batch)
            
            scores_batch = self.evaluation_function_batch(
                incomplete_codes_batch, source_codes_eval_batch, summarizations_eval_batch
            )

            for i, score in enumerate(scores_batch):
                current_reordered_code = reordered_codes_batch_for_llm[i]
                current_topo_sort = topo_sorts_for_llm_batch_phase1[i]
                reordered_code_hash = hashlib.md5(current_reordered_code.encode()).hexdigest()[:16]
                self.results_cache[reordered_code_hash] = (score, current_reordered_code)

                if score is None:
                    stats["failed_llm_evaluations"] += 1
                    logger.warning(f"Initial sample (from batch evaluation) failed.")
                    continue
                
                stats["successful_llm_evaluations"] +=1
                all_scores_valid.append(score)
                if score < best_score:
                    best_score = score
                    best_code = current_reordered_code
                    best_topo_sort = current_topo_sort
                    logger.info(f"New best adversarial score from Phase 1 batch: {best_score}")

        stats["unique_perturbations"] = len(unique_reorderings_generated_hashes)

        if best_topo_sort is None and not all_scores_valid: # Check if any valid score was found
            logger.warning("No valid reorderings found or evaluable in initial sampling")
            return original_code_param, best_score, all_scores_valid, stats # best_score remains inf
        
        # --- Phase 2: Neighborhood search iterations (Batched) ---
        logger.info("Phase 2: Neighborhood search (batched)...")
        change_prob = self.change_probability

        for iteration in range(self.iterations):
            if best_topo_sort is None: # Should have a valid topo sort if we got here from phase 1
                logger.warning(f"Iteration {iteration+1}: best_topo_sort is None, cannot generate neighbors. Breaking.")
                break
            improved_in_iteration = False # Renamed for clarity
            logger.info(f"Iteration {iteration+1}/{self.iterations}, change probability: {change_prob:.2f}")

            neighbor_sorts_phase2 = []
            for _ in range(self.neighborhood_samples):
                 stats["total_topo_sorts_generated"] += 1
                 neighbor_sorts_phase2.append(
                     neighborhood_topological_sort(base_nx_graph, best_topo_sort, change_prob)
                 )
            
            incomplete_codes_batch_p2: List[str] = []
            reordered_codes_batch_for_llm_p2: List[str] = []
            topo_sorts_for_llm_batch_phase2: List[TopoSort] = []

            for i, neighbor_sort in enumerate(neighbor_sorts_phase2):
                stats["total_reordering_attempts"] += 1
                reordered_code, _, _, _ = self.get_reordered_code(original_code_param, neighbor_sort, base_graph, base_ast_tree)

                if reordered_code is None or reordered_code == original_code_param:
                    stats["failed_reorderings"] +=1
                    logger.warning(f"Neighborhood sample {i+1} reordering failed or produced original code, skipping.")
                    continue
                
                reordered_code_hash = hashlib.md5(reordered_code.encode()).hexdigest()[:16]
                if reordered_code_hash not in unique_reorderings_generated_hashes:
                     unique_reorderings_generated_hashes.add(reordered_code_hash)
                     stats["unique_reorderings_generated"] +=1

                if reordered_code_hash in self.results_cache:
                    logger.info(f"Cache hit for reordering {reordered_code_hash[:8]} in Phase 2 prep")
                    cached_score_tuple = self.results_cache.get(reordered_code_hash)
                    if cached_score_tuple and cached_score_tuple[0] is not None:
                        stats["cache_hits_reordering_score"] +=1
                        score_int = int(cached_score_tuple[0])
                        all_scores_valid.append(score_int)
                        if score_int < best_score:
                            best_score = score_int
                            best_code = cached_score_tuple[1]
                            best_topo_sort = neighbor_sort
                            improved_in_iteration = True
                            logger.info(f"New best adversarial score from Phase 2 cache: {best_score}")
                    else:
                        stats["cache_hits_reordering_failed_eval"] +=1
                    continue

                incomplete_reordered = incomplete_code_transform(reordered_code)
                if incomplete_reordered is None:
                    logger.warning(f"Transformation to incomplete code failed for neighbor sample {i+1}.")
                    stats["transform_to_incomplete_failed"] +=1
                    self.results_cache[reordered_code_hash] = (None, reordered_code)
                    continue
                
                incomplete_codes_batch_p2.append(incomplete_reordered)
                reordered_codes_batch_for_llm_p2.append(reordered_code)
                topo_sorts_for_llm_batch_phase2.append(neighbor_sort)

            if incomplete_codes_batch_p2:
                stats["evaluation_attempts_for_new_reorderings"] += len(incomplete_codes_batch_p2)
                logger.info(f"Evaluating {len(incomplete_codes_batch_p2)} neighborhood samples in a batch.")
                source_codes_eval_batch_p2 = [source_code_for_eval] * len(incomplete_codes_batch_p2)
                summarizations_eval_batch_p2 = [summarization_for_eval] * len(incomplete_codes_batch_p2)

                scores_batch_p2 = self.evaluation_function_batch(
                    incomplete_codes_batch_p2, source_codes_eval_batch_p2, summarizations_eval_batch_p2
                )

                for i, score in enumerate(scores_batch_p2):
                    current_reordered_code = reordered_codes_batch_for_llm_p2[i]
                    current_topo_sort = topo_sorts_for_llm_batch_phase2[i]
                    reordered_code_hash = hashlib.md5(current_reordered_code.encode()).hexdigest()[:16]
                    self.results_cache[reordered_code_hash] = (score, current_reordered_code)

                    if score is None:
                        stats["failed_llm_evaluations"] += 1
                        logger.warning(f"Neighborhood sample (from batch evaluation) failed.")
                        continue
                    
                    stats["successful_llm_evaluations"] +=1
                    all_scores_valid.append(score)
                    if score < best_score:
                        best_score = score
                        best_code = current_reordered_code
                        best_topo_sort = current_topo_sort
                        improved_in_iteration = True
                        logger.info(f"New best adversarial score from Phase 2 batch: {best_score}")
            
            if not improved_in_iteration:
                change_prob = min(0.8, change_prob * 1.5) # Increase exploration
            else:
                change_prob = max(0.1, change_prob * 0.9) # Decrease exploration, stick to promising areas
        
        stats["unique_perturbations"] = len(unique_reorderings_generated_hashes)
        
        logger.info(f"Multi-step search completed. Best adversarial score: {best_score}")
        logger.info(f"Stats: {stats}")
        if stats["evaluation_attempts_for_new_reorderings"] > 0:
            success_rate = (stats["successful_llm_evaluations"] * 100) / (stats["evaluation_attempts_for_new_reorderings"] + 1e-6) # avoid div by zero
            logger.info(f"LLM evaluation success rate for new reorderings: {stats['successful_llm_evaluations']}/{stats['evaluation_attempts_for_new_reorderings']} ({success_rate:.1f}%)")
        else:
            logger.info("No new reorderings were sent for LLM evaluation.")
        
        if not all_scores_valid and best_score == float('inf'): # Check if any score was actually found
            logger.warning("No valid evaluations found throughout the search, returning original code")
            return original_code_param, float('inf'), [], stats
        
        return best_code, best_score, all_scores_valid, stats

def setup_llm(
    checkpoint: str = "google/gemma-3-1b-it",
    access_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    role: str = "judge"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Setting up {role} model: {checkpoint} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=access_token,
        cache_dir=cache_dir
    )

    if tokenizer.pad_token_id is None:
        logger.info(f"Tokenizer pad_token_id not set for {role} model. Setting to eos_token_id: {tokenizer.eos_token_id}")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if "gemma-3" in checkpoint.lower():
        logger.info(f"Detected Gemma 3 model, using Gemma3ForCausalLM")
        model_class = Gemma3ForCausalLM
    else:
        logger.info(f"Using AutoModelForCausalLM for model: {checkpoint}")
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=access_token,
        cache_dir=cache_dir
    )

    if model.generation_config.do_sample is False: 
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    logger.info(f"{role.capitalize()} model loaded. Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    return model, tokenizer, device

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 2048, # Default for completion
    temperature: float = 0.0
) -> List[str]:
    if isinstance(prompts, str): # Ensure it's a list for consistency
        prompts = [prompts]

    # Tokenizer should already be configured for left padding and pad_token_id in setup_llm
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,        # Pad to the longest sequence in the batch
        truncation=True,     # Truncate sequences longer than model's max length
        add_special_tokens=False # Assuming chat template already added them
    ).to(model.device)

    generate_kwargs = {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs.get('attention_mask'), # Use .get for safety
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["do_sample"] = True
    else: # Greedy decoding for temperature 0
        generate_kwargs["do_sample"] = False
    
    outputs = model.generate(**generate_kwargs)

    responses = [tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                 for output in outputs]
    return responses

def extract_code_from_response(response: str) -> str:
    if not response:
        return ""
    assistant_part = response
    if "assistant\n" in response: # Common prefix for chat models
        assistant_part = response.split("assistant\n", 1)[1].strip()
    code_blocks = re.findall(r"```python\n([\s\S]*?)```", assistant_part)
    if code_blocks:
        return code_blocks[-1].strip() # Return the last Python code block
    return assistant_part.strip() # Fallback if no ```python ``` found

def validate_completion(completed_code: str, incomplete_code: str) -> bool:
    logger.debug("Validating completion...") # Changed to debug
    if not completed_code:
        return False
    incomplete_pass_count = incomplete_code.count("pass")
    completed_pass_count = completed_code.count("pass")
    if completed_pass_count < incomplete_pass_count: # Fewer 'pass' statements implies implementation
        return True
    if len(completed_code) > len(incomplete_code) * 1.2: # Substantially longer
        return True
    return False

def extract_rating(response_text: str) -> Optional[str]:
    match = re.search(r'RATING:\s*([1-5])\b', response_text)
    if match:
        return match.group(1)
    match = re.search(r'\b([1-5])\b', response_text.strip()) # Handles just a number
    if match:
        return match.group(1)
    match = re.search(r'model\s*(\d+)\s*$', response_text.strip()) # Handles "model X" at the end
    if match and 1 <= int(match.group(1)) <= 5:
        return match.group(1)
    return None

class LLMEvaluator:
    def __init__(
        self,
        judge_model: PreTrainedModel,
        judge_tokenizer: PreTrainedTokenizer,
        completion_model: PreTrainedModel,
        completion_tokenizer: PreTrainedTokenizer,
        completion_temperature: float = 0,
        judgement_temperature: float = 0,
        completion_model_name: str = "",
        max_batch_size: int = 8
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
    
    def _prepare_completion_prompts(self, incomplete_codes: List[str], summarizations: List[str]) -> List[str]:
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
            # Use the completion_tokenizer here
            prompts.append(self.completion_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return prompts

    def _prepare_judgement_prompts(self, source_codes: List[str], summarizations: List[str], completed_codes: List[str]) -> List[str]:
        prompts = []
        for source_code, summarization, completed_code in zip(source_codes, summarizations, completed_codes):
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
            # Use the judge_tokenizer here
            prompts.append(self.judge_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return prompts

    def generate_completions_batch(self, incomplete_codes: List[str], summarizations: List[str]) -> List[str]:
        final_completions = [""] * len(incomplete_codes)
        prompts_to_generate = []
        indices_to_generate = []
        cache_keys_to_generate = []

        for i, (inc_code, summ) in enumerate(zip(incomplete_codes, summarizations)):
            cache_key = hashlib.md5((inc_code + summ + self.completion_model_name).encode()).hexdigest()
            if cache_key in self.completion_cache:
                logger.debug(f"Using cached completion for item {i + 1}") # Debug level
                final_completions[i] = self.completion_cache[cache_key]
            else:
                # Prepare prompt using the correct method for completion
                prompts_to_generate.append(self._prepare_completion_prompts([inc_code], [summ])[0])
                indices_to_generate.append(i)
                cache_keys_to_generate.append(cache_key)

        if prompts_to_generate:
            logger.info(f"Generating {len(prompts_to_generate)} completions in batches of up to {self.max_batch_size}")
            generated_code_snippets = []
            for j in range(0, len(prompts_to_generate), self.max_batch_size):
                batch_prompts = prompts_to_generate[j:j+self.max_batch_size]
                logger.info(f"Generating completion with {self.completion_model_name} for batch (size: {len(batch_prompts)})")
                try:
                    responses = generate_text(
                        self.completion_model, self.completion_tokenizer,
                        prompts=batch_prompts, temperature=self.completion_temperature, max_new_tokens=2048
                    )
                    generated_code_snippets.extend([extract_code_from_response(res) for res in responses])
                except Exception as e:
                    logger.error(f"Batch completion generation failed: {e}")
                    logger.error(traceback.format_exc())
                    generated_code_snippets.extend([""] * len(batch_prompts))

            for original_idx, cache_key, completed_code_str, inc_code_for_validation in zip(indices_to_generate, cache_keys_to_generate, generated_code_snippets, [incomplete_codes[k] for k in indices_to_generate]):
                is_valid = validate_completion(completed_code_str, inc_code_for_validation)
                logger.debug(f"Completion for item {original_idx + 1} valid: {is_valid}")
                if not is_valid:
                    logger.warning(f"Invalid completion detected for item {original_idx + 1}")
                    final_completions[original_idx] = ""
                    self.completion_cache[cache_key] = ""
                else:
                    final_completions[original_idx] = completed_code_str
                    self.completion_cache[cache_key] = completed_code_str
        return final_completions

    # Inside class LLMEvaluator

    def judge_completions_batch(self, source_codes: List[str], summarizations: List[str], completed_codes: List[str]) -> List[Optional[int]]:
        final_ratings: List[Optional[int]] = [None] * len(source_codes)
        items_to_process = [] # List of (index, src, summ, comp, cache_key)
        
        # 1. Initial Cache Check and Prepare Items
        for i, (src_code, summ, comp_code) in enumerate(zip(source_codes, summarizations, completed_codes)):
            if not comp_code:
                final_ratings[i] = None
                continue
            cache_key = hashlib.md5((src_code + summ + comp_code).encode()).hexdigest()
            if cache_key in self.judgement_cache and self.judgement_cache[cache_key] is not None:
                logger.debug(f"Using cached judgement for item {i + 1}")
                final_ratings[i] = self.judgement_cache[cache_key]
            else:
                items_to_process.append({"original_idx": i, "src": src_code, "summ": summ, "comp": comp_code, "key": cache_key, "status": "pending"})

        if not items_to_process:
            return final_ratings

        # 2. Process items in batches with retry
        current_batch_size = self.max_batch_size # Start with the optimal general batch size
        
        # Loop until all pending items are processed or deemed unprocessable
        processed_indices_this_pass = set() 
        
        # Keep track of items that failed even with BS=1 to avoid infinite loops on them
        permanently_failed_indices = set()

        while any(item['status'] == 'pending' for item in items_to_process):
            batch_items_to_attempt = [item for item in items_to_process if item['status'] == 'pending' and item['original_idx'] not in permanently_failed_indices][:current_batch_size]
            
            if not batch_items_to_attempt: # No more processable items in this loop (e.g. all remaining are permanently failed)
                if any(item['status'] == 'pending' for item in items_to_process): # Should only be permanently failed ones
                    logger.warning(f"Remaining pending items are all marked as permanently failed. Stopping processing loop.")
                break


            batch_prompts_to_judge = []
            batch_original_indices = []
            batch_cache_keys = []

            for item_data in batch_items_to_attempt:
                prompt = self._prepare_judgement_prompts([item_data["src"]],[item_data["summ"]],[item_data["comp"]])[0]
                batch_prompts_to_judge.append(prompt)
                batch_original_indices.append(item_data["original_idx"])
                batch_cache_keys.append(item_data["key"])
                
                # Log token count before attempting
                token_count = len(self.judge_tokenizer(prompt, add_special_tokens=False)['input_ids'])
                logger.info(f"JUDGEMENT ATTEMPT (Batch Size: {len(batch_items_to_attempt)}, Item original_idx: {item_data['original_idx']}): Prompt token count: {token_count}")


            if not batch_prompts_to_judge: # Should not happen if batch_items_to_attempt was populated
                break

            logger.info(f"Attempting to generate {len(batch_prompts_to_judge)} judgements with batch size {len(batch_prompts_to_judge)}")
            
            judgement_responses = []
            try:
                responses = generate_text(
                    self.judge_model, self.judge_tokenizer,
                    prompts=batch_prompts_to_judge, temperature=self.judgement_temperature, max_new_tokens=64
                )
                judgement_responses.extend(responses)
                
                # If successful, process responses and update status
                for i, response_text in enumerate(judgement_responses):
                    original_idx = batch_original_indices[i]
                    cache_key = batch_cache_keys[i]
                    rating = extract_rating(response_text)
                    item_to_update = next(item for item in items_to_process if item["original_idx"] == original_idx)
                    
                    if rating:
                        rating_int = int(rating)
                        logger.info(f"Extracted rating for item {original_idx + 1}: {rating_int}")
                        final_ratings[original_idx] = rating_int
                        self.judgement_cache[cache_key] = rating_int
                        item_to_update['status'] = 'processed'
                    else:
                        logger.warning(f"Rating extraction failed for item {original_idx + 1}. Response: '{response_text[:100]}...'")
                        final_ratings[original_idx] = None
                        self.judgement_cache[cache_key] = None
                        item_to_update['status'] = 'failed_extraction' # Or 'processed' if you consider this a form of processing

                # If this batch was successful with current_batch_size, try to reset to max_batch_size for next iteration
                # (unless current_batch_size is already max_batch_size)
                if len(batch_items_to_attempt) == current_batch_size and current_batch_size < self.max_batch_size :
                    # Optional: If you want to be more aggressive in trying to go back to larger batches
                    # current_batch_size = self.max_batch_size
                    # logger.info(f"Successful batch, resetting attempt batch size to {current_batch_size} for next pass.")
                    pass


            except RuntimeError as e: # Catch generic RuntimeErrors which include OOM
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM encountered with batch size {len(batch_items_to_attempt)}. Items: {[item['original_idx'] for item in batch_items_to_attempt]}. Error: {e}")
                    logger.error(traceback.format_exc()) # Log full traceback for OOM
                    
                    if len(batch_items_to_attempt) == 1:
                        # OOM with batch size 1, this item is problematic
                        failed_item_original_idx = batch_items_to_attempt[0]['original_idx']
                        logger.error(f"Item {failed_item_original_idx} caused OOM even with batch size 1. Marking as permanently failed.")
                        item_to_update = next(item for item in items_to_process if item["original_idx"] == failed_item_original_idx)
                        item_to_update['status'] = 'failed_oom_bs1'
                        self.judgement_cache[item_to_update['key']] = None # Cache as failed
                        final_ratings[failed_item_original_idx] = None
                        permanently_failed_indices.add(failed_item_original_idx)
                    else:
                        # Reduce batch size and retry this set of items (or just mark them as pending for next loop with smaller BS)
                        logger.info(f"Reducing trial batch size for next attempt due to OOM.")
                        # No items in this batch were processed, they remain 'pending'
                        # The outer while loop will pick them up again.
                        # The key is to ensure the next `current_batch_size` is smaller.
                    
                    # Reduce batch size for the next attempt in the *outer while loop*
                    current_batch_size = max(1, len(batch_items_to_attempt) // 2) # Or just set to 1
                    logger.info(f"Next attempt for pending items will use batch size at most {current_batch_size}")
                    # Continue to the next iteration of the while loop, which will form a new (potentially smaller) batch.

                else: # Other runtime error
                    logger.error(f"Runtime error during batch judgement: {e}")
                    logger.error(traceback.format_exc())
                    for item_data in batch_items_to_attempt: # Mark all items in this failed batch
                        item_data['status'] = 'failed_runtime'
                        self.judgement_cache[item_data['key']] = None
                        final_ratings[item_data['original_idx']] = None
                    current_batch_size = self.max_batch_size # Reset for next distinct set of items
            except Exception as e_gen: # Broader exception for other generate_text issues
                logger.error(f"General exception during batch judgement generation: {e_gen}")
                logger.error(traceback.format_exc())
                for item_data in batch_items_to_attempt: # Mark all items in this failed batch
                    item_data['status'] = 'failed_exception'
                    self.judgement_cache[item_data['key']] = None
                    final_ratings[item_data['original_idx']] = None
                current_batch_size = self.max_batch_size # Reset

            # After each attempt (success or handled OOM for a batch), reset current_batch_size for the next *new* set of pending items.
            # The reduction to handle OOM for a *specific* failing batch is temporary for that batch's items.
            if len(batch_items_to_attempt) < current_batch_size and current_batch_size > 1:
                pass # Keep reduced batch size if the last attempt was smaller than current_batch_size due to pending item count
            else:
                current_batch_size = self.max_batch_size


        # Final pass for any items that failed extraction but not OOM, mark them as None in final_ratings
        for item_data in items_to_process:
            if item_data['status'] not in ['processed', 'failed_oom_bs1']:
                logger.warning(f"Item {item_data['original_idx']+1} was not successfully processed. Final status: {item_data['status']}")
                if final_ratings[item_data['original_idx']] is None: # If not already set by OOM BS1 case
                    final_ratings[item_data['original_idx']] = None
                    self.judgement_cache[item_data['key']] = None


        return final_ratings

    def evaluate_batch(self, incomplete_codes: List[str], source_codes: List[str], summarizations: List[str]) -> List[Optional[int]]:
        if not incomplete_codes:
            return []

        logger.info(f"Starting batch completion for {len(incomplete_codes)} items.")
        completed_codes = self.generate_completions_batch(incomplete_codes, summarizations)
        
        valid_indices = [i for i, code in enumerate(completed_codes) if code] # Get indices of successful completions
        if not valid_indices:
            logger.warning("No valid completions generated in the batch.")
            return [None] * len(incomplete_codes) # Return all Nones if no valid completions

        source_codes_for_judgement = [source_codes[i] for i in valid_indices]
        summarizations_for_judgement = [summarizations[i] for i in valid_indices]
        completed_codes_for_judgement = [completed_codes[i] for i in valid_indices]
        
        logger.info(f"Starting batch judgement for {len(completed_codes_for_judgement)} validly completed items.")
        ratings_for_valid = self.judge_completions_batch(source_codes_for_judgement, summarizations_for_judgement, completed_codes_for_judgement)

        final_ratings: List[Optional[int]] = [None] * len(incomplete_codes)
        for i, rating in enumerate(ratings_for_valid):
            original_idx = valid_indices[i] # Map back to original index in the batch
            final_ratings[original_idx] = rating
            if rating is not None:
                 logger.info(f"Final evaluation rating for item {original_idx + 1}: {rating}")
            else:
                logger.warning(f"Judgement failed for item {original_idx + 1}")
        
        return final_ratings

    def evaluate(self, incomplete_code: str, source_code: str, summarization: str) -> Optional[int]:
        results = self.evaluate_batch([incomplete_code], [source_code], [summarization])
        return results[0] if results else None
    
def has_perturbation_potential(code_str: str) -> bool:
    try:
        tree = ast.parse(code_str)
        top_level_stmts = [node for node in tree.body 
                          if not (isinstance(node, ast.Expr) and 
                                 isinstance(node.value, (ast.Constant, ast.Str)))] # ast.Str for older pythons
        if len(top_level_stmts) < 3: # Heuristic: need at least 3 statements for meaningful reordering
            return False
        return True
    except SyntaxError: # If code can't be parsed, it can't be perturbed by this method
        return False
    except Exception as e: # Catch other parsing related errors
        logger.warning(f"Could not assess perturbation potential due to parsing error: {e}")
        return False

def analyze_score_distributions(score_distributions: Dict[str, List[int]], output_dir: str = "") -> None:
    filtered_distributions: Dict[str, List[int]] = {}
    for sample_id, scores in score_distributions.items():
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            filtered_distributions[sample_id] = valid_scores
    
    all_scores = [score for scores in filtered_distributions.values() for score in scores]
    
    if not all_scores:
        logger.warning("No valid scores to analyze for overall distribution.")
        return
    
    score_counts = Counter(all_scores)
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    variance = np.var(all_scores)
    logger.info(f"Overall statistics - Mean: {mean_score:.2f}, Median: {median_score:.2f}, Variance: {variance:.2f}")
    
    plt.figure(figsize=(12, 10)) # Adjusted figure size
    
    plt.subplot(2, 1, 1)
    plt.hist(all_scores, bins=np.arange(0.5, 6.5, 1), alpha=0.7, edgecolor='black')
    plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='g', linestyle='--', label=f'Median: {median_score:.2f}')
    plt.title(f'Distribution of Adversarial Scores (Valid Evaluations Only)\nVariance: {variance:.2f}')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    x_labels = sorted(score_counts.keys())
    y_values = [score_counts[score] for score in x_labels]
    plt.bar(x_labels, y_values, color='skyblue', edgecolor='black')
    plt.title('Count of Each Score Value (Valid Evaluations Only)')
    plt.xlabel('Score (1-5)')
    plt.ylabel('Count')
    plt.xticks(range(1, 6))
    for i, count in enumerate(y_values):
        plt.text(x_labels[i], count + 0.05 * max(y_values), str(count), ha='center') # Relative offset for text
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'score_distribution_summary.png')
    plt.savefig(fig_path)
    logger.info(f"Score distribution summary visualization saved to {fig_path}")
    plt.close() # Close the figure
    
    sample_stats: Dict[str, Dict[str, Union[int, float, str]]] = {} # Adjusted type for 'id'
    for sample_id_key, scores in filtered_distributions.items():
        sample_stats[str(sample_id_key)] = { # Ensure sample_id is string for JSON
            'min': min(scores), 'max': max(scores),
            'mean': np.mean(scores), 'median': np.median(scores),
            'variance': np.var(scores), 'count': len(scores),
            'valid_percentage': (len(scores) / len(score_distributions.get(sample_id_key, []))) * 100
                               if score_distributions.get(sample_id_key) else 0
        }
    
    stats_path = os.path.join(output_dir, 'per_sample_score_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(sample_stats, f, indent=2)
    logger.info(f"Per-sample score statistics saved to {stats_path}")

def generate_per_sample_histograms(
    score_distributions: Dict[str, List[int]], 
    base_output_dir: str = "",
    histogram_dir: str = "sample_histograms",
    original_scores: Optional[Dict[str, int]] = None # Make original_scores optional
) -> None:
    output_dir_path = os.path.join(base_output_dir, histogram_dir) # Renamed to avoid conflict
    os.makedirs(output_dir_path, exist_ok=True)
    
    total_samples = len(score_distributions)
    logger.info(f"Generating histograms for {total_samples} samples into {output_dir_path}")
    
    for i, (sample_id, scores) in enumerate(score_distributions.items()):
        valid_scores = [score for score in scores if score is not None]
        if not valid_scores:
            logger.warning(f"Sample {sample_id} has no valid scores, skipping histogram.")
            continue
        
        mean_score = np.mean(valid_scores)
        median_score = np.median(valid_scores)
        min_score = min(valid_scores)
        max_score = max(valid_scores)
        score_variance = np.var(valid_scores)
        
        original_score_val = None # Renamed
        if original_scores and sample_id in original_scores:
            original_score_val = original_scores[sample_id]
        
        str_id = str(sample_id)
        clean_id = ''.join(c for c in str_id if c.isalnum() or c in '._-')
        
        plt.figure(figsize=(10, 6))
        score_counts = Counter(valid_scores)
        bins = np.arange(0.5, 6.5, 1)
        plt.hist(valid_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2)
        
        plt.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.2f}')
        
        if original_score_val is not None:
            plt.axvline(original_score_val, color='purple', linestyle='-', linewidth=3, label=f'Original Score: {original_score_val}')
            max_height = plt.gca().get_ylim()[1]
            marker_height = max_height * 0.75
            plt.plot([original_score_val], [marker_height], 'v', color='purple', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
            plt.text(original_score_val, marker_height * 1.05, f"Original: {original_score_val}", color='purple', fontweight='bold', ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='purple', boxstyle='round,pad=0.2'))
        
        for score_val_hist in range(1, 6): # Renamed
            count = score_counts.get(score_val_hist, 0)
            if count > 0:
                plt.text(score_val_hist, count + 0.02 * max(score_counts.values(), default=1), str(count), ha='center', fontweight='bold') # Adjusted offset

        title = f'Rating Distribution for Sample {str_id}\n'
        if original_score_val is not None: title += f'Original Score: {original_score_val}, '
        title += f'Min: {min_score}, Max: {max_score}, Var: {score_variance:.2f}, n={len(valid_scores)}'
        
        plt.title(title, fontsize=12)
        plt.xlabel('Rating (1-5)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.xticks(range(1, 6))
        plt.grid(True, alpha=0.4, linestyle=':')
        plt.legend()
        
        output_path_fig = os.path.join(output_dir_path, f'histogram_sample_{clean_id}.png') # Renamed
        plt.tight_layout()
        plt.savefig(output_path_fig, dpi=100)
        plt.close()
        
        if (i + 1) % 10 == 0 or i + 1 == total_samples:
            logger.info(f"Generated {i + 1}/{total_samples} per-sample histograms.")
    
    logger.info(f"All per-sample histograms saved to {output_dir_path}/")

# ... (keep generate_histogram_grid and analyze_adversarial_vs_normal_scores as they are, they seem okay)
def generate_histogram_grid(
    score_distributions: Dict[str, List[int]],
    max_samples: int = 16,
    output_dir: str = "",
    output_filename: str = "histogram_grid.png",
    original_scores: Optional[Dict[str, int]] = None
) -> None:
    output_path = os.path.join(output_dir, output_filename)
    valid_samples = {}
    for sample_id, scores in score_distributions.items():
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            valid_samples[sample_id] = valid_scores
            if len(valid_samples) >= max_samples:
                break
    if not valid_samples:
        logger.warning("No valid samples with scores, cannot generate histogram grid.")
        return

    n_samples = len(valid_samples)
    grid_size_cols = int(np.ceil(np.sqrt(n_samples)))
    grid_size_rows = int(np.ceil(n_samples / grid_size_cols))

    fig, axes = plt.subplots(grid_size_rows, grid_size_cols, figsize=(grid_size_cols * 4, grid_size_rows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    for i, (sample_id, scores) in enumerate(valid_samples.items()):
        ax = axes_flat[i]
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        original_score_val = original_scores.get(sample_id) if original_scores else None

        ax.hist(scores, bins=np.arange(0.5, 6.5, 1), alpha=0.75, color='steelblue', edgecolor='black', linewidth=1)
        ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color='green', linestyle='dotted', linewidth=1.5, label=f'Median: {median_score:.2f}')

        if original_score_val is not None:
            ax.axvline(original_score_val, color='purple', linestyle='solid', linewidth=2.5, label=f'Original: {original_score_val}')
            max_hist_height = ax.get_ylim()[1]
            # ax.text(original_score_val, max_hist_height * 0.9, "Orig", color='purple', fontsize=8, fontweight='bold', ha='center', va='top',
            #        bbox=dict(facecolor='white', alpha=0.5, pad=0.2))

        title_str = f'Sample {str(sample_id)}'
        if original_score_val is not None: title_str += f' (Orig: {original_score_val})'
        ax.set_title(title_str, fontsize=9)
        ax.set_xlabel('Rating', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3, linestyle=':')
        if i == 0: ax.legend(fontsize=7, loc='upper right')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f'Rating Distributions for {n_samples} Samples ( Adversarial Runs )', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust for suptitle and bottom
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Histogram grid saved to {output_path}")


def analyze_adversarial_vs_normal_scores(results: List[Dict[str, Any]], original_scores: Dict[str, int], output_dir:str = "", output_basename:str="adversarial_vs_normal_analysis"):
    output_path = os.path.join(output_dir, f"{output_basename}.json")
    viz_path = os.path.join(output_dir, f"{output_basename}.png")

    if not results or not original_scores:
        logger.warning("No results or original scores available for comparison.")
        return

    score_pairs = []
    for result in results:
        sample_id = result.get('id')
        if not sample_id or sample_id not in original_scores:
            continue
        original_score = original_scores[sample_id]
        adversarial_score = result.get('adversarial_score')
        if original_score is not None and adversarial_score is not None and adversarial_score != float('inf'):
            score_pairs.append((original_score, adversarial_score))

    if not score_pairs:
        logger.warning("No valid score pairs found for comparison.")
        return

    original_scores_list, adversarial_scores_list = zip(*score_pairs)
    avg_original = np.mean(original_scores_list)
    avg_adversarial = np.mean(adversarial_scores_list)
    median_original = np.median(original_scores_list)
    median_adversarial = np.median(adversarial_scores_list)
    differences = [orig - adv for orig, adv in score_pairs]
    avg_difference = np.mean(differences)
    worse_count = sum(1 for orig, adv in score_pairs if adv < orig)
    same_count = sum(1 for orig, adv in score_pairs if adv == orig)
    better_count = sum(1 for orig, adv in score_pairs if adv > orig)

    analysis_data = { # Renamed
        "sample_count": len(score_pairs),
        "original_scores_stats": { # Renamed for clarity
            "average": float(avg_original), "median": float(median_original),
            "min": float(min(original_scores_list)), "max": float(max(original_scores_list))
        },
        "adversarial_scores_stats": { # Renamed
            "average": float(avg_adversarial), "median": float(median_adversarial),
            "min": float(min(adversarial_scores_list)), "max": float(max(adversarial_scores_list))
        },
        "comparison_stats": { # Renamed
            "average_difference_orig_minus_adv": float(avg_difference), # Clarified
            "count_adv_worse_than_orig": worse_count, "percent_adv_worse": float(worse_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
            "count_adv_same_as_orig": same_count, "percent_adv_same": float(same_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
            "count_adv_better_than_orig": better_count, "percent_adv_better": float(better_count * 100 / len(score_pairs)) if len(score_pairs) > 0 else 0,
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)

    plt.figure(figsize=(14, 12)) # Adjusted figure size

    plt.subplot(2, 2, 1)
    bp_data = [original_scores_list, adversarial_scores_list] # Renamed for clarity
    plt.boxplot(bp_data, labels=['Original Scores', 'Adversarial Scores'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red', linewidth=2))
    plt.title('Score Distribution: Original vs. Adversarial', fontsize=12)
    plt.ylabel('Score (1-5)', fontsize=10)
    plt.yticks(np.arange(min(min(original_scores_list, default=1), min(adversarial_scores_list, default=1)), max(max(original_scores_list, default=5), max(adversarial_scores_list, default=5)) + 1))
    plt.grid(True, linestyle=':', alpha=0.7)
    # Add means
    plt.scatter([1, 2], [avg_original, avg_adversarial], color='darkred', marker='o', s=50, zorder=3, label=f'Means:\nOrig: {avg_original:.2f}\nAdv: {avg_adversarial:.2f}')
    plt.legend(fontsize=8)

    plt.subplot(2, 2, 2)
    plt.hist(differences, bins=np.arange(min(differences)-0.5, max(differences)+1.5, 1) if differences else 10 , alpha=0.75, color='salmon', edgecolor='black')
    plt.axvline(avg_difference, color='darkred', linestyle='--', linewidth=1.5, label=f'Avg Diff: {avg_difference:.2f}')
    plt.title('Histogram of Score Differences (Original - Adversarial)', fontsize=12)
    plt.xlabel('Score Difference (Original - Adversarial)', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(2, 2, 3)
    plt.scatter(original_scores_list, adversarial_scores_list, alpha=0.6, color='green', edgecolors='darkgreen')
    min_val_scatter = min(min(original_scores_list, default=1), min(adversarial_scores_list, default=1)) # default values
    max_val_scatter = max(max(original_scores_list, default=5), max(adversarial_scores_list, default=5))
    plt.plot([min_val_scatter, max_val_scatter], [min_val_scatter, max_val_scatter], 'k--', alpha=0.5, label='y=x (No Change)')
    plt.title('Original Score vs. Adversarial Score', fontsize=12)
    plt.xlabel('Original Score', fontsize=10)
    plt.ylabel('Adversarial Score', fontsize=10)
    plt.xticks(np.arange(min_val_scatter, max_val_scatter + 1))
    plt.yticks(np.arange(min_val_scatter, max_val_scatter + 1))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=8)
    plt.axis('equal') # Ensure a square plot for y=x line

    plt.subplot(2, 2, 4)
    bar_labels = ['Orig Avg', 'Adv Avg', 'Orig Med', 'Adv Med']
    bar_values = [avg_original, avg_adversarial, median_original, median_adversarial]
    colors = ['cornflowerblue', 'lightcoral', 'mediumseagreen', 'plum']
    bars = plt.bar(bar_labels, bar_values, color=colors, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    plt.title('Summary Score Statistics', fontsize=12)
    plt.ylabel('Score (1-5)', fontsize=10)
    plt.ylim(0, max(bar_values, default=5) * 1.15) # Dynamic Y limit
    plt.grid(True, linestyle=':', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)
    plt.close()

    logger.info(f"\nAdversarial vs Normal Analysis Results:")
    logger.info(f"  Original average score: {avg_original:.2f}, median: {median_original:.2f}")
    logger.info(f"  Adversarial average score: {avg_adversarial:.2f}, median: {median_adversarial:.2f}")
    logger.info(f"  Average difference (Original - Adversarial): {avg_difference:.2f}")
    logger.info(f"  Samples made worse by adversary: {worse_count}/{len(score_pairs)} ({analysis_data['comparison_stats']['percent_adv_worse']:.1f}%)")
    logger.info(f"  Analysis JSON saved to {output_path}")
    logger.info(f"  Analysis visualization saved to {viz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find adversarial code reorderings using batched search.")
    parser.add_argument("--input", type=str, default="/nfs/homedirs/hifl/Masterarbeit/pythonProject7/third_party/CodeScope/data/incomplete_code_summarization_data_python.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, default="adversarial_reorderings.jsonl", help="Base name for the output JSONL file (will be placed in model-specific directory).")
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-12b-it", help="Judge LLM model.")
    parser.add_argument("--completion_model", type=str, default="google/gemma-3-1b-it", help="Completion LLM model.")
    parser.add_argument("--access_token", type=str, default=None, help="HuggingFace access token.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models.")
    parser.add_argument("--sample_limit", type=int, default=50, help="Max samples to process.")
    parser.add_argument("--initial_samples", type=int, default=1, help="Initial random samples for search.")
    parser.add_argument("--neighborhood_samples", type=int, default=1, help="Neighborhood samples per iteration.")
    parser.add_argument("--iterations", type=int, default=1, help="Neighborhood search iterations.")
    parser.add_argument("--change_probability", type=float, default=0.3, help="Initial change probability in neighborhood search.")
    # Add a new argument for starting batch size for the finder
    parser.add_argument("--finder_start_batch_size", type=int, default=4, help="Starting batch size for find_executable_batch_size (try 1 or 2 for very large models/prompts).")


    args = parser.parse_args()
    
    completion_model_name_for_dir = args.completion_model.split('/')[-1] # Renamed for clarity
    output_dir_main = f"results_{completion_model_name_for_dir}_abcdefg" # Renamed
    os.makedirs(output_dir_main, exist_ok=True)
    
    args.output = os.path.join(output_dir_main, os.path.basename(args.output)) # Ensure output is in the model-specific dir
    logger.info(f"Results will be saved to directory: {output_dir_main}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    judge_model_instance: Optional[PreTrainedModel] = None
    judge_tokenizer_instance: Optional[PreTrainedTokenizer] = None
    completion_model_instance: Optional[PreTrainedModel] = None
    completion_tokenizer_instance: Optional[PreTrainedTokenizer] = None

    OPTIMAL_COMPLETION_BATCH_SIZE = 1
    OPTIMAL_JUDGEMENT_BATCH_SIZE = 1

    # --- Dummy prompts for find_executable_batch_size ---
    # These should represent WORST-CASE (longest, most complex) inputs your models will see.
    # Using actual long examples from your dataset would be even better if feasible.
    long_code_snippet = "class LongAndComplex:\n" + "    def method_with_many_lines(self, p1, p2, p3):\n" + \
                        "        # This is a very long comment to increase token count\n" * 10 + \
                        "        if p1 > p2:\n" + \
                        "            result = 0\n" + \
                        "            for i in range(p1 * 100):\n" + \
                        "                result += (i % (p2 + 1)) * p3 - (i // (p3 + 1))\n" * 20 + \
                        "        else:\n" + \
                        "            result = 1\n" + \
                        "            for j in range(p2 * 100):\n" + \
                        "                result *= (j % (p1 + 1) + 0.5) / (p3 + 0.1)\n" * 20 + \
                        "        return result\n" * 5 + "pass\n" # Ensure 'pass' for incomplete_code_transform
    
    long_summary = "This is an extremely detailed summarization designed to test the maximum token length for prompts. " + \
                   "It discusses various complex software engineering concepts, design patterns, and architectural considerations. " * 10 + \
                   "The goal is to ensure that the batch size determination can handle prompts that are very verbose and push the limits of the context window."

    # Placeholder for evaluator instance needed for prompt preparation in test ops
    # This temp evaluator is only for its _prepare_*_prompts methods.
    # We'll create the real evaluator later with the determined batch sizes.
    # For this temp one, models and tokenizers will be filled in shortly.
    temp_evaluator_for_bs_finder = LLMEvaluator(
        judge_model=None, judge_tokenizer=None, # Will be set
        completion_model=None, completion_tokenizer=None, # Will be set
        max_batch_size=1 # Placeholder, not used by _prepare methods
    )

    if args.judge_model == args.completion_model:
        logger.info(f"Judge and Completion models are the same: {args.judge_model}. Loading once.")
        single_model, single_tokenizer, _ = setup_llm(
            checkpoint=args.judge_model, access_token=args.access_token, cache_dir=args.cache_dir, role="shared"
        )
        judge_model_instance, judge_tokenizer_instance = single_model, single_tokenizer
        completion_model_instance, completion_tokenizer_instance = single_model, single_tokenizer
        
        temp_evaluator_for_bs_finder.judge_model = single_model
        temp_evaluator_for_bs_finder.judge_tokenizer = single_tokenizer
        temp_evaluator_for_bs_finder.completion_model = single_model
        temp_evaluator_for_bs_finder.completion_tokenizer = single_tokenizer


        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Attempting to find executable batch size for the shared model (testing with completion-like load)...")

            @find_executable_batch_size(starting_batch_size=args.finder_start_batch_size)
            def _bs_test_op_shared(bs: int, model_obj: PreTrainedModel, tokenizer_obj: PreTrainedTokenizer, evaluator_obj: LLMEvaluator):
                logger.debug(f"BS FINDER (Shared): Testing with batch_size = {bs}")
                # Use realistic prompts via the evaluator's preparation methods
                dummy_summaries = [long_summary] * bs
                dummy_inc_codes = [long_code_snippet] * bs
                
                prompts = evaluator_obj._prepare_completion_prompts(dummy_inc_codes, dummy_summaries)
                
                inputs = tokenizer_obj(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False).to(model_obj.device)
                with torch.no_grad():
                    # Test with max_new_tokens matching actual completion generation
                    _ = model_obj.generate(inputs['input_ids'], attention_mask=inputs.get('attention_mask'), max_new_tokens=2048)
                return bs # Not used by decorator, but good practice

            try:
                optimal_shared_bs = _bs_test_op_shared(single_model, single_tokenizer, temp_evaluator_for_bs_finder)
                OPTIMAL_COMPLETION_BATCH_SIZE = optimal_shared_bs
                OPTIMAL_JUDGEMENT_BATCH_SIZE = optimal_shared_bs
                logger.info(f"Found optimal shared batch size: {optimal_shared_bs}")
            except RuntimeError as e:
                logger.warning(f"Could not automatically find optimal shared batch size (RuntimeError: {e}). Using default: 1")
            except Exception as e:
                logger.warning(f"An unexpected error occurred while finding shared batch size: {e}. Using default: 1")

    else: # Different models for judge and completion
        logger.info(f"Setting up judge model: {args.judge_model}")
        judge_model_instance, judge_tokenizer_instance, _ = setup_llm(
            checkpoint=args.judge_model, access_token=args.access_token, cache_dir=args.cache_dir, role="judge"
        )
        logger.info(f"Setting up completion model: {args.completion_model}")
        completion_model_instance, completion_tokenizer_instance, _ = setup_llm(
            checkpoint=args.completion_model, access_token=args.access_token, cache_dir=args.cache_dir, role="completion"
        )

        temp_evaluator_for_bs_finder.judge_model = judge_model_instance
        temp_evaluator_for_bs_finder.judge_tokenizer = judge_tokenizer_instance
        temp_evaluator_for_bs_finder.completion_model = completion_model_instance
        temp_evaluator_for_bs_finder.completion_tokenizer = completion_tokenizer_instance

        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("Attempting to find executable batch size for completion model...")
            @find_executable_batch_size(starting_batch_size=args.finder_start_batch_size)
            def _bs_test_op_completion(bs: int, model_obj: PreTrainedModel, tokenizer_obj: PreTrainedTokenizer, evaluator_obj: LLMEvaluator):
                logger.debug(f"BS FINDER (Completion): Testing with batch_size = {bs}")
                dummy_summaries = [long_summary] * bs
                dummy_inc_codes = [long_code_snippet] * bs
                prompts = evaluator_obj._prepare_completion_prompts(dummy_inc_codes, dummy_summaries)
                inputs = tokenizer_obj(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False).to(model_obj.device)
                with torch.no_grad():
                    _ = model_obj.generate(inputs['input_ids'], attention_mask=inputs.get('attention_mask'), max_new_tokens=2048)
                return bs
            try:
                OPTIMAL_COMPLETION_BATCH_SIZE = _bs_test_op_completion(completion_model_instance, completion_tokenizer_instance, temp_evaluator_for_bs_finder)
                logger.info(f"Found optimal completion batch size: {OPTIMAL_COMPLETION_BATCH_SIZE}")
            except RuntimeError as e: logger.warning(f"Could not find optimal completion batch size (RuntimeError: {e}). Default: 1")
            except Exception as e: logger.warning(f"Error finding completion batch size: {e}. Default: 1")

            torch.cuda.empty_cache()
            logger.info("Attempting to find executable batch size for judgement model...")
            @find_executable_batch_size(starting_batch_size=args.finder_start_batch_size)
            def _bs_test_op_judgement(bs: int, model_obj: PreTrainedModel, tokenizer_obj: PreTrainedTokenizer, evaluator_obj: LLMEvaluator):
                logger.debug(f"BS FINDER (Judgement): Testing with batch_size = {bs}")
                dummy_src = [long_code_snippet] * bs # Source can also be long
                dummy_summ = [long_summary] * bs
                dummy_comp = [long_code_snippet + "\n    # Some completion code\n" + "    final_result = p1 + p2 + p3\n" *5] * bs # Completed code
                prompts = evaluator_obj._prepare_judgement_prompts(dummy_src, dummy_summ, dummy_comp)
                inputs = tokenizer_obj(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False).to(model_obj.device)
                with torch.no_grad():
                    _ = model_obj.generate(inputs['input_ids'], attention_mask=inputs.get('attention_mask'), max_new_tokens=64)
                return bs
            try:
                OPTIMAL_JUDGEMENT_BATCH_SIZE = _bs_test_op_judgement(judge_model_instance, judge_tokenizer_instance, temp_evaluator_for_bs_finder)
                logger.info(f"Found optimal judgement batch size: {OPTIMAL_JUDGEMENT_BATCH_SIZE}")
            except RuntimeError as e: logger.warning(f"Could not find optimal judgement batch size (RuntimeError: {e}). Default: 1")
            except Exception as e: logger.warning(f"Error finding judgement batch size: {e}. Default: 1")
    
    del temp_evaluator_for_bs_finder # Clean up the temporary evaluator

    # Instantiate the final evaluator with the determined (or single) models and batch sizes
    final_max_batch_size = min(OPTIMAL_COMPLETION_BATCH_SIZE, OPTIMAL_JUDGEMENT_BATCH_SIZE)
    if final_max_batch_size < 1: final_max_batch_size = 1 # Ensure at least 1
    logger.info(f"Setting LLMEvaluator max_batch_size to: {final_max_batch_size}")

    evaluator = LLMEvaluator(
        judge_model=judge_model_instance,
        judge_tokenizer=judge_tokenizer_instance,
        completion_model=completion_model_instance,
        completion_tokenizer=completion_tokenizer_instance,
        completion_model_name=args.completion_model.split('/')[-1],
        max_batch_size=final_max_batch_size
    )
        
    searcher = AdversarialReorderingSearch(
        evaluation_function_batch=evaluator.evaluate_batch,
        initial_samples=args.initial_samples,
        neighborhood_samples=args.neighborhood_samples,
        iterations=args.iterations,
        change_probability=args.change_probability,
    )
    
    dataset = load_dataset("json", split="train", data_files=str(args.input))
    if args.sample_limit > 0 and args.sample_limit < len(dataset): # Check if limit is smaller
        dataset = dataset.select(range(args.sample_limit))
    
    results: List[Dict[str, Any]] = []
    filtered_count: int = 0
    total_count: int = len(dataset)
    evaluation_failures: int = 0
    score_distributions: Dict[str, List[int]] = {} # sample_id -> list of scores
    original_scores: Dict[str, int] = {} # sample_id -> original score

    for i, sample in enumerate(dataset):
        sample_id_str = str(sample.get('id', f'unknown_id_{i}')) # Ensure ID is a string
        logger.info(f"\nProcessing sample {i+1}/{len(dataset)}: {sample_id_str}")
        
        if "source_code" not in sample or not sample["source_code"]:
            logger.warning(f"Sample {sample_id_str} lacks source_code, skipping.")
            continue
        source_code = sample["source_code"]

        if not has_perturbation_potential(source_code):
            logger.info(f"Sample {sample_id_str} has low perturbation potential, skipping.")
            filtered_count += 1
            continue

        summarization = sample.get("human_summarization", "")
        if not summarization:
            logger.warning(f"Sample {sample_id_str} lacks human_summarization, skipping LLM evaluation steps for this sample.")
            # Decide if you want to proceed with perturbation but no LLM eval, or just skip entirely
            # For now, we'll skip LLM-dependent parts
            results.append({**sample, "original_code": source_code, "perturbed_code": source_code, "adversarial_score": float('inf'), "notes": "Skipped LLM eval due to missing summarization"})
            continue


        logger.info(f"Evaluating original code for sample {sample_id_str}")
        incomplete_original = incomplete_code_transform(source_code)
        original_score_val = None # Initialize
        if incomplete_original:
            original_score_val = evaluator.evaluate(incomplete_original, source_code, summarization)
            if original_score_val is not None:
                original_scores[sample_id_str] = original_score_val
                logger.info(f"Original code score for {sample_id_str}: {original_score_val}")
            else:
                logger.warning(f"Failed to get original score for {sample_id_str}.")
        else:
            logger.warning(f"Failed to transform original code to incomplete for {sample_id_str}.")


        try:
            start_time = time.time()
            worst_reordering, score, all_scores_for_sample, search_stats = searcher.search(
                source_code, source_code, summarization
            )
            search_time = time.time() - start_time
            
            if score == float('inf') and not all_scores_for_sample:
                logger.warning(f"Sample {sample_id_str} had no valid adversarial evaluations, skipping from detailed results.")
                evaluation_failures += 1
                results.append({**sample, "original_code": source_code, "perturbed_code": source_code, "adversarial_score": float('inf'), "original_score": original_score_val, "notes": "No valid adversarial evaluations"})
                continue
            
            score_distributions[sample_id_str] = all_scores_for_sample
            if all_scores_for_sample: # Check if list is not empty
                logger.info(f"Score distribution for {sample_id_str} - Min: {min(all_scores_for_sample)}, Max: {max(all_scores_for_sample)}, "
                            f"Mean: {np.mean(all_scores_for_sample):.2f}, Count: {len(all_scores_for_sample)}")
            
            logger.info(f"Search for {sample_id_str} completed in {search_time:.2f} seconds.")
            
            incomplete_adversarial = incomplete_code_transform(worst_reordering)
            adversarial_completion = ""
            if incomplete_adversarial:
                completions_list = evaluator.generate_completions_batch([incomplete_adversarial], [summarization])
                adversarial_completion = completions_list[0] if completions_list else ""
            
            if worst_reordering == source_code: logger.warning(f"No successful perturbation found for {sample_id_str} - using original code.")
            
            result_entry = sample.copy()
            result_entry["id"] = sample_id_str # Ensure 'id' field is the one we are using
            result_entry["original_code"] = source_code
            result_entry["perturbed_code"] = worst_reordering
            result_entry["incomplete_perturbed_code"] = incomplete_adversarial
            result_entry["adversarial_score"] = score if score != float('inf') else None # Store None if no score found
            result_entry["original_score"] = original_score_val
            result_entry["search_time_seconds"] = search_time # Clarified unit
            result_entry["search_stats"] = search_stats # Renamed for clarity
            result_entry["adversarial_completion"] = adversarial_completion
            result_entry["score_distribution_all_attempts"] = all_scores_for_sample # Renamed
            results.append(result_entry)
            
        except Exception as e_sample:
            logger.error(f"Error processing sample {sample_id_str}: {e_sample}")
            logger.error(traceback.format_exc())
            results.append({**sample, "id": sample_id_str, "original_code": source_code, "perturbed_code": source_code, "adversarial_score": float('inf'), "original_score": original_score_val, "error": str(e_sample)})
            continue

    logger.info(f"\nFiltered out {filtered_count}/{total_count} samples with low perturbation potential.")
    logger.info(f"Encountered {evaluation_failures}/{total_count-filtered_count} samples with no valid adversarial evaluations found.")
    logger.info(f"Successfully processed and stored results for {len([r for r in results if r.get('adversarial_score') is not None and r.get('adversarial_score') != float('inf')])} samples.")

    stats_output_path = args.output.replace('.jsonl', '_search_stats_per_sample.json') # Renamed
    all_sample_search_stats: Dict[str, Dict[str, Any]] = {} # Renamed
    for res in results:
        s_id = str(res.get('id', 'unknown')) # Ensure string ID
        if 'search_stats' in res:
            all_sample_search_stats[s_id] = res['search_stats']
    with open(stats_output_path, 'w', encoding='utf-8') as fout:
        json.dump(all_sample_search_stats, fout, indent=2)
    logger.info(f"Search statistics per sample saved to {stats_output_path}")

    with open(args.output, 'w', encoding='utf-8') as fout:
        for res in results:
            fout.write(json.dumps(res) + "\n")
    logger.info(f"\nDetailed results saved to {args.output}")

    distribution_output_path = args.output.replace('.jsonl', '_score_distributions.json') # Changed to .json for a single dict
    # Convert sample_id keys in score_distributions to string for JSON compatibility
    string_keyed_score_distributions = {str(k): v for k, v in score_distributions.items()}
    with open(distribution_output_path, 'w', encoding='utf-8') as fout:
        json.dump(string_keyed_score_distributions, fout, indent=2)
    logger.info(f"Score distributions saved to {distribution_output_path}")

    if string_keyed_score_distributions: # Check if there's anything to analyze
        analyze_score_distributions(string_keyed_score_distributions, output_dir=output_dir_main)
        logger.info("Generating per-sample histograms...")
        generate_per_sample_histograms(string_keyed_score_distributions, original_scores=original_scores, base_output_dir=output_dir_main)
        if len(string_keyed_score_distributions) > 1:
            generate_histogram_grid(string_keyed_score_distributions, original_scores=original_scores, output_dir=output_dir_main)
        logger.info("Histogram generation complete.")
        logger.info("Analyzing adversarial vs normal scores...")
        analyze_adversarial_vs_normal_scores(results, original_scores, output_dir=output_dir_main)
    else:
        logger.info("No score distributions available to analyze or visualize.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.checkpoint') # Example: ignore specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning) # General FutureWarning
    main()