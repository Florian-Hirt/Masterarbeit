#!/usr/bin/env python
from typing import Dict, List, Any
import json
import argparse
import time
import torch
import logging
import warnings
from datasets import load_dataset
import numpy as np
import os
import traceback


# Import existing modules
from code_completion_generation import incomplete_code_transform
from masterarbeit.analysis.visualization import analyze_score_distributions, generate_per_sample_histograms, generate_histogram_grid, analyze_adversarial_vs_normal_scores
from masterarbeit.attacks.reordering.search import AdversarialReorderingSearch
from masterarbeit.metrics.adversarial import LLMEvaluator
from masterarbeit.models.llm_adapters import setup_llm
from masterarbeit.utils.adversarial import has_perturbation_potential

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 256



def setup_models_with_batch_size(args):
    """Setup models and determine optimal batch sizes."""
    
    # Simplified batch size determination
    if args.judge_model == args.completion_model:
        logger.info(f"Judge and Completion models are the same: {args.judge_model}")
        model, tokenizer, _ = setup_llm(
            logger = logger,
            checkpoint=args.judge_model, 
            access_token=args.access_token, 
            cache_dir=args.cache_dir, 
            role="shared", 
            use_8bit=args.use_8bit
        )
        
        # For 8-bit models, use conservative batch size
        max_batch_size = 1 if args.use_8bit else args.finder_start_batch_size
        
        return {
            "judge_model": model,
            "judge_tokenizer": tokenizer,
            "completion_model": model,
            "completion_tokenizer": tokenizer,
            "max_batch_size": max_batch_size
        }
    else:
        # Different models
        judge_model, judge_tokenizer, _ = setup_llm(
            logger = logger,
            checkpoint=args.judge_model,
            access_token=args.access_token,
            cache_dir=args.cache_dir,
            role="judge",
            use_8bit=args.use_8bit
        )
        
        completion_model, completion_tokenizer, _ = setup_llm(
            logger = logger,
            checkpoint=args.completion_model,
            access_token=args.access_token,
            cache_dir=args.cache_dir,
            role="completion",
            use_8bit=args.use_8bit
        )
        
        # Conservative batch size for 8-bit models
        max_batch_size = 1 if args.use_8bit else args.finder_start_batch_size
        
        return {
            "judge_model": judge_model,
            "judge_tokenizer": judge_tokenizer,
            "completion_model": completion_model,
            "completion_tokenizer": completion_tokenizer,
            "max_batch_size": max_batch_size
        }

def process_dataset(args, evaluator, searcher, output_dir):
    """Process the dataset and generate results."""
    dataset = load_dataset("json", split="train", data_files=str(args.input))
    if args.sample_limit > 0 and args.sample_limit < len(dataset):
        dataset = dataset.select(range(args.sample_limit))
    
    results = []
    score_distributions = {}
    original_scores = {}
    
    for i, sample in enumerate(dataset):
        sample_id = str(sample.get('id', f'unknown_id_{i}'))
        logger.info(f"\nProcessing sample {i+1}/{len(dataset)}: {sample_id}")
        
        if "source_code" not in sample or not sample["source_code"]:
            logger.warning(f"Sample {sample_id} lacks source_code, skipping.")
            continue
            
        source_code = sample["source_code"]
        
        if not has_perturbation_potential(source_code):
            logger.info(f"Sample {sample_id} has low perturbation potential, skipping.")
            continue
        
        summarization = sample.get("human_summarization", "")
        if not summarization:
            logger.warning(f"Sample {sample_id} lacks human_summarization, skipping.")
            continue
        
        # Process sample
        result = process_single_sample(
            sample, sample_id, source_code, summarization, 
            evaluator, searcher, original_scores, score_distributions
        )
        results.append(result)
    
    # Save results
    save_results(results, args.output, score_distributions)
    
    # Generate visualizations
    if score_distributions:
        generate_visualizations(logger, score_distributions, original_scores, output_dir, results)
    
    return results

def process_single_sample(sample, sample_id, source_code, summarization, 
                         evaluator, searcher, original_scores, score_distributions):
    """Process a single code sample."""
    # Evaluate original code
    incomplete_original = incomplete_code_transform(source_code)
    original_score = None
    
    if incomplete_original:
        original_score = evaluator.evaluate(incomplete_original, source_code, summarization)
        if original_score is not None:
            original_scores[sample_id] = original_score
            logger.info(f"Original code score for {sample_id}: {original_score}")
    
    # Search for adversarial reordering
    try:
        start_time = time.time()
        worst_reordering, score, all_scores, search_stats = searcher.search(
            source_code, source_code, summarization
        )
        search_time = time.time() - start_time
        
        score_distributions[sample_id] = all_scores
        
        if all_scores:
            logger.info(f"Score distribution for {sample_id} - "
                       f"Min: {min(all_scores)}, Max: {max(all_scores)}, "
                       f"Mean: {np.mean(all_scores):.2f}, Count: {len(all_scores)}")
        
        # Create result entry
        result_entry = sample.copy()
        result_entry.update({
            "id": sample_id,
            "original_code": source_code,
            "perturbed_code": worst_reordering,
            "adversarial_score": score if score != float('inf') else None,
            "original_score": original_score,
            "search_time_seconds": search_time,
            "search_stats": search_stats,
            "score_distribution_all_attempts": all_scores
        })
        
        return result_entry
        
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        logger.error(traceback.format_exc())
        
        return {
            **sample,
            "id": sample_id,
            "original_code": source_code,
            "perturbed_code": source_code,
            "adversarial_score": None,
            "original_score": original_score,
            "error": str(e)
        }

def save_results(results, output_path, score_distributions):
    """Save all results and statistics."""
    # Save main results
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    # Save search statistics
    stats_path = output_path.replace('.jsonl', '_search_stats_per_sample.json')
    all_stats = {str(res.get('id', 'unknown')): res.get('search_stats', {}) 
                 for res in results if 'search_stats' in res}
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    
    # Save score distributions
    dist_path = output_path.replace('.jsonl', '_score_distributions.json')
    with open(dist_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in score_distributions.items()}, f, indent=2)

def generate_visualizations(logger, score_distributions, original_scores, output_dir, results):
    """Generate all visualizations."""
    analyze_score_distributions(logger, score_distributions, output_dir=output_dir)
    generate_per_sample_histograms(logger, score_distributions, original_scores=original_scores, 
                                  base_output_dir=output_dir)
    if len(score_distributions) > 1:
        generate_histogram_grid(logger, score_distributions, original_scores=original_scores, 
                               output_dir=output_dir)
    analyze_adversarial_vs_normal_scores(logger, results, original_scores, output_dir=output_dir)



def main() -> None:
    """Main function with improved structure."""
    parser = argparse.ArgumentParser(description="Find adversarial code reorderings using batched search.")
    parser.add_argument("--input", type=str, default="data/incomplete_code_summarization_data_python.jsonl")
    parser.add_argument("--output", type=str, default="adversarial_reorderings.jsonl")
    parser.add_argument("--judge_model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--completion_model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--sample_limit", type=int, default=50)
    parser.add_argument("--initial_samples", type=int, default=3)
    parser.add_argument("--neighborhood_samples", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--change_probability", type=float, default=0.3)
    parser.add_argument("--finder_start_batch_size", type=int, default=8)
    parser.add_argument("--use_8bit", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Setup output directory
    completion_model_name = args.completion_model.split('/')[-1]
    output_dir = f"results_{completion_model_name}"
    os.makedirs(output_dir, exist_ok=True)
    args.output = os.path.join(output_dir, os.path.basename(args.output))
    logger.info(f"Results will be saved to directory: {output_dir}")
    
    # Setup models with batch size determination
    models_config = setup_models_with_batch_size(args)
    
    # Create evaluator
    evaluator = LLMEvaluator(
        logger= logger,
        judge_model=models_config["judge_model"],
        judge_tokenizer=models_config["judge_tokenizer"],
        completion_model=models_config["completion_model"],
        completion_tokenizer=models_config["completion_tokenizer"],
        completion_model_name=completion_model_name,
        max_batch_size=models_config["max_batch_size"]
    )
    
    # Create searcher
    searcher = AdversarialReorderingSearch(
        logger = logger,
        evaluation_function_batch=evaluator.evaluate_batch,
        initial_samples=args.initial_samples,
        neighborhood_samples=args.neighborhood_samples,
        iterations=args.iterations,
        change_probability=args.change_probability,
    )
    
    # Process dataset
    results = process_dataset(args, evaluator, searcher, output_dir)
    
    logger.info(f"\nProcessing complete. Results saved to {args.output}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.checkpoint') 
    warnings.filterwarnings("ignore", category=FutureWarning) 
    main()