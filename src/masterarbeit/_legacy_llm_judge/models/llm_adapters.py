#!/usr/bin/env python
from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, Gemma3ForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig
from tenacity import retry, stop_after_attempt, wait_random_exponential
import traceback
import logging

# Module logger
logger = logging.getLogger(__name__)


def setup_llm(
    logger,
    checkpoint: str = "google/gemma-3-1b-it",
    access_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    role: str = "judge",
    use_8bit: bool = False
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure pad_token_id is within vocabulary bounds
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= len(tokenizer):
        logger.warning(f"pad_token_id {tokenizer.pad_token_id} >= vocab_size {len(tokenizer)}, resetting to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    if "gemma-3" in checkpoint.lower():
        logger.info(f"Detected Gemma 3 model, using Gemma3ForCausalLM")
        model_class = Gemma3ForCausalLM
    else:
        logger.info(f"Using AutoModelForCausalLM for model: {checkpoint}")
        model_class = AutoModelForCausalLM

    # Create model loading arguments
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "token": access_token,
        "cache_dir": cache_dir
    }
    
    if use_8bit:
        logger.info(f"Loading {checkpoint} with 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        # Only set torch_dtype if not using quantization
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = model_class.from_pretrained(checkpoint, **model_kwargs)

    if use_8bit and hasattr(model, 'resize_token_embeddings'):
        vocab_size = len(tokenizer)
        model_vocab_size = model.get_input_embeddings().weight.shape[0]
        if vocab_size > model_vocab_size:
            logger.warning(f"Tokenizer vocab size ({vocab_size}) > model vocab size ({model_vocab_size}). Resizing embeddings.")
            model.resize_token_embeddings(vocab_size)

    # Set generation config
    if not use_8bit and hasattr(model, 'generation_config') and model.generation_config.do_sample is False:
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    logger.info(f"{role.capitalize()} model loaded. Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    return model, tokenizer, device


def generate_text_no_retry(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Union[str, List[str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    logger_instance: Optional[logging.Logger] = None
) -> List[str]:
    """Generate text WITHOUT retry decorator for better error visibility."""
    # Use module logger if no logger instance provided
    if logger_instance is None:
        logger_instance = logger
        
    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    logger_instance.debug(f"generate_text called with {len(prompts)} prompts")
    logger_instance.debug(f"First prompt (truncated): {prompts[0][:200]}...")
    
    try:
        # Tokenize with proper error handling
        logger_instance.debug("Starting tokenization...")
        
        # Get model max length
        max_length = getattr(tokenizer, 'model_max_length', 4096)
        if max_length > 100000:  # Some tokenizers have unrealistic max lengths
            max_length = 4096
            
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False  # Assuming chat template already added them
        )
        logger_instance.debug(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}")
        
        # Check for token bounds
        max_token_id = inputs['input_ids'].max().item() if inputs['input_ids'].numel() > 0 else 0
        vocab_size = len(tokenizer)
        
        # Get model vocab size properly
        if hasattr(model, 'get_input_embeddings'):
            model_vocab_size = model.get_input_embeddings().weight.shape[0]
        else:
            model_vocab_size = vocab_size
            
        logger_instance.debug(f"Max token ID: {max_token_id}, Tokenizer vocab: {vocab_size}, Model vocab: {model_vocab_size}")
        
        if max_token_id >= model_vocab_size:
            logger_instance.warning(f"Token ID {max_token_id} exceeds model vocabulary size {model_vocab_size}")
            # Clamp tokens
            inputs['input_ids'] = torch.clamp(inputs['input_ids'], max=model_vocab_size - 1)
            logger_instance.warning("Clamped out-of-bounds token IDs")
        
        # Move to device
        logger_instance.debug("Moving inputs to device...")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            # For models with device_map
            input_device = next(iter(model.hf_device_map.values()))
            # Convert device string to actual device
            if isinstance(input_device, str):
                input_device = torch.device(input_device)
            elif isinstance(input_device, int):
                input_device = torch.device(f"cuda:{input_device}")
                
            inputs = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            logger_instance.debug(f"Inputs moved to device: {input_device}")
        else:
            # For regular models
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            logger_instance.debug(f"Inputs moved to device: {device}")
        
        # Prepare generation kwargs
        generate_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs.get('attention_mask'),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # Remove None values
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False
            
        logger_instance.debug(f"Generation kwargs: {list(generate_kwargs.keys())}")
        
        # Generate
        logger_instance.debug("Starting generation...")
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        logger_instance.debug(f"Generation complete. Output shape: {outputs.shape}")
        
        # Decode responses
        responses = []
        input_length = inputs['input_ids'].shape[1]
        for i, output in enumerate(outputs):
            # Only decode the generated part
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(response)
            logger_instance.debug(f"Response {i+1} length: {len(response)} chars")
            logger.debug(f"Responses {responses}")
            
        return responses
        
    except Exception as e:
        logger_instance.error(f"Error in generate_text_no_retry: {type(e).__name__}: {str(e)}")
        logger_instance.error(f"Full traceback:\n{traceback.format_exc()}")
        raise

# Wrap with retry decorator
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Union[str, List[str]],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    logger_instance: Optional[logging.Logger] = None
) -> List[str]:
    """Generate text with retry logic."""
    return generate_text_no_retry(model, tokenizer, prompts, max_new_tokens, temperature, logger_instance)