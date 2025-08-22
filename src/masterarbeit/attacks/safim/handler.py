#!/usr/bin/env python

import ast
import re
import textwrap
import logging

logger = logging.getLogger(__name__)

class ImprovedSaFiMCompleter:
    """Improved version with better error handling and debugging"""
    
    def __init__(self, model, tokenizer, batch_size, max_tokens, temp):
        self.model = model
        self.tokenizer = tokenizer 
        self.max_batch_size = batch_size
        self.completion_max_new_tokens = max_tokens
        self.completion_temperature = temp
        self.completion_cache = {}

    def extract_code_from_llm_output(self, raw_output: str) -> str:
        """Enhanced code extraction with better handling of different formats"""
        if not raw_output or not raw_output.strip():
            return "pass"
        
        # Remove common prefixes
        prefixes = [
            "Here's the code:", "Here is the code:", "The code is:", "Solution:",
            "Here's the replacement:", "The replacement code is:", "Replace with:",
            "Code to replace TODO:", "Replacement for TODO:", "Here's what goes in the TODO:",
            "The missing code is:", "Fill in with:", "Complete with:"
        ]
        
        cleaned = raw_output.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Extract from markdown code blocks (most common case)
        code_block_patterns = [
            r"```(?:python)?\s*\n(.*?)```",  # ```python ... ``` or ``` ... ```
            r"```(.*?)```",                   # Any ``` ... ``` block
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code and code != "pass":
                    return code
        
        # If no markdown blocks, try to extract meaningful code lines
        lines = cleaned.split('\n')
        code_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, comments, and obvious non-code
            if not stripped or stripped.startswith('#'):
                continue
            # Skip explanatory text (heuristic)
            if any(word in stripped.lower() for word in ['here', 'this', 'should', 'will', 'the code']):
                continue
            # This looks like code
            if any(char in stripped for char in ['=', '(', ')', ':', 'if', 'for', 'while', 'def', 'return']):
                code_lines.append(line.rstrip())
        
        if code_lines:
            # Try to maintain relative indentation
            if len(code_lines) > 1:
                # Find minimum indentation (excluding first line)
                min_indent = float('inf')
                for line in code_lines[1:]:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        min_indent = min(min_indent, indent)
                
                # Remove common indentation
                if min_indent != float('inf') and min_indent > 0:
                    dedented_lines = [code_lines[0]]  # Keep first line as-is
                    for line in code_lines[1:]:
                        if line.strip():
                            dedented_lines.append(line[min_indent:])
                        else:
                            dedented_lines.append('')
                    code_lines = dedented_lines
            
            result = '\n'.join(code_lines)
            # Validate the extracted code
            try:
                # Try to parse as a statement or expression
                ast.parse(result)
                return result
            except SyntaxError:
                # If it fails, try wrapping in a function
                try:
                    wrapped = f"def temp_func():\n" + textwrap.indent(result, "    ")
                    ast.parse(wrapped)
                    return result
                except SyntaxError:
                    # Last resort: return something simple
                    return "pass"
        
        # Final fallback
        return "pass"

    def substitute_todo_with_completion(self, skeleton: str, completion: str) -> str:
        """Improved TODO substitution with better indentation handling"""
        
        # Find the TODO marker
        todo_markers = [
            "# TODO: Your code here", "# TODO: write your code here", 
            "# TODO: Implement this", "# TODO: finish this",
            "#TODO: Your code here", "# TODO: Add your code here"
        ]
        
        marker_found = None
        for marker in todo_markers:
            if marker in skeleton:
                marker_found = marker
                break
        
        if not marker_found:
            logger.warning("No TODO marker found for substitution")
            return skeleton + "\n" + completion
        
        # Clean the completion
        completion = completion.strip()
        if not completion or completion.lower() in ["pass", "none", "..."]:
            completion = "pass"
        
        # Split skeleton into lines
        lines = skeleton.splitlines()
        result_lines = []
        
        for i, line in enumerate(lines):
            if marker_found in line:
                # Get the indentation level of the TODO line
                leading_space = line[:line.find(marker_found)]
                
                # Handle the completion
                completion_lines = completion.splitlines()
                
                if len(completion_lines) == 1:
                    # Single line completion
                    result_lines.append(leading_space + completion_lines[0].lstrip())
                else:
                    # Multi-line completion - preserve relative indentation
                    first_line = completion_lines[0].lstrip()
                    result_lines.append(leading_space + first_line)
                    
                    # For subsequent lines, maintain their relative indentation
                    if len(completion_lines) > 1:
                        # Find the base indentation of the completion (from second line)
                        base_indent = 0
                        for comp_line in completion_lines[1:]:
                            if comp_line.strip():
                                base_indent = len(comp_line) - len(comp_line.lstrip())
                                break
                        
                        for comp_line in completion_lines[1:]:
                            if comp_line.strip():
                                # Remove base indentation and add our target indentation
                                relative_indent = len(comp_line) - len(comp_line.lstrip()) - base_indent
                                new_line = leading_space + "    " + " " * max(0, relative_indent) + comp_line.lstrip()
                                result_lines.append(new_line)
                            else:
                                result_lines.append("")
            else:
                result_lines.append(line)
        
        result = "\n".join(result_lines)
        
        # Validate the result
        try:
            ast.parse(result)
            return result
        except SyntaxError as e:
            logger.warning(f"Syntax error after substitution: {e}")
            # Try a simpler substitution as fallback
            simple_result = skeleton.replace(marker_found, "pass  # SUBSTITUTION_ERROR")
            return simple_result

    def complete_safim_skeletons_batch(self, descs, skels):
        """Main completion method with improved error handling"""
        if not skels:
            return []
        
        final_codes = [""] * len(skels)
        to_process_indices = []
        llm_descs = []
        llm_skels = []
        
        # Check cache first
        for i, (desc, skel) in enumerate(zip(descs, skels)):
            cache_key = f"{desc}|||{skel}"
            if cache_key in self.completion_cache:
                final_codes[i] = self.completion_cache[cache_key]
            else:
                to_process_indices.append(i)
                llm_descs.append(desc)
                llm_skels.append(skel)
        
        if not llm_skels:
            return final_codes
        
        logger.info(f"Processing {len(llm_skels)} skeletons with LLM...")
        
        # Process in batches
        all_completions = []
        for batch_start in range(0, len(llm_skels), self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, len(llm_skels))
            batch_descs = llm_descs[batch_start:batch_end]
            batch_skels = llm_skels[batch_start:batch_end]
            
            try:
                batch_completions = self._generate_completions_batch(batch_descs, batch_skels)
                all_completions.extend(batch_completions)
            except Exception as e:
                logger.error(f"Batch completion failed: {e}")
                # Fallback for this batch
                all_completions.extend(["pass  # LLM_ERROR"] * len(batch_skels))
        
        # Process results
        for i, completion in enumerate(all_completions):
            orig_idx = to_process_indices[i]
            desc = llm_descs[i]
            skel = llm_skels[i]
            
            # Extract and clean the completion
            cleaned_completion = self.extract_code_from_llm_output(completion)
            
            # Substitute into skeleton
            final_code = self.substitute_todo_with_completion(skel, cleaned_completion)
            
            # Cache and store result
            cache_key = f"{desc}|||{skel}"
            self.completion_cache[cache_key] = final_code
            final_codes[orig_idx] = final_code
            
            # Additional validation
            try:
                ast.parse(final_code)
                logger.debug(f"Successfully completed skeleton {orig_idx}")
            except SyntaxError as e:
                logger.warning(f"Final code has syntax error for skeleton {orig_idx}: {e}")
                logger.debug(f"Final code:\n{final_code[:500]}...")
        
        return final_codes

    def _generate_completions_batch(self, descs, skels):
        """Generate completions using the LLM - improved prompting"""
        prompts = []
        
        for desc, skel in zip(descs, skels):
            # More focused prompt
            prompt_text = f"""Complete the Python code by replacing the TODO comment with appropriate implementation.

Problem: {desc}

Code to complete:
{skel}

Instructions:
- Replace ONLY the TODO comment with working code
- Maintain proper indentation
- Do not include explanations or the full program
- Provide only the code that should replace the TODO

Completion:"""
            
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted_prompt)
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        generation_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs.get('attention_mask'),
            "max_new_tokens": self.completion_max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.completion_temperature > 0,
        }
        
        if self.completion_temperature > 0:
            generation_kwargs.update({
                "temperature": self.completion_temperature,
                "top_k": 40,
                "top_p": 0.95
            })
        
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        
        # Decode outputs
        completions = []
        input_length = inputs['input_ids'].shape[1]
        
        for output in outputs:
            generated_tokens = output[input_length:]
            completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            completions.append(completion.strip())
        
        return completions


# Additional debugging utilities
def debug_completion_process(skeleton: str, completion: str, final_code: str):
    """Helper function to debug the completion process"""
    print("=== DEBUG COMPLETION PROCESS ===")
    print(f"Original skeleton:\n{skeleton}\n")
    print(f"LLM completion:\n{completion}\n")
    print(f"Final code:\n{final_code}\n")
    
    try:
        ast.parse(final_code)
        print("✓ Final code is syntactically valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in final code: {e}")
        print(f"Error occurred around line {e.lineno}")
    
    print("=== END DEBUG ===\n")


def validate_and_fix_code(code: str) -> str:
    """Attempt to automatically fix common syntax issues"""
    try:
        ast.parse(code)
        return code
    except SyntaxError as e:
        # Try some common fixes
        lines = code.splitlines()
        
        # Fix 1: Remove duplicate lines (common issue we saw)
        seen_lines = []
        fixed_lines = []
        for line in lines:
            if line.strip() and line not in seen_lines:
                seen_lines.append(line)
                fixed_lines.append(line)
            elif not line.strip():
                fixed_lines.append(line)
        
        try:
            fixed_code = '\n'.join(fixed_lines)
            ast.parse(fixed_code)
            return fixed_code
        except SyntaxError:
            pass
        
        # Fix 2: Try to dedent if there are indentation issues
        try:
            dedented = textwrap.dedent(code)
            ast.parse(dedented)
            return dedented
        except SyntaxError:
            pass
        
        # If all fixes fail, return a safe fallback
        return "pass  # AUTO_FIX_FAILED"