#!/usr/bin/env python
from typing import Optional
import ast
import re

def extract_code_from_response(response: str) -> str:
    """Extract code from LLM response."""
    if not response:
        return ""
    assistant_part = response
    if "assistant\n" in response:
        assistant_part = response.split("assistant\n", 1)[1].strip()
    code_blocks = re.findall(r"```python\n([\s\S]*?)```", assistant_part)
    if code_blocks:
        return code_blocks[-1].strip()
    return assistant_part.strip()

def validate_completion(completed_code: str, incomplete_code: str) -> bool:
    """Validate that code was actually completed."""
    if not completed_code:
        return False
    incomplete_pass_count = incomplete_code.count("pass")
    completed_pass_count = completed_code.count("pass")
    if completed_pass_count < incomplete_pass_count:
        return True
    if len(completed_code) > len(incomplete_code) * 1.2:
        return True
    return False

def extract_rating(response_text: str) -> Optional[str]:
    """Extract rating from judge response."""
    patterns = [
        r'RATING:\s*([1-5])\b',
        r'\b([1-5])\b',
        r'model\s*(\d+)\s*$'
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text.strip())
        if match:
            rating = match.group(1)
            if 1 <= int(rating) <= 5:
                return rating
    return None

def has_perturbation_potential(code_str: str) -> bool:
    """Check if code has enough statements for meaningful reordering."""
    try:
        tree = ast.parse(code_str)
        top_level_stmts = [node for node in tree.body 
                          if not (isinstance(node, ast.Expr) and 
                                 isinstance(node.value, (ast.Constant, ast.Str)))]
        return len(top_level_stmts) >= 3
    except (SyntaxError, Exception):
        return False