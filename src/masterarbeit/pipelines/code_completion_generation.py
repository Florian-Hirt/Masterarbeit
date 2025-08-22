#!/usr/bin/env python
import re
import json
import torch
import logging
import argparse
import warnings
import ast


class IncompleteTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.modified = False

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # Check if the function has a docstring as the first statement
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0]
            node.body = [docstring, ast.Pass()]
        else:
            node.body = [ast.Pass()]
        # Mark that a function body was substituted with a pass
        self.modified = True
        return node


def incomplete_code_transform(code_str):
    try:
        tree = ast.parse(code_str)
        transformer = IncompleteTransformer()
        tree = transformer.visit(tree)
        # Only return the transformed code if at least one function was modified.
        if not transformer.modified:
            return None
        incomplete_code = ast.unparse(tree)
        return incomplete_code
    except Exception as e:
        print(f"Transformation failed: {e}")
        return None
    

def main():
    in_file = "data/code_summarization_data_python.jsonl"
    out_file = "data/incomplete_code_summarization_data_python.jsonl"
    
    total_samples = 0
    successful = 0
    skipped = 0

    with open(in_file, "r") as f, open(out_file, "w") as g:
        for line in f:
            total_samples += 1
            example = json.loads(line)
            original_code = example.get("source_code", "")
            transformed_code = incomplete_code_transform(original_code)
            # Only consider samples where at least one function was modified.
            if transformed_code is None:
                skipped += 1
                continue
            example["incomplete_code"] = transformed_code
            json.dump(example, g)
            g.write("\n")
            successful += 1

    print("Total samples processed:", total_samples)
    print("Successfully transformed samples:", successful)
    print("Skipped samples:", skipped)


if __name__ == '__main__':
    main()