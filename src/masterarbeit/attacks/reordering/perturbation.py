# File: perturbation.py
"""
This module implements various transformations and perturbations on Python code.
MODIFICATIONS:
- Removed duplicated AST/graph utility functions; now imports from Dataflow_parser.
- gast.ExceptionHandler changed to ast.ExceptHandler (via 'import gast as ast').
- Integrated placeholder strategy for ground truth completion.
- Logging refined.
"""

import ast as py_ast # Python's own ast for specific non-gast tasks if needed
import gast # Direct import of gast
import random
from collections import deque
import astunparse
import networkx as nx
import string
import builtins
import os
import sys
import multiprocessing as mp
import logging
from typing import Optional, Tuple, Any, Dict, Set, List # Added more types
import uuid # For unique marker
import textwrap
import re

# Update system path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import centralized utilities from Dataflow_parser
from digraph_transformer import dataflow_parser as dfp # Using an alias for clarity
from python_graphs.program_graph_dataclasses import EdgeType
from python_graphs import program_graph_dataclasses as pb  

# Get a logger instance
logger = logging.getLogger(__name__)

# Define a unique string that will be used to mark the replacement site for the ground truth
# This string should be highly unlikely to appear naturally in code.
# It will be inserted as a gast.Constant value, so its unparsed form will be a Python string.
PERTURBATION_PLACEHOLDER_STR_RAW = f"__GROUND_TRUTH_REPLACEMENT_SITE__{uuid.uuid4().hex}__"

# =============================================================================
# Graph and AST Utilities (Now mostly imported from Dataflow_parser)
# =============================================================================

def non_deterministic_topological_sort(graph):
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains a cycle!")
    graph_copy = graph.copy()
    topo_order = []
    sources = deque([node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0])
    while sources:
        random_index = random.randint(0, len(sources) - 1)
        node = sources[random_index]
        sources.remove(node) 
        topo_order.append(node)
        successors = list(graph_copy.successors(node)) 
        for successor in successors:
            graph_copy.remove_edge(node, successor)
            if graph_copy.in_degree(successor) == 0:
                sources.append(successor)
        graph_copy.remove_node(node) 
    return topo_order

def neighborhood_topological_sort(graph, current_sort, change_probability=0.3):
    position_map = {node: idx for idx, node in enumerate(current_sort)}
    graph_copy = graph.copy()
    new_sort = []
    sources = deque([node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0])
    while sources:
        if len(sources) > 1 and random.random() < change_probability:
            random_index = random.randint(0, len(sources) - 1)
            node = sources[random_index]
        else:
            node = min(sources, key=lambda n: position_map.get(n, float('inf')))
        sources.remove(node)
        new_sort.append(node)
        successors = list(graph_copy.successors(node))
        for successor in successors:
            graph_copy.remove_edge(node, successor)
            if graph_copy.in_degree(successor) == 0:
                sources.append(successor)
        graph_copy.remove_node(node)
    if not is_valid_topological_sort(graph, new_sort):
        # Use logger if available
        current_logger = logging.getLogger(__name__)
        current_logger.warning("Invalid topological sort generated in neighborhood. Returning the original sort.")
        return current_sort
    return new_sort

def is_valid_topological_sort(graph: nx.DiGraph, sort_order: List[Any]) -> bool: # Added types
    if set(sort_order) != set(graph.nodes()):
        logger.debug(f"Invalid sort: Node sets differ. Expected {len(graph.nodes())}, got {len(sort_order)}")
        return False
    position = {node: idx for idx, node in enumerate(sort_order)}
    for u, v in graph.edges():
        if position.get(u, -1) >= position.get(v, -1): # Use .get for safety
            return False
    return True

# =============================================================================
# AST Reconstruction Classes and Functions
# =============================================================================
class GroundTruthPlaceholderInserter(gast.NodeTransformer):
    def __init__(self, gt_node_ids: Set[int], placeholder_text: str, graph, topo_sort):
        self.gt_node_ids = gt_node_ids
        self.placeholder_stmt = gast.Expr(value=gast.Constant(value=placeholder_text, kind=None))
        self.graph = graph # ProgramGraph
        self.topo_sort = topo_sort # List of ProgramGraphNode IDs
        self.done_replacing = False
        self.entry_gt_node_ast_id = None

        # Determine the AST ID of the first statement of the ground truth based on topological sort
        if self.gt_node_ids and self.graph and self.topo_sort:
            pg_nodes = getattr(self.graph, 'nodes', None)
            if pg_nodes:
                for node_id_in_sort in self.topo_sort:
                    if node_id_in_sort in pg_nodes and hasattr(pg_nodes[node_id_in_sort], 'ast_node'):
                        ast_node_candidate = pg_nodes[node_id_in_sort].ast_node
                        if id(ast_node_candidate) in self.gt_node_ids and isinstance(ast_node_candidate, gast.stmt):
                            self.entry_gt_node_ast_id = id(ast_node_candidate)
                            break
        if not self.entry_gt_node_ast_id:
             logger.warning("GT Placeholder: Could not identify a clear entry statement for ground truth replacement.")


    def visit_list_of_statements(self, body_list: List[gast.AST]) -> List[gast.AST]:
        if not isinstance(body_list, list): return body_list
        new_body = []
        gt_block_active = False
        for stmt_idx, stmt in enumerate(body_list):
            if not isinstance(stmt, gast.AST):
                new_body.append(stmt) # Keep non-AST items (e.g. strings in Module docstring)
                continue

            is_current_stmt_gt_entry = (not self.done_replacing and 
                                        self.entry_gt_node_ast_id and 
                                        id(stmt) == self.entry_gt_node_ast_id)

            if is_current_stmt_gt_entry:
                new_body.append(self.placeholder_stmt)
                self.done_replacing = True
                gt_block_active = True # Start skipping subsequent GT nodes in this block
            elif gt_block_active and id(stmt) in self.gt_node_ids:
                # This statement is part of the ground truth block but not the entry, skip it.
                continue
            else:
                gt_block_active = False # No longer in a GT block (or was never in one)
                new_body.append(self.visit(stmt)) # Visit children of non-GT statements
        return new_body

    # Need to override visit_X for all X that have a 'body' or list of statements
    def visit_Module(self, node: gast.Module) -> gast.Module:
        node.body = self.visit_list_of_statements(node.body)
        # self.generic_visit(node) # Don't generic_visit Module if body is handled
        return node

    def visit_FunctionDef(self, node: gast.FunctionDef) -> gast.FunctionDef:
        node.body = self.visit_list_of_statements(node.body)
        # Process decorators, args, return annotations if they aren't part of GT
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        if node.args: node.args = self.visit(node.args)
        if hasattr(node, 'returns') and node.returns: node.returns = self.visit(node.returns)
        return node
    
    def visit_AsyncFunctionDef(self, node: gast.AsyncFunctionDef) -> gast.AsyncFunctionDef:
        node.body = self.visit_list_of_statements(node.body)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        if node.args: node.args = self.visit(node.args)
        if hasattr(node, 'returns') and node.returns: node.returns = self.visit(node.returns)
        return node

    def visit_ClassDef(self, node: gast.ClassDef) -> gast.ClassDef:
        node.body = self.visit_list_of_statements(node.body)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def visit_If(self, node: gast.If) -> gast.If:
        node.test = self.visit(node.test)
        node.body = self.visit_list_of_statements(node.body)
        node.orelse = self.visit_list_of_statements(node.orelse) if node.orelse else []
        return node
    
    def visit_For(self, node: gast.For) -> gast.For: # gast.For
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self.visit_list_of_statements(node.body)
        node.orelse = self.visit_list_of_statements(node.orelse) if node.orelse else []
        return node

    def visit_While(self, node: gast.While) -> gast.While: # gast.While
        node.test = self.visit(node.test)
        node.body = self.visit_list_of_statements(node.body)
        node.orelse = self.visit_list_of_statements(node.orelse) if node.orelse else []
        return node

    def visit_Try(self, node: gast.Try) -> gast.Try: # gast.Try
        node.body = self.visit_list_of_statements(node.body)
        new_handlers = []
        for handler in node.handlers: # gast.ExceptHandler
            if isinstance(handler, gast.ExceptHandler): # Check it's the correct type
                handler.body = self.visit_list_of_statements(handler.body)
                if handler.type: handler.type = self.visit(handler.type)
                # handler.name is str or None
            new_handlers.append(handler) # keep handler even if not ExceptHandler type (should not happen)
        node.handlers = new_handlers
        node.orelse = self.visit_list_of_statements(node.orelse) if node.orelse else []
        node.finalbody = self.visit_list_of_statements(node.finalbody) if node.finalbody else []
        return node

    def visit_With(self, node: gast.With) -> gast.With: # gast.With
        new_items = []
        for item in node.items: # item is gast.withitem
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars: item.optional_vars = self.visit(item.optional_vars)
            new_items.append(item)
        node.items = new_items
        node.body = self.visit_list_of_statements(node.body)
        return node

class ASTReconstructor(gast.NodeTransformer):
    def __init__(self, graph, topo_sort, show_reorderings=False, ground_truth_node_ids=None, logger_instance=None): 
        self.graph = graph
        self.topo_sort = topo_sort # List of ProgramGraphNode IDs
        self.ast_map = {node_id: graph.nodes[node_id].ast_node for node_id in topo_sort if node_id in graph.nodes and hasattr(graph.nodes[node_id], 'ast_node')}
        self.show_reorderings = show_reorderings
        self.reordering_counts = {} if show_reorderings else None
        self.ground_truth_node_ids = ground_truth_node_ids if ground_truth_node_ids is not None else set()
        self.logger = logger_instance if logger_instance else logging.getLogger(__name__) # Renamed from logger to logger_instance

    def _is_likely_string_producing_node(self, ast_node):
        if isinstance(ast_node, gast.Constant) and isinstance(ast_node.value, str):
            return True
        if isinstance(ast_node, gast.Call):
            if isinstance(ast_node.func, gast.Name) and ast_node.func.id == 'str':
                return True
        if isinstance(ast_node, gast.JoinedStr): 
            return True
        return False

    def visit_Module(self, node: gast.Module) -> gast.Module:
        self.generic_visit(node) # Visit children first to reconstruct them
        node.body = self.reorder_body(node.body)
        return node

    def visit_FunctionDef(self, node: gast.FunctionDef):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        return node

    def visit_ClassDef(self, node: gast.ClassDef) -> gast.ClassDef:
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        return node
    
    def visit_If(self, node: gast.If):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        node.orelse = self.reorder_body(node.orelse)
        return node
    
    def visit_For(self, node: gast.For):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        node.orelse = self.reorder_body(node.orelse)
        return node

    def visit_While(self, node: gast.While):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        node.orelse = self.reorder_body(node.orelse)
        return node
    
    def visit_Try(self, node: gast.Try):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        for handler in node.handlers:
            handler.body = self.reorder_body(handler.body)
        node.orelse = self.reorder_body(node.orelse)
        node.finalbody = self.reorder_body(node.finalbody)
        return node
    
    def visit_With(self, node: gast.With):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        return node
    
    def visit_BinOp(self, node: gast.BinOp) -> gast.BinOp: # gast.BinOp
        self.generic_visit(node) # Visit children first to reconstruct them
        is_ground_truth_binop = id(node) in self.ground_truth_node_ids
        
        op_name = node.op.__class__.__name__
        can_swap = False
        if op_name in ("Mult", "BitOr", "BitXor", "BitAnd"): 
            can_swap = True
        elif op_name == "Add":
            is_left_str = self._is_likely_string_producing_node(node.left)
            is_right_str = self._is_likely_string_producing_node(node.right)
            if not (is_left_str or is_right_str): # Don't swap string additions
                can_swap = True 
        
        if can_swap:
            if self.show_reorderings and id(node) not in self.reordering_counts: 
                self.reordering_counts[id(node)] = 2 # Mark as having 2 potential orders
            if not is_ground_truth_binop and random.choice([True, False]):
                node.left, node.right = node.right, node.left
        return node
    
    def visit_BoolOp(self, node: gast.BoolOp) -> gast.BoolOp:
        self.logger.debug(f"ASTReconstructor visiting BoolOp (id: {id(node)}):")
        if node.op:
            self.logger.debug(f"  BoolOp.op type: {type(node.op).__name__}, BoolOp.op object: {node.op}")
        else:
            self.logger.error(f"  BoolOp.op is None for BoolOp id {id(node)}!") # This would be a major issue
        
        self.logger.debug(f"  BoolOp has {len(node.values)} values.")

        new_values = []
        for i, value_expr in enumerate(node.values):
            self.logger.debug(f"  BoolOp.values[{i}] BEFORE visit: type={type(value_expr).__name__}, id={id(value_expr)}")
            visited_value = self.visit(value_expr) 
            if visited_value is None:
                self.logger.error(f"  Operand {i} of BoolOp (id: {id(node)}) became None after visit! Original type: {type(value_expr).__name__}")
                # Substitute a placeholder if an operand is lost, to make debugging easier later
                new_values.append(gast.Constant(value=f"ERROR_OPERAND_{i}_WAS_NONE", kind=None)) 
            elif not isinstance(visited_value, gast.AST):
                self.logger.error(f"  Operand {i} of BoolOp (id: {id(node)}) became non-AST: {visited_value} (type: {type(visited_value)})")
                new_values.append(gast.Constant(value=f"ERROR_OPERAND_{i}_NON_AST", kind=None))
            else:
                new_values.append(visited_value)
            self.logger.debug(f"  BoolOp.values[{i}] AFTER visit: type={type(new_values[-1]).__name__}")
        
        node.values = new_values
        return node
    
    def visit_Compare(self, node: gast.Compare) -> gast.Compare: # gast.Compare
        self.generic_visit(node) # Visit children first to reconstruct them
        is_ground_truth_compare = id(node) in self.ground_truth_node_ids

        if len(node.ops) == 1 and node.ops[0].__class__.__name__ in ("Eq", "NotEq"):
            if self.show_reorderings and id(node) not in self.reordering_counts:
                self.reordering_counts[id(node)] = 2
            if not is_ground_truth_compare and random.choice([True, False]):
                node.left, node.comparators[0] = node.comparators[0], node.left
        return node
    
    def reorder_body(self, body_list: List[gast.AST]) -> List[gast.AST]: # Type hint for body_list
        if not isinstance(body_list, list) or not body_list:
            return body_list
        
        # Map original AST node objects (if they are statements in this body) to their ProgramGraphNode IDs
        ast_to_node_id = {id(pg_node.ast_node): pg_id for pg_id, pg_node in self.ast_map.items() if hasattr(pg_node, 'ast_node')}
        
        body_pg_ids = []
        valid_stmts_in_body = []
        for stmt_node in body_list:
            if not isinstance(stmt_node, gast.AST): continue 
            valid_stmts_in_body.append(stmt_node) # Keep track of actual AST statements
            stmt_ast_id = id(stmt_node)
            if stmt_ast_id in ast_to_node_id:
                body_pg_ids.append(ast_to_node_id[stmt_ast_id])        
        # If no valid ProgramGraph IDs found for statements in this body, return original
        if not body_pg_ids:
            return valid_stmts_in_body # Return list of original AST stmts

        body_pg_ids_set = set(body_pg_ids) # Use set for efficient lookups
        
        if self.show_reorderings:
            pass
        reordered_body_stmts = []
        for pg_node_id_in_topo_sort in self.topo_sort:
            if pg_node_id_in_topo_sort in body_pg_ids_set: # If this node from global sort belongs to current body
                ast_node_to_add = self.ast_map.get(pg_node_id_in_topo_sort)
                if ast_node_to_add:
                    # Crucially, ensure we only add statements that were originally in *this* body_list
                    if any(id(ast_node_to_add) == id(orig_stmt) for orig_stmt in valid_stmts_in_body):
                        if ast_node_to_add not in reordered_body_stmts: # Avoid duplicates if topo_sort has them (should not)
                            reordered_body_stmts.append(ast_node_to_add)
        
        # Ensure all original statements are present, in case topo_sort was incomplete or problematic for this sub-body
        if len(reordered_body_stmts) != len(valid_stmts_in_body):
            return valid_stmts_in_body
            
        if not reordered_body_stmts and valid_stmts_in_body : 
             pass 

        return reordered_body_stmts

def count_topological_sorts(subgraph: nx.DiGraph) -> int:
    # (Implementation from original, seems standard DP approach)
    nodes = list(subgraph.nodes())
    n = len(nodes)
    if n == 0: return 1 # No nodes, one way to sort (empty list)
    if n > 20: # Heuristic limit, DP is 2^N * N
        logger.warning(f"Subgraph too large ({n} nodes) for exact topological sort counting, returning estimated -1.")
        return -1 # Indicate too large

    index_map = {node: i for i, node in enumerate(nodes)}
    # Adjacency list for predecessors
    pred_list = [[] for _ in range(n)]
    for u, v in subgraph.edges():
        # Ensure u and v are in index_map, can happen if subgraph is not dense
        if u in index_map and v in index_map:
             pred_list[index_map[v]].append(index_map[u])
    
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        if dp[mask] == 0: continue # Optimization: if mask is not reachable
        for i in range(n):
            # If node i is not in current mask AND all its predecessors are in mask
            if not (mask & (1 << i)): 
                # Check predecessors
                node_i_has_all_preds_in_mask = True
                for pred_idx in pred_list[i]:
                    if not (mask & (1 << pred_idx)):
                        node_i_has_all_preds_in_mask = False
                        break
                if node_i_has_all_preds_in_mask:
                    dp[mask | (1 << i)] += dp[mask]
    return dp[(1 << n) - 1]

def count_topological_sorts_worker(subgraph, queue):
    try:
        result = count_topological_sorts(subgraph)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def count_topological_sorts_with_timeout(subgraph: nx.DiGraph, timeout: int = 10, logger_instance=None) -> Optional[int]: # Reduced default timeout
    current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    if not subgraph.nodes(): return 1
    
    # Heuristic check: if too many nodes for DP approach
    if len(subgraph.nodes()) > 20: # Align with limit in count_topological_sorts
        current_logger.info(f"Skipping exact topological sort count for large subgraph ({len(subgraph.nodes())} nodes).")
        return -1 # Placeholder for "too large to compute quickly"

    queue = mp.Queue()
    # Using try-except for Process to handle potential OS errors (e.g., too many processes)
    try:
        p = mp.Process(target=count_topological_sorts_worker, args=(subgraph, queue))
        p.start()
        p.join(timeout)
    except Exception as e:
        current_logger.error(f"Failed to start/join multiprocessing for topological sort counting: {e}")
        return None # Indicate error

    if p.is_alive():
        p.terminate()
        p.join() # Ensure termination
        current_logger.warning(f"Counting topological sorts for subgraph with {len(subgraph.nodes())} nodes timed out after {timeout}s.")
        return None # Indicate timeout
    else:
        try:
            result = queue.get_nowait() # Use get_nowait if process finished
        except mp.queues.Empty:
            current_logger.error("Multiprocessing queue was empty after join, count_topological_sorts_worker might have crashed silently.")
            return None
        except Exception as e: # Other queue errors
            current_logger.error(f"Error retrieving result from queue: {e}")
            return None

        if isinstance(result, Exception):
            current_logger.error(f"Error during counting topological sorts in worker: {result}")
            return None
        return result


def graph_to_ast(graph, topo_sort, show_reorderings=False, 
                 ground_truth_node_ids: Optional[Set[int]] = None, # AST node object IDs
                 logger_instance=None):
    current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    
    # Determine the root AST node for reconstruction (usually a Module node)
    module_ast_node_in_graph = None
    if graph.root_id in graph.nodes and hasattr(graph.nodes[graph.root_id], 'ast_node'):
         module_ast_node_in_graph = graph.nodes[graph.root_id].ast_node
    elif topo_sort and topo_sort[0] in graph.nodes and hasattr(graph.nodes[topo_sort[0]], 'ast_node'):
         # Fallback: use the AST node of the first element in topo_sort if it's the module
         # This is less reliable but can be a guess if root_id is problematic
         candidate_root = graph.nodes[topo_sort[0]].ast_node
         if isinstance(candidate_root, gast.Module):
             module_ast_node_in_graph = candidate_root
             current_logger.debug("Used first node in topo_sort as module root for AST reconstruction.")
         else: # If the first node isn't a Module, search for it.
             for node_id in topo_sort:
                 if node_id in graph.nodes and hasattr(graph.nodes[node_id], 'ast_node'):
                     node_obj = graph.nodes[node_id].ast_node
                     if isinstance(node_obj, gast.Module):
                         module_ast_node_in_graph = node_obj
                         current_logger.debug(f"Found Module AST node via topo_sort (id: {node_id}) for reconstruction.")
                         break
    
    if not module_ast_node_in_graph or not isinstance(module_ast_node_in_graph, gast.Module):
        current_logger.error("Cannot determine the root gast.Module AST node for reconstruction. Aborting graph_to_ast.")
        # Try to find any gast.Module in the graph as a last resort
        for pg_node_id, pg_node_obj in graph.nodes.items():
            if hasattr(pg_node_obj, 'ast_node') and isinstance(pg_node_obj.ast_node, gast.Module):
                module_ast_node_in_graph = pg_node_obj.ast_node
                current_logger.warning(f"Fallback: Found a gast.Module node (ID: {pg_node_id}) to use as root.")
                break
        if not module_ast_node_in_graph or not isinstance(module_ast_node_in_graph, gast.Module):
            raise ValueError("Failed to find a valid gast.Module root for AST reconstruction.")


    reconstructor = ASTReconstructor(graph, topo_sort, show_reorderings, ground_truth_node_ids, logger_instance)
    reconstructed_ast = reconstructor.visit(module_ast_node_in_graph) # visit returns the modified node

    # Insert placeholder for ground truth completion
    if ground_truth_node_ids:
        placeholder_inserter = GroundTruthPlaceholderInserter(
            ground_truth_node_ids, 
            PERTURBATION_PLACEHOLDER_STR_RAW, 
            graph, 
            topo_sort
        )
        reconstructed_ast = placeholder_inserter.visit(reconstructed_ast)
        current_logger.debug(f"placeholder_inserter.done_replacing is {placeholder_inserter.done_replacing}")
        if not placeholder_inserter.done_replacing:
            current_logger.warning("GroundTruthPlaceholderInserter did not replace any node. Placeholder might be missing.")


    reordering_counts = reconstructor.reordering_counts if show_reorderings else None
    return reconstructed_ast, reordering_counts 

# =============================================================================
# Module Statement Reordering, Variable Renaming (Mostly from original)
# NodeFinder, get_ast_nodes_for_string (from original)
# =============================================================================

def get_name_from_gast(node): # Renamed to avoid conflict with dfp.get_name
    """Helper to get name from gast.Name or gast.arg, specific to variable_renaming context."""
    if isinstance(node, gast.Name): return node.id
    if isinstance(node, gast.arg): return node.arg # gast.arg uses .arg
    return None

def get_name(node):
    """
    Helper function to extract the name from a node.

    Args:
        node: An AST node.

    Returns:
        str or None: The name attribute from the node.
    """
    return getattr(node, 'id', getattr(node, 'arg', None))


# VariableRenaming, find_cycle (now uses dfp.would_create_cycle)
# get_ast_nodes_for_string are kept, ensuring gast usage.

def generate_new_name(length=8):
    first_char = random.choice(string.ascii_uppercase) 
    rest = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length-1))
    return first_char + rest

def collect_global_vars(ast_tree: gast.Module) -> Set[str]:
    global_vars = set()
    for node in ast_tree.body:
        if isinstance(node, gast.Assign):
            for target in node.targets:
                if isinstance(target, gast.Name): global_vars.add(target.id)
        elif isinstance(node, gast.AugAssign):
            if isinstance(node.target, gast.Name): global_vars.add(node.target.id)
    return global_vars

def variable_renaming(ast_tree: gast.AST, rename_functions=True, logger_instance=None):
    current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    defined_names = set() # Names defined by FunctionDef, ClassDef at scanned scope
    function_renames = {} # Maps original func name to new func name

    class DefCollector(gast.NodeVisitor):
        def visit_FunctionDef(self, node: gast.FunctionDef):
            defined_names.add(node.name)
            self.generic_visit(node) # Visit decorators etc.
        def visit_AsyncFunctionDef(self, node: gast.AsyncFunctionDef): # gast.AsyncFunctionDef
            defined_names.add(node.name)
            self.generic_visit(node)
        def visit_ClassDef(self, node: gast.ClassDef):
            defined_names.add(node.name)
            self.generic_visit(node)
    DefCollector().visit(ast_tree)
    
    global_vars = set()
    if isinstance(ast_tree, gast.Module):
        global_vars = collect_global_vars(ast_tree)

    builtin_names_set = set(dir(builtins)) 
    global_mapping = {} # For top-level (module) variables
    for var_name in global_vars:
        if var_name not in defined_names and var_name not in builtin_names_set:
            global_mapping[var_name] = generate_new_name(length=random.randint(3, 10))

    class VNTransformer(gast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.scope_stack = [global_mapping.copy()] # Start with global scope mapping

        def _enter_scope(self): self.scope_stack.append({})
        def _exit_scope(self): self.scope_stack.pop()

        def _get_new_name(self, old_name, is_definition=False, is_function_def=False):
            # Check current scope first, then outer scopes
            for scope in reversed(self.scope_stack):
                if old_name in scope:
                    return scope[old_name]
            
            # If not found in any scope and it's a definition, create a new name
            if is_definition:
                # Avoid renaming builtins or already defined special names unless forced (e.g. function rename)
                if not is_function_def and (old_name in builtin_names_set or old_name in defined_names):
                    return old_name # Don't rename if it's a load of a class/func name defined elsewhere
                
                new_name = generate_new_name(length=random.randint(3,10))
                while new_name in self.scope_stack[-1].values(): # Ensure uniqueness in current scope
                     new_name = generate_new_name(length=random.randint(3,10))
                self.scope_stack[-1][old_name] = new_name
                return new_name
            
            # If it's a load and not found, it's a global or builtin (or error) - don't rename here
            return old_name


        def visit_FunctionDef(self, node: gast.FunctionDef):
            original_name = node.name
            if rename_functions and original_name not in builtin_names_set :
                # Function names are defined in the PARENT scope
                parent_scope = self.scope_stack[-1]
                if original_name not in parent_scope: # If not already renamed by an outer scope alias
                    new_func_name = generate_new_name(length=random.randint(4,12))
                    while new_func_name in parent_scope.values(): new_func_name = generate_new_name(length=random.randint(4,12))
                    parent_scope[original_name] = new_func_name
                    node.name = new_func_name
                    function_renames[original_name] = new_func_name # Track for Call site updates
                else: # Already mapped in parent scope (e.g. global mapping)
                    node.name = parent_scope[original_name]
                    function_renames[original_name] = node.name


            self._enter_scope() # New scope for function body and arguments
            # Rename arguments (params) - these are definitions in the new scope
            if node.args: # gast.arguments
                for arg_node in node.args.args: # list of gast.arg
                    arg_node.arg = self._get_new_name(arg_node.arg, is_definition=True)
                if node.args.vararg: node.args.vararg.arg = self._get_new_name(node.args.vararg.arg, is_definition=True)
                if node.args.kwarg: node.args.kwarg.arg = self._get_new_name(node.args.kwarg.arg, is_definition=True)
                if hasattr(node.args, 'posonlyargs'):
                    for arg_node in node.args.posonlyargs: arg_node.arg = self._get_new_name(arg_node.arg, is_definition=True)
                if hasattr(node.args, 'kwonlyargs'):
                    for arg_node in node.args.kwonlyargs: arg_node.arg = self._get_new_name(arg_node.arg, is_definition=True)

            node.body = [self.visit(stmt) for stmt in node.body]
            self._exit_scope()
            return node

        def visit_AsyncFunctionDef(self, node: gast.AsyncFunctionDef):
            # Similar to FunctionDef
            # ... (implementation needed if async functions are common) ...
            return self.visit_FunctionDef(node) # Leverage existing logic

        def visit_ClassDef(self, node: gast.ClassDef):
            original_name = node.name
            if rename_functions and original_name not in builtin_names_set: # 'rename_functions' can apply to classes too
                parent_scope = self.scope_stack[-1]
                if original_name not in parent_scope:
                    new_class_name = generate_new_name(length=random.randint(4,12))
                    while new_class_name in parent_scope.values(): new_class_name = generate_new_name(length=random.randint(4,12))
                    parent_scope[original_name] = new_class_name
                    node.name = new_class_name
                    # No function_renames here, class renames are handled by standard scope lookup
                else:
                    node.name = parent_scope[original_name]

            self._enter_scope() # New scope for class body
            node.body = [self.visit(stmt) for stmt in node.body]
            self._exit_scope()
            return node

        def visit_Name(self, node: gast.Name):
            if isinstance(node.ctx, gast.Store):
                node.id = self._get_new_name(node.id, is_definition=True)
            elif isinstance(node.ctx, gast.Load):
                if node.id in function_renames: # Prioritize direct function rename mapping for calls
                    node.id = function_renames[node.id]
                else:
                    node.id = self._get_new_name(node.id, is_definition=False)
            # gast.Del also uses names, handle if necessary
            return node
        
        def visit_Assign(self, node: gast.Assign):
            # Visit value first, as it might use names before they are (re)defined by targets
            node.value = self.visit(node.value)
            # Then visit targets, which are definitions
            new_targets = []
            for target in node.targets:
                new_targets.append(self.visit(target)) # visit will handle Name with Store ctx
            node.targets = new_targets
            return node

        def visit_AugAssign(self, node: gast.AugAssign):
            # Target is both a load and a store
            node.target = self.visit(node.target) # Renames if it's a Name node
            node.value = self.visit(node.value)
            return node

    return VNTransformer().visit(ast_tree)


def find_cycle_in_pg(graph, logger_instance=None): # Renamed, takes ProgramGraph-like
    current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    # Convert ProgramGraph to NetworkX DiGraph for cycle detection
    nx_graph = nx.DiGraph()
    if hasattr(graph, 'nodes') and isinstance(graph.nodes, dict):
        nx_graph.add_nodes_from(graph.nodes.keys())
    else:
        current_logger.warning("find_cycle_in_pg: graph.nodes is not a dict. Cycle detection might be incomplete.")
        return None

    if hasattr(graph, 'edges') and isinstance(graph.edges, list):
        for edge in graph.edges:
            if hasattr(edge, 'id1') and hasattr(edge, 'id2'):
                 nx_graph.add_edge(edge.id1, edge.id2)
    else:
        current_logger.warning("find_cycle_in_pg: graph.edges is not a list. Cycle detection might be incomplete.")
        return None
        
    try:
        cycle = nx.find_cycle(nx_graph, orientation='original')
        return cycle
    except nx.NetworkXNoCycle:
        return None
    except Exception as e: # Catch other potential NetworkX errors
        current_logger.error(f"Error during nx.find_cycle: {e}", exc_info=True)
        return None


# In perturbation.py
import textwrap
import gast
import astunparse
import logging

# class NodeFinder(gast.NodeVisitor):
#     def __init__(self, target_code_string: str, logger_instance=None, _precomputed_normalized_strings: Optional[List[str]] = None,
#                  _precomputed_target_types: Optional[Set[type]] = None):
#         self.logger = logger_instance if logger_instance else logging.getLogger(__name__)
#         self.found_node_ids = set()
        
#         self.target_strings_normalized = [] # Canonical forms of GT parts

#         # 1. Aggressively clean the input ground truth string
#         raw_gt = target_code_string

#         # Remove common truncation markers like "..." possibly followed by a comment
#         # Handles "..." at the end of lines or the string.
#         raw_gt = re.sub(r"\.\.\.(\s*#.*)?$", "", raw_gt, flags=re.MULTILINE).strip()
        
#         # Remove entire lines that look like error messages if they are on their own
#         # This is a heuristic and might be too aggressive for some rare valid code.
#         error_patterns = [
#             re.compile(r"^\s*IndentationError:.*", flags=re.MULTILINE),
#             re.compile(r"^\s*SyntaxError:.*", flags=re.MULTILINE),
#             re.compile(r"^\s*Traceback \(most recent call last\):.*", flags=re.MULTILINE),
#             # Add other common error patterns if observed
#         ]
#         for pattern in error_patterns:
#             raw_gt = pattern.sub("", raw_gt).strip()

#         if not raw_gt:
#             self.logger.warning("NodeFinder: Target code string is empty or only ellipsis/errors/whitespace after cleaning.")
#             return

#         # 2. Attempt to parse the cleaned, dedented block as a whole module
#         try:
#             dedented_target_code = textwrap.dedent(raw_gt)
#             gt_ast_module = gast.parse(dedented_target_code)
#             for stmt_node in gt_ast_module.body:
#                 if isinstance(stmt_node, (gast.stmt, gast.expr)): # Check if it's an AST node
#                     # Unparse each statement/expression to get its canonical form
#                     # If it's an expression, gast.parse would wrap it in Expr statement.
#                     # We want the canonical form of the actual statement/expression.
#                     node_to_unparse = stmt_node
#                     if isinstance(stmt_node, gast.Expr) and isinstance(stmt_node.value, gast.expr):
#                         # If GT was 'a+b', gast.parse makes it gast.Expr(value=gast.BinOp(...))
#                         # We want to match 'a+b', so unparse the .value
#                         # However, if the target is print(a+b), it's an Expr(value=Call(...)) and should be kept as Expr
#                         # This logic needs to be careful. For now, let's unparse the statement.
#                         pass # Keep node_to_unparse as stmt_node
                    
#                     temp_module = gast.Module(body=[node_to_unparse], type_ignores=[])
#                     unparsed_form = astunparse.unparse(temp_module).strip()
#                     if unparsed_form: # Avoid adding empty strings if unparsing fails weirdly
#                         self.target_strings_normalized.append(unparsed_form)
#             if self.target_strings_normalized:
#                 self.logger.debug(f"NodeFinder (Block Parse): Normalized GT forms: {self.target_strings_normalized}")
#                 return
#         except SyntaxError as e_block:
#             self.logger.warning(f"NodeFinder: Block parse failed (Error: {e_block}). Cleaned GT: '{raw_gt[:150]}...'. Trying alternatives.")
#         except Exception as e_generic_block: # Catch other potential gast/astunparse errors
#             self.logger.error(f"NodeFinder: Generic error during block parse/unparse (Error: {e_generic_block}). Cleaned GT: '{raw_gt[:150]}...'")


#         # 3. If block parse failed, try parsing as a single expression (if it's a single line)
#         if not self.target_strings_normalized and '\n' not in raw_gt:
#             try:
#                 expr_node = gast.parse(raw_gt, mode='eval').body 
#                 unparsed_form = astunparse.unparse(expr_node).strip()
#                 if unparsed_form:
#                     self.target_strings_normalized.append(unparsed_form)
#                 if self.target_strings_normalized:
#                     self.logger.debug(f"NodeFinder (Single Expr Parse): Normalized GT: {self.target_strings_normalized}")
#                     return 
#             except SyntaxError as e_expr:
#                 self.logger.warning(f"NodeFinder: Single expression parse failed (Error: {e_expr}). Cleaned GT: '{raw_gt[:150]}...'. Trying individual lines.")
#             except Exception as e_generic_expr:
#                  self.logger.error(f"NodeFinder: Generic error during single expr parse/unparse (Error: {e_generic_expr}). Cleaned GT: '{raw_gt[:150]}...'")

#         # 4. Fallback: Try to parse line-by-line if the above failed
#         # This is the most error-prone part for multi-line statements that are not valid alone.
#         if not self.target_strings_normalized:
#             lines = raw_gt.splitlines()
#             parsed_lines_as_targets = []
#             for line_content in lines:
#                 stripped_line = line_content.strip()
#                 if not stripped_line: continue
                
#                 dedented_line = textwrap.dedent(stripped_line) # Dedent individual line
                
#                 # Try parsing as a statement
#                 try:
#                     line_ast_module = gast.parse(dedented_line) 
#                     if line_ast_module.body:
#                         temp_module = gast.Module(body=[line_ast_module.body[0]], type_ignores=[])
#                         unparsed_form = astunparse.unparse(temp_module).strip()
#                         if unparsed_form:
#                             parsed_lines_as_targets.append(unparsed_form)
#                         continue # Successfully parsed as a statement
#                 except SyntaxError:
#                     # If statement parsing fails, try as an expression
#                     try:
#                         expr_node = gast.parse(dedented_line, mode='eval').body
#                         unparsed_form = astunparse.unparse(expr_node).strip()
#                         if unparsed_form:
#                            parsed_lines_as_targets.append(unparsed_form)
#                     except SyntaxError:
#                         self.logger.warning(f"NodeFinder: Line could not be parsed as stmt or expr: '{stripped_line}'")
#                         # If a line fails, we might lose the context for a multi-line statement
#                         # For now, we just collect what we can parse line-by-line.
#                     except Exception as e_generic_line_expr:
#                         self.logger.error(f"NodeFinder: Generic error during line-expr parse/unparse: {e_generic_line_expr} for line '{stripped_line}'")

#                 except Exception as e_generic_line_stmt:
#                     self.logger.error(f"NodeFinder: Generic error during line-stmt parse/unparse: {e_generic_line_stmt} for line '{stripped_line}'")


#             if parsed_lines_as_targets:
#                 self.target_strings_normalized = parsed_lines_as_targets
#                 self.logger.debug(f"NodeFinder (Line-by-Line Parse): Normalized GT forms: {self.target_strings_normalized}")
#             else:
#                 # Very last resort: use the cleaned raw_gt if NOTHING could be parsed
#                 self.logger.error(f"NodeFinder: All parsing attempts failed for GT. Using cleaned raw string as target: '{raw_gt[:150]}...'")
#                 if raw_gt: # Ensure it's not empty
#                     self.target_strings_normalized.append(raw_gt)

#     def visit(self, node): # visit method from previous good version
#         if not self.target_strings_normalized:
#             super().generic_visit(node)
#             return

#         current_unparsed_form = ""
#         match_found = False

#         try:
#             # Unparse based on node type
#             if isinstance(node, gast.stmt):
#                 temp_module = gast.Module(body=[node], type_ignores=[]) # Wrap stmt for unparsing
#                 current_unparsed_form = astunparse.unparse(temp_module).strip()
#             elif isinstance(node, gast.expr): # Includes Call, Name, BinOp etc.
#                 current_unparsed_form = astunparse.unparse(node).strip()
            
#             if current_unparsed_form:
#                 if current_unparsed_form in self.target_strings_normalized:
#                     match_found = True
        
#         except Exception as e_unparse:
#             # self.logger.debug(f"NodeFinder: Error unparsing node {type(node).__name__} for matching: {e_unparse}")
#             pass
        
#         if match_found:
#             # self.logger.debug(f"NodeFinder: Matched node {type(node).__name__} (ID: {id(node)}) which unparses to '{current_unparsed_form[:100]}...'")
#             for sub_node in gast.walk(node): # Add the matched node and all its children
#                 self.found_node_ids.add(id(sub_node))
#             return # Don't visit children of a matched node further
        
#         super().generic_visit(node)

class NodeFinder(gast.NodeVisitor):
    def __init__(self, target_code_string: str, logger_instance=None, 
                 # New params for using precomputed values
                 _precomputed_normalized_strings: Optional[List[str]] = None,
                 _precomputed_target_types: Optional[Set[type]] = None):
        self.logger = logger_instance if logger_instance else logging.getLogger(__name__) # Use module logger as fallback
        self.found_node_ids = set()
        
        if _precomputed_normalized_strings is not None and _precomputed_target_types is not None:
            self.target_strings_normalized = _precomputed_normalized_strings
            self.target_node_types = _precomputed_target_types
            self.logger.debug("NodeFinder initialized with precomputed normalized strings and types.")
            return

        self.target_strings_normalized = []
        self.target_node_types = set() # Store types of nodes that form the GT

        # 1. Aggressively clean the input ground truth string (existing logic)
        raw_gt = target_code_string
        raw_gt = re.sub(r"\.\.\.(\s*#.*)?$", "", raw_gt, flags=re.MULTILINE).strip()
        error_patterns = [
            re.compile(r"^\s*IndentationError:.*", flags=re.MULTILINE),
            re.compile(r"^\s*SyntaxError:.*", flags=re.MULTILINE),
            re.compile(r"^\s*Traceback \(most recent call last\):.*", flags=re.MULTILINE),
        ]
        for pattern in error_patterns:
            raw_gt = pattern.sub("", raw_gt).strip()

        if not raw_gt:
            self.logger.warning("NodeFinder.__init__: Target code string is empty or only junk after cleaning.")
            return

        # 2. Attempt to parse the cleaned, dedented block as a whole module (existing logic)
        parsed_successfully = False
        try:
            dedented_target_code = textwrap.dedent(raw_gt)
            gt_ast_module = gast.parse(dedented_target_code)
            for stmt_node in gt_ast_module.body:
                if isinstance(stmt_node, (gast.stmt, gast.expr)):
                    node_to_unparse = stmt_node
                    temp_module = gast.Module(body=[node_to_unparse], type_ignores=[])
                    unparsed_form = astunparse.unparse(temp_module).strip()
                    if unparsed_form:
                        self.target_strings_normalized.append(unparsed_form)
                        self.target_node_types.add(type(node_to_unparse)) # Capture type
            if self.target_strings_normalized:
                self.logger.debug(f"NodeFinder.__init__ (Block Parse): Normalized GT forms: {self.target_strings_normalized}")
                parsed_successfully = True
        except SyntaxError as e_block:
            self.logger.warning(f"NodeFinder.__init__: Block parse failed (Error: {e_block}). Cleaned GT: '{raw_gt[:150]}...'. Trying alternatives.")
        except Exception as e_generic_block:
            self.logger.error(f"NodeFinder.__init__: Generic error during block parse/unparse (Error: {e_generic_block}). Cleaned GT: '{raw_gt[:150]}...'")

        # 3. If block parse failed, try parsing as a single expression (existing logic)
        if not parsed_successfully and '\n' not in raw_gt:
            try:
                expr_node = gast.parse(raw_gt, mode='eval').body 
                unparsed_form = astunparse.unparse(expr_node).strip()
                if unparsed_form:
                    self.target_strings_normalized.append(unparsed_form)
                    self.target_node_types.add(type(expr_node)) # Capture type
                if self.target_strings_normalized:
                    self.logger.debug(f"NodeFinder.__init__ (Single Expr Parse): Normalized GT: {self.target_strings_normalized}")
                    parsed_successfully = True
            except SyntaxError as e_expr:
                self.logger.warning(f"NodeFinder.__init__: Single expression parse failed (Error: {e_expr}). Cleaned GT: '{raw_gt[:150]}...'. Trying individual lines.")
            except Exception as e_generic_expr:
                 self.logger.error(f"NodeFinder.__init__: Generic error during single expr parse/unparse (Error: {e_generic_expr}). Cleaned GT: '{raw_gt[:150]}...'")
        
        # 4. Fallback: Try to parse line-by-line (existing logic)
        if not parsed_successfully:
            lines = raw_gt.splitlines()
            parsed_lines_as_targets = []
            parsed_line_types = set()
            for line_content in lines:
                stripped_line = line_content.strip()
                if not stripped_line: continue
                dedented_line = textwrap.dedent(stripped_line)
                
                line_parsed_this_iteration = False
                try: # Try as statement
                    line_ast_module = gast.parse(dedented_line) 
                    if line_ast_module.body and isinstance(line_ast_module.body[0], (gast.stmt, gast.expr)):
                        line_node_to_unparse = line_ast_module.body[0]
                        temp_module = gast.Module(body=[line_node_to_unparse], type_ignores=[])
                        unparsed_form = astunparse.unparse(temp_module).strip()
                        if unparsed_form:
                            parsed_lines_as_targets.append(unparsed_form)
                            parsed_line_types.add(type(line_node_to_unparse))
                            line_parsed_this_iteration = True
                except SyntaxError: # Try as expression if statement fails
                    try:
                        expr_node = gast.parse(dedented_line, mode='eval').body
                        unparsed_form = astunparse.unparse(expr_node).strip()
                        if unparsed_form:
                           parsed_lines_as_targets.append(unparsed_form)
                           parsed_line_types.add(type(expr_node))
                           line_parsed_this_iteration = True
                    except SyntaxError:
                        self.logger.warning(f"NodeFinder.__init__: Line could not be parsed as stmt or expr: '{stripped_line}'")
                    except Exception as e_generic_line_expr:
                        self.logger.error(f"NodeFinder.__init__: Generic error during line-expr parse/unparse: {e_generic_line_expr} for line '{stripped_line}'")
                except Exception as e_generic_line_stmt:
                    self.logger.error(f"NodeFinder.__init__: Generic error during line-stmt parse/unparse: {e_generic_line_stmt} for line '{stripped_line}'")
            
            if parsed_lines_as_targets:
                self.target_strings_normalized = parsed_lines_as_targets
                self.target_node_types = parsed_line_types
                self.logger.debug(f"NodeFinder.__init__ (Line-by-Line Parse): Normalized GT forms: {self.target_strings_normalized}")
            else:
                self.logger.error(f"NodeFinder.__init__: All parsing attempts failed for GT. Using cleaned raw string as fallback target: '{raw_gt[:150]}...'")
                if raw_gt: # Ensure it's not empty
                    self.target_strings_normalized.append(raw_gt)
                    # Cannot determine type for raw string, so target_node_types might be empty
                    # which means the type-check optimization in visit() won't prune much.


    def visit(self, node):
        if not self.target_strings_normalized: # No targets to match
            super().generic_visit(node)
            return

        # Optimization: Only attempt to unparse and match if node type is relevant
        if self.target_node_types and not isinstance(node, tuple(self.target_node_types)):
            super().generic_visit(node) # Still need to visit children
            return

        current_unparsed_form = ""
        match_found = False
        try:
            if isinstance(node, gast.stmt):
                temp_module = gast.Module(body=[node], type_ignores=[])
                current_unparsed_form = astunparse.unparse(temp_module).strip()
            elif isinstance(node, gast.expr):
                current_unparsed_form = astunparse.unparse(node).strip()
            
            if current_unparsed_form:
                # OPTIMIZATION: Check if any target string could be a substring of current_unparsed_form
                # This is a quick pre-filter. A more precise match would be `in self.target_strings_normalized`
                # For simplicity and correctness, stick to exact match for now.
                if current_unparsed_form in self.target_strings_normalized:
                    match_found = True
        
        except Exception as e_unparse:
            # self.logger.debug(f"NodeFinder.visit: Error unparsing node {type(node).__name__} for matching: {e_unparse}")
            pass
        
        if match_found:
            # self.logger.debug(f"NodeFinder.visit: Matched node {type(node).__name__} (ID: {id(node)}) which unparses to '{current_unparsed_form[:100]}...'")
            for sub_node in gast.walk(node):
                self.found_node_ids.add(id(sub_node))
            return # Don't visit children of a fully matched node further
        
        super().generic_visit(node)

# def get_ast_nodes_for_string(root_ast_node: gast.AST, target_string: str, logger_instance=None) -> Set[int]:
#     current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
#     finder = NodeFinder(target_string, logger_instance=current_logger)
#     finder.visit(root_ast_node)
#     return finder.found_node_ids

# =============================================================================
# Main Perturbation Function
# =============================================================================

def perturbation(code: str, 
                 original_eval_prompt_template: Optional[str] = None, 
                 ground_truth_code_str: Optional[str] = None, 
                 apply_perturbation: bool = True, 
                 apply_renaming: bool = False, 
                 show_reorderings: bool = False, 
                 logger_instance=None,
                 _precomputed_normalized_gt_strings: Optional[List[str]] = None,
                 _precomputed_gt_node_types: Optional[Set[type]] = None):
    current_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    
    if not apply_perturbation and not apply_renaming:
        # If ground_truth_code_str is provided, we need to return code with {{completion}}
        if ground_truth_code_str and original_eval_prompt_template:
            return original_eval_prompt_template, 0 # No reorderings
        return code, 0 # No reorderings

    total_reorderings = 0
    reconstructed_ast: Optional[gast.AST] = None # Explicitly gast.AST or None

    if apply_perturbation:
        current_logger.debug("Starting perturbation: Building graph and AST from full code.")
        
        try:
            initial_ast_tree = gast.parse(code)
        except SyntaxError as e:
            current_logger.error(f"Perturbation: Initial gast.parse failed for code. Error: {e}")
            current_logger.debug(f"Problematic code for gast.parse:\n{code[:500]}")
            return code, 0 # Return original code on parsing failure
            
        graph, tree = dfp.get_program_graph(initial_ast_tree) # tree is initial_ast_tree

        if graph is None or tree is None: # Should not happen if get_program_graph is robust
            current_logger.error("Perturbation: Failed to build program graph or AST. Aborting.")
            return code, 0 

        ground_truth_ast_node_ids: Set[int] = set()
        if ground_truth_code_str and tree:
            if _precomputed_normalized_gt_strings is not None and _precomputed_gt_node_types is not None:
                # Use NodeFinder initialized with precomputed values
                node_matcher = NodeFinder(
                    target_code_string="", # Not used when precomputed are provided
                    logger_instance=current_logger,
                    _precomputed_normalized_strings=_precomputed_normalized_gt_strings,
                    _precomputed_target_types=_precomputed_gt_node_types
                )
                node_matcher.visit(tree)
                ground_truth_ast_node_ids = node_matcher.found_node_ids
                current_logger.debug(f"Identified {len(ground_truth_ast_node_ids)} AST node IDs for ground truth using precomputed forms.")
            else:
                # Fallback: This should ideally not be hit if the calling code is updated
                current_logger.warning("Perturbation called without precomputed GT forms; "
                                       "falling back to on-the-fly NodeFinder initialization.")
                legacy_finder = NodeFinder(ground_truth_code_str, logger_instance=current_logger)
                legacy_finder.visit(tree)
                ground_truth_ast_node_ids = legacy_finder.found_node_ids




            # Important: `tree` here is the AST of the `full_code`.
            # ground_truth_ast_node_ids = get_ast_nodes_for_string(tree, ground_truth_code_str, logger_instance=current_logger)

        # Apply graph transformations (imported from dfp)
        dfp.add_sequential_dependencies_for_jumps(graph, tree)
        dfp.remove_next_syntax_edges_until_first_function_call(graph, tree)
        dfp.remove_last_reads(graph, tree)
        
        ast_order_visitor = dfp.ASTOrder(graph)
        ast_order_visitor.visit(tree)
        current_logger.debug("Perturbation: AST order visitor collected dependencies.")
        ast_order_visitor.reorder_graph()
        current_logger.debug("Perturbation: Graph reordered based on AST order visitor.")

        dfp.remove_cfg_next_edges_between_functions(graph)

        current_logger.debug("Perturbation: Adding import and control block dependencies.")

        dfp.add_import_dependencies(graph, tree)
        current_logger.debug("Perturbation: Import dependencies added.")
        dfp.add_control_block_dependencies(graph)
        current_logger.debug("Perturbation: Control block dependencies added.")

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(graph.nodes.keys()) # graph.nodes should be a dict: id -> ProgramGraphNode
        for edge_obj in graph.edges: # graph.edges should be a list of Edge objects
            if hasattr(edge_obj, 'id1') and hasattr(edge_obj, 'id2'):
                 nx_graph.add_edge(edge_obj.id1, edge_obj.id2)
        current_logger.debug("Perturbation: Converted program graph to NetworkX DiGraph for cycle detection.")
            
        if nx.is_directed_acyclic_graph(nx_graph):
            # logger.debug("Graph is a DAG. Proceeding with non-deterministic topological sort.")
            topo_sort = non_deterministic_topological_sort(nx_graph)
            
            # Reconstruct AST from graph and new topological order
            # Pass original graph, topo_sort (IDs), and ground_truth_ast_node_ids (AST object IDs)
            current_logger.debug("Perturbation: Reconstructing AST from graph and topological sort.")
            reconstructed_ast, reordering_counts = graph_to_ast(
                graph, topo_sort, 
                show_reorderings=show_reorderings, 
                ground_truth_node_ids=ground_truth_ast_node_ids, # For protecting ops inside GT
                logger_instance=current_logger
            )
            current_logger.debug("Perturbation: AST reconstruction completed.")

            if show_reorderings and reordering_counts:
                pass # Simplified logging for now

            if not reconstructed_ast: # Ensure AST reconstruction was successful
                current_logger.error("AST reconstruction failed (graph_to_ast returned None).")
                return code, 0                
        else:
            current_logger.warning("Graph contains a cycle after modifications and cannot be topologically sorted. Returning original code structure.")
        
        if nx.is_directed_acyclic_graph(nx_graph):
            reconstructed_ast = tree # Or gast.parse(code) to be safe if tree was modified
    else: 
        try:
            reconstructed_ast = gast.parse(code)
        except SyntaxError as e:
            current_logger.error(f"Perturbation (renaming only path): gast.parse failed for code. Error: {e}")
            return code, 0

    if reconstructed_ast is None: # Should not happen if logic above is correct
        current_logger.error("Reconstructed AST is None before potential renaming. Returning original code.")
        return code, 0

    if apply_renaming:
        reconstructed_ast = variable_renaming(reconstructed_ast, rename_functions=True, logger_instance=current_logger)

    try:
        generated_code_final = astunparse.unparse(reconstructed_ast)
        
        # Sanity check: try to parse the generated code
        try:
            gast.parse(generated_code_final) 
        except SyntaxError as se_final:
            current_logger.error(f"Syntax error in the *final* generated code by perturbation module: {se_final}")
            current_logger.debug(f"Problematic final generated code snippet:\n{generated_code_final[:1000]}...")
            return code, 0 # Return original if final code is invalid
        
       
        return generated_code_final, total_reorderings

    except Exception as e_unparse:
        current_logger.error(f"Error during final unparsing of AST: {e_unparse}", exc_info=True)
        return code, 0 

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    
    sample_code_context = """
import sys
def solve():
    n = int(input())
    s = input().split('W')
    for i in s:
        # Placeholder will go here
        # {{completion}} 
        if (bs ^ rs):
            print('NO')
            return
    print('YES')

for t in range(int(input())):
    solve()
"""
    sample_ground_truth_actual_code = """bs = ('B' in i)
        rs = ('R' in i)"""

    # Construct the full code as if the LLM provided the ground_truth
    full_code_for_perturbation = sample_code_context.replace("# {{completion}}", sample_ground_truth_actual_code)
    
    logger.info("Testing perturbation with ground truth marking:")
    reordered_code_containing_placeholder_str, reorder_count = perturbation(
        code=full_code_for_perturbation, # The full code
        original_eval_prompt_template=None, # Not needed for this specific test path
        ground_truth_code_str=sample_ground_truth_actual_code, # Pass the actual GT code string
        apply_perturbation=True,
        apply_renaming=False, # Keep renaming off for now to isolate placeholder issues
        logger_instance=logger
    )

    print("\n--- Reordered code with placeholder marker ---")
    print(reordered_code_containing_placeholder_str)
    print("--- End of reordered code ---")

    # Now, how backup_safim2.py would use it:
    unparsed_placeholder_marker = astunparse.unparse(gast.Expr(value=gast.Constant(value=PERTURBATION_PLACEHOLDER_STR_RAW, kind=None))).strip()
    print(f"\nLooking for: '{unparsed_placeholder_marker}'")

    if unparsed_placeholder_marker in reordered_code_containing_placeholder_str:
        final_prompt_for_llm = reordered_code_containing_placeholder_str.replace(unparsed_placeholder_marker, "{{completion}}", 1)
        print("\n--- Final prompt for LLM (after replacing placeholder) ---")
        print(final_prompt_for_llm)
    else:
        print(f"\nERROR: Placeholder '{unparsed_placeholder_marker}' (from raw '{PERTURBATION_PLACEHOLDER_STR_RAW}') NOT FOUND in reordered code.")

