# File: perturbation_transformer.py
"""
This module implements various transformations and perturbations on Python code.
It includes functionalities for AST (Abstract Syntax Tree) reordering, variable renaming,
dependency analysis, and generating a modified code from a program graph.
"""

import ast
import gast
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
from typing import Optional

# Update system path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from digraph_transformer import dataflow_parser
from python_graphs.program_graph_dataclasses import EdgeType
from python_graphs import program_graph_dataclasses as pb  

# =============================================================================
# Graph and AST Utilities
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

def is_valid_topological_sort(graph, sort_order):
    if set(sort_order) != set(graph.nodes()):
        return False
    position = {node: idx for idx, node in enumerate(sort_order)}
    edges = graph.edges()
    for u, v in edges:
        if position[u] >= position[v]:
            return False
    return True

# =============================================================================
# AST Reconstruction Classes and Functions
# =============================================================================

class ASTReconstructor(ast.NodeTransformer):
    """
    Reconstructs an AST based on a given program graph and topological order.
    """
    def __init__(self, graph, topo_sort, show_reorderings=False, ground_truth_node_ids=None, logger=None): 
        self.graph = graph
        self.topo_sort = topo_sort
        self.ast_map = {node_id: graph.nodes[node_id].ast_node for node_id in topo_sort if node_id in graph.nodes and hasattr(graph.nodes[node_id], 'ast_node')}
        self.reconstructed_body = []
        self.show_reorderings = show_reorderings
        self.reordering_counts = {} if show_reorderings else None
        self.ground_truth_node_ids = ground_truth_node_ids if ground_truth_node_ids is not None else set()
        self.logger = logger if logger else logging.getLogger(__name__)

    def _is_likely_string_producing_node(self, ast_node):
        if isinstance(ast_node, gast.Constant) and isinstance(ast_node.value, str):
            return True
        if isinstance(ast_node, gast.Call):
            if isinstance(ast_node.func, gast.Name) and ast_node.func.id == 'str':
                return True
        if isinstance(ast_node, gast.JoinedStr): 
            return True
        return False

    def visit_Module(self, node: gast.Module):
        self.generic_visit(node)
        node.body = self.reorder_body(node.body)
        return node

    def visit_FunctionDef(self, node: gast.FunctionDef):
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
    
    def visit_BinOp(self, node: gast.BinOp):
        self.generic_visit(node)
        is_ground_truth_node = id(node) in self.ground_truth_node_ids
        op_name = node.op.__class__.__name__
        can_swap = False
        if op_name in ("Mult", "BitOr", "BitXor", "BitAnd"): 
            can_swap = True
        elif op_name == "Add":
            is_left_str = self._is_likely_string_producing_node(node.left)
            is_right_str = self._is_likely_string_producing_node(node.right)
            if not (is_left_str or is_right_str):
                can_swap = True 
        if can_swap:
            if self.show_reorderings and id(node) not in self.reordering_counts: 
                self.reordering_counts[id(node)] = 2
            if not is_ground_truth_node and random.choice([True, False]):
                node.left, node.right = node.right, node.left
        return node
    
    def visit_Compare(self, node: gast.Compare):
        self.generic_visit(node)
        is_ground_truth_node = id(node) in self.ground_truth_node_ids
        if len(node.ops) == 1 and node.ops[0].__class__.__name__ in ("Eq", "NotEq"):
            if self.show_reorderings:
                self.reordering_counts[id(node)] = 2
            if not is_ground_truth_node and random.choice([True, False]):
                node.left, node.comparators[0] = node.comparators[0], node.left
        return node

    def reorder_body(self, body):
        if not isinstance(body, list) or not body: # Ensure body is a list
            return body
        ast_to_node_id = {id(ast_node): node_id for node_id, ast_node in self.ast_map.items()}
        body_ids = []
        for stmt in body:
            if not isinstance(stmt, gast.AST): continue # Skip non-AST elements
            stmt_id = id(stmt)
            if stmt_id in ast_to_node_id:
                body_ids.append(ast_to_node_id[stmt_id])
        body_ids = list(set(body_ids))
        id_set = set(body_ids)
        
        if self.show_reorderings:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(id_set)
            for edge in self.graph.edges: # Ensure self.graph.edges contains Edge objects as expected by your code
                if hasattr(edge, 'id1') and hasattr(edge, 'id2') and edge.id1 in id_set and edge.id2 in id_set:
                    subgraph.add_edge(edge.id1, edge.id2)
            global_order = {node_id: idx for idx, node_id in enumerate(self.topo_sort)}
            external_dep = {}
            for node_id in id_set:
                ext_order = None
                for edge in self.graph.edges:
                    if hasattr(edge, 'id2') and hasattr(edge, 'id1') and edge.id2 == node_id and edge.id1 not in id_set:
                        order_val = global_order.get(edge.id1, -1)
                        if ext_order is None or order_val > ext_order:
                            ext_order = order_val
                if ext_order is not None:
                    external_dep[node_id] = ext_order
            for u in id_set:
                for v in id_set:
                    if u == v: continue
                    if u in external_dep and v in external_dep:
                        if external_dep[u] < external_dep[v] and not nx.has_path(subgraph, v, u):
                            subgraph.add_edge(u, v)
            if len(subgraph.nodes) <= 1:
                num_reorderings = 1
            else:
                num_reorderings = count_topological_sorts_with_timeout(subgraph, logger=self.logger) # Pass logger
            self.reordering_counts[id(body)] = num_reorderings

        reordered_body = [self.ast_map[node_id] for node_id in self.topo_sort if node_id in id_set]
        return reordered_body
    

def count_topological_sorts(subgraph: nx.DiGraph) -> int:
    nodes = list(subgraph.nodes())
    n = len(nodes)
    index_map = {node: i for i, node in enumerate(nodes)}
    pred_list = [[] for _ in range(n)]
    for u, v in subgraph.edges():
        pred_list[index_map[v]].append(index_map[u])
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                if all(mask & (1 << pred) for pred in pred_list[i]):
                    dp[mask | (1 << i)] += dp[mask]
    return dp[(1 << n) - 1]

def count_topological_sorts_worker(subgraph, queue):
    try:
        result = count_topological_sorts(subgraph)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def count_topological_sorts_with_timeout(subgraph: nx.DiGraph, timeout: int = 120, logger=None) -> Optional[int]: # Added logger
    current_logger = logger if logger else logging.getLogger(__name__)
    queue = mp.Queue()
    p = mp.Process(target=count_topological_sorts_worker, args=(subgraph, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        current_logger.warning("Counting topological sorts timed out.") # Use logger
        return None
    else:
        result = queue.get()
        if isinstance(result, Exception):
            current_logger.error(f"Error during counting topological sorts: {result}") # Use logger
            return None
        return result

def graph_to_ast(graph, topo_sort, show_reorderings=False, ground_truth_node_ids=None, logger=None): # Added logger
    if graph.root_id in graph.nodes:
         module_ast_node_in_graph = graph.nodes[graph.root_id].ast_node
    elif topo_sort and topo_sort[0] in graph.nodes:
         module_ast_node_in_graph = graph.nodes[topo_sort[0]].ast_node
    else:
        # Use logger if available
        current_logger = logger if logger else logging.getLogger(__name__)
        current_logger.error("Cannot determine the root AST node for reconstruction.")
        raise ValueError("Cannot determine the root AST node for reconstruction.")

    reconstructor = ASTReconstructor(graph, topo_sort, show_reorderings=show_reorderings, ground_truth_node_ids=ground_truth_node_ids, logger=logger) # Pass logger
    reconstructor.visit(module_ast_node_in_graph)
    reordering_counts = reconstructor.reordering_counts if show_reorderings else None
    return module_ast_node_in_graph, reordering_counts 


# =============================================================================
# AST Graph Reordering and Edge Removal
# =============================================================================

class ASTOrder(gast.NodeVisitor): # Use gast.NodeVisitor
    def __init__(self, graph):
        self.index = 0
        self.node_to_order = {}
        self.graph = graph

    def visit(self, node):
        self.node_to_order[id(node)] = self.index
        self.index += 1
        self.generic_visit(node)

    def reorder_graph(self):
        edges_to_remove = []
        for edge in self.graph.edges:
            # Ensure nodes exist in graph.nodes and have ast_node attribute
            if not (hasattr(edge, 'id1') and hasattr(edge, 'id2') and \
                    edge.id1 in self.graph.nodes and edge.id2 in self.graph.nodes and \
                    hasattr(self.graph.nodes[edge.id1], 'ast_node') and \
                    hasattr(self.graph.nodes[edge.id2], 'ast_node')):
                continue # Skip malformed edges or nodes

            # Ensure ast_node IDs are in node_to_order
            node1_ast_id = id(self.graph.nodes[edge.id1].ast_node)
            node2_ast_id = id(self.graph.nodes[edge.id2].ast_node)
            if node1_ast_id not in self.node_to_order or node2_ast_id not in self.node_to_order:
                continue


            if self.node_to_order[node1_ast_id] > self.node_to_order[node2_ast_id]:
                if self.creates_cycle(edge):
                    edges_to_remove.append(edge)
            elif edge.id1 == edge.id2:
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            if edge in self.graph.edges: # Check existence before removal
                 self.graph.edges.remove(edge)

    def creates_cycle(self, edge_to_test): # Renamed edge to edge_to_test
        visited = set()
        # Start search from the target of the edge to see if we can reach its source
        stack = [edge_to_test.id2] # Use edge_to_test
        while stack:
            node_id = stack.pop()
            if node_id == edge_to_test.id1: # Use edge_to_test
                return True # Cycle detected
            if node_id not in visited:
                visited.add(node_id)
                # Find successors of node_id in the graph *excluding* the edge_to_test if it were reversed
                for e in self.graph.edges:
                    if e.id1 == node_id:
                        # If we consider adding edge_to_test (reversed: id2->id1),
                        # we search paths from id2. If we find id1, it's a cycle.
                        stack.append(e.id2)
        return False

class ParentTrackingVisitor(gast.NodeVisitor): # Use gast.NodeVisitor
    def __init__(self):
        self.parent_map = {}

    def visit(self, node):
        for child_field_name, child_value in gast.iter_fields(node):
            if isinstance(child_value, list):
                for item in child_value:
                    if isinstance(item, gast.AST):
                        self.parent_map[item] = node
                        self.visit(item)
            elif isinstance(child_value, gast.AST):
                self.parent_map[child_value] = node
                self.visit(child_value)
    
    def get_parents(self, node):
        parents = []
        current = self.parent_map.get(node, None)
        while current:
            parents.append(current)
            current = self.parent_map.get(current, None)
        return parents
    
    
def would_create_cycle(source_id, target_id, graph):
        if source_id == target_id: return True
        visited = set()
        to_visit = deque([target_id]) # Use deque for efficient pop
        while to_visit:
            current = to_visit.popleft() # Use popleft for BFS-like traversal
            if current == source_id: return True
            if current not in visited:
                visited.add(current)
                for edge in graph.edges:
                    if edge.id1 == current:
                        to_visit.append(edge.id2)
        return False

def remove_last_reads(graph, ast_tree):
    visitor = ParentTrackingVisitor()
    visitor.visit(ast_tree)
    edges_to_remove = []
    for edge in graph.edges:
        if not (hasattr(edge, 'type') and hasattr(edge, 'id1') and hasattr(edge, 'id2')): continue
        if edge.type == EdgeType.LAST_READ: 
            node1_obj = graph.get_node(edge.id1)
            node2_obj = graph.get_node(edge.id2)
            if not node1_obj or not node2_obj or not hasattr(node1_obj, 'ast_node') or not hasattr(node2_obj, 'ast_node'): continue

            node1, node2 = node1_obj.ast_node, node2_obj.ast_node
            parents1 = visitor.get_parents(node1)
            parents2 = visitor.get_parents(node2)
            parents = parents1 + parents2
            parent_classes = [p.__class__.__name__ for p in parents]
            if "Call" in parent_classes:
                white_list_builtin = ["sum", "mean", "max", "min", "len", "sorted", "reversed", "enumerate", "range", "zip", "map", "filter", "all", "any"]
                white_list_numpy = ["array", "arange", "linspace", "zeros", "ones", "empty", "full", "eye", "identity", "random", "dot", "matmul", "linalg", "fft", "mean", "median", "std", "var", "sum", "prod", "cumsum", "cumprod", "min", "max", "argmin", "argmax", "argsort", "sort", "unique", "reshape", "transpose", "concatenate", "stack", "hstack", "vstack", "split", "hsplit", "vsplit"]
                if any([isinstance(parent, gast.Call) and isinstance(parent.func, gast.Attribute) and (parent.func.attr in white_list_builtin or parent.func.attr in white_list_numpy) for parent in parents]):
                    edges_to_remove.append(edge) 
            else:
                edges_to_remove.append(edge)
    for edge in edges_to_remove:
        if edge in graph.edges: graph.edges.remove(edge)

def add_control_block_dependencies(graph):
    def get_descendants(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return set()
        
        visited.add(node_id)
        descendants = set()
        
        for edge in graph.edges:
            if edge.id1 == node_id:
                descendants.add(edge.id2)
                descendants.update(get_descendants(edge.id2, visited))
        
        return descendants
    
    for node_id in graph.nodes:
        ast_node = graph.get_node(node_id).ast_node
        node_type = ast_node.__class__.__name__

        if node_type in ["If", "For", "Try", "While", "With"]:
            # Get all nodes in the block
            block_nodes = set()
            if hasattr(ast_node, 'body'):
                block_nodes.update(graph.get_node_by_ast_node(stmt).id for stmt in ast_node.body)
            if hasattr(ast_node, 'orelse'):
                block_nodes.update(graph.get_node_by_ast_node(stmt).id for stmt in ast_node.orelse)

            if node_type == "Try":
                for handler in ast_node.handlers:
                    block_nodes.update(graph.get_node_by_ast_node(stmt).id for stmt in handler.body)
            if node_type == "With" and hasattr(ast_node, 'items'):
                block_nodes.update(graph.get_node_by_ast_node(item.context_expr).id for item in ast_node.items)

            if node_type in ["If", "While"]:
                block_nodes.add(graph.get_node_by_ast_node(ast_node.test).id)
            elif node_type == "For":
                block_nodes.add(graph.get_node_by_ast_node(ast_node.target).id)
                block_nodes.add(graph.get_node_by_ast_node(ast_node.iter).id)

            all_descendants = set()
            for block_node in block_nodes:
                all_descendants.update(get_descendants(block_node))

            block_nodes.update(all_descendants)

            for inner_node_id in block_nodes:
                for edge in graph.edges:
                    if edge.id2 == inner_node_id and edge.id1 not in block_nodes and edge.id1 != node_id:
                        new_edge = pb.Edge(id1=edge.id1, id2=node_id, type=edge.type)
                        if new_edge not in graph.edges and not would_create_cycle(edge.id1, node_id, graph=graph):
                            graph.add_edge(new_edge)
                            

def remove_cfg_next_edges_between_functions(graph):
    """
    Removes CFG_NEXT edges between functions if they are within the same module or class.

    Args:
        graph: The program graph.
    """
    edges_to_remove = []
    for edge in graph.edges:
        if edge.type == EdgeType.CFG_NEXT:
            node1 = graph.get_node(edge.id1).ast_node
            node2 = graph.get_node(edge.id2).ast_node
            if node1.__class__.__name__ == "FunctionDef" and node2.__class__.__name__ == "FunctionDef":
                visitor = ParentTrackingVisitor()
                root_node = graph.nodes[graph.root_id].ast_node
                visitor.visit(root_node)
                parent1 = visitor.get_parents(node1)
                parent2 = visitor.get_parents(node2)
                parent1_classes = [p.__class__.__name__ for p in parent1]
                parent2_classes = [p.__class__.__name__ for p in parent2]
                if any(cls in ["Module", "ClassDef"] for cls in parent1_classes) and any(cls in ["Module", "ClassDef"] for cls in parent2_classes):
                    edges_to_remove.append(edge)

    for edge in edges_to_remove:
        graph.edges.remove(edge)

def remove_next_syntax_edges_until_first_function_call(graph, ast_tree):
    """
    Removes next syntax edges until the first non-built-in function call is encountered.

    Args:
        graph: The program graph.
        ast_tree: The AST of the program.
    """
    edges_to_keep = []
    found_first_function_call = False

    for edge in graph.edges:
        if edge.type.value == 9:  # Edge type for next syntax
            if not found_first_function_call:
                node = graph.get_node(edge.id1).ast_node
                if isinstance(node, gast.gast.Call) or any(isinstance(child, gast.gast.Call) for child in ast.iter_child_nodes(node)):
                    visitor = ParentTrackingVisitor()
                    visitor.visit(ast_tree)
                    parents = visitor.get_parents(node)
                    parent_classes = [p.__class__.__name__ for p in parents]
                    
                    if "FunctionDef" not in parent_classes:
                        if isinstance(node, gast.gast.Call):
                            func_name = node.func.id if isinstance(node.func, gast.gast.Name) else None
                        else:
                            func_name = None
                            for child in ast.iter_child_nodes(node):
                                if isinstance(child, gast.gast.Call):
                                    func_name = child.func.id if isinstance(child.func, gast.gast.Name) else None
                                    break
                        if func_name and func_name not in dir(__builtins__):
                            found_first_function_call = True
                            edges_to_keep.append(edge)
            else:
                edges_to_keep.append(edge)
        else:
            edges_to_keep.append(edge)

    graph.edges = edges_to_keep

class ImportDependencyVisitor(gast.NodeVisitor):
    """
    After the ProgramGraph is built, walk the AST again.
    Track each import as if it were a variable assignment.
    Then add edges from the import statement to every usage of that name.
    """
    def __init__(self, graph):
        self.graph = graph
        # Map a variable name to the node ID where it's "imported"
        self.import_writes = {}

    def visit_Import(self, node: gast.Import):
        node_obj = self.graph.get_node_by_ast_node(node)
        if not node_obj: return
        node_id = node_obj.id
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            self.import_writes[local_name] = node_id
        self.generic_visit(node)

    def visit_ImportFrom(self, node: gast.ImportFrom):
        node_obj = self.graph.get_node_by_ast_node(node)
        if not node_obj: return
        node_id = node_obj.id
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            self.import_writes[local_name] = node_id
        self.generic_visit(node)

    def visit_Name(self, node):
        """
        Whenever we see a usage of a name in `Load` context, check if it was imported.
        If so, add a LAST_READ edge from the import node to this usage node.
        """
        if isinstance(node.ctx, gast.Load):
            name = node.id
            if name in self.import_writes:
                import_node_id = self.import_writes[name]
                usage_node_id = self.graph.get_node_by_ast_node(node).id
                # Add the dataflow edge if it doesn't already exist
                new_edge = pb.Edge(
                    id1=import_node_id,
                    id2=usage_node_id,
                    type=EdgeType.LAST_READ
                )
                if new_edge not in self.graph.edges:

                    self.graph.add_edge(new_edge)
        self.generic_visit(node)

def add_import_dependencies(graph, ast_tree):
    """
    Top-level helper that runs the ImportDependencyVisitor.
    """
    visitor = ImportDependencyVisitor(graph)
    visitor.visit(ast_tree)

    # For each imported name, find usages in assignment
    for var_name, import_node_id in visitor.import_writes.items():
        for node_id, node in graph.nodes.items():
            ast_node = node.ast_node
            
            # Check if this node uses the imported name
            class UsageVisitor(gast.NodeVisitor):
                def __init__(self):
                    self.uses_import = False
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, gast.Load) and node.id == var_name:
                        self.uses_import = True
                    self.generic_visit(node)
            
            visitor = UsageVisitor()
            visitor.visit(ast_node)
            
            if visitor.uses_import:
                # Add a COMPUTED_FROM edge 
                edge = pb.Edge(
                    id1=import_node_id,
                    id2=node_id,
                    type=EdgeType.COMPUTED_FROM  # Use stronger dependency
                )
                
                if edge not in graph.edges and not would_create_cycle(import_node_id, node_id, graph=graph):
                    graph.add_edge(edge)

# =============================================================================
# Global Variable and Function Analysis
# =============================================================================

def get_name(node):
    """
    Helper fundef get_nction to extract the name from a node.

    Args:
        node: An AST node.

    Returns:
        str or None: The name attribute from the node.
    """
    return getattr(node, 'id', getattr(node, 'arg', None))

def analyze_function_globals(func_node: gast.FunctionDef): # Use gast.FunctionDef
    used_globals = set()
    local_vars = {get_name(arg) for arg in func_node.args.args if get_name(arg) is not None}
    class LocalCollector(gast.NodeVisitor):
        def visit_Assign(self, node: gast.Assign):
            for target in node.targets:
                n = get_name(target)
                if n: local_vars.add(n)
            self.generic_visit(node)
        def visit_AugAssign(self, node: gast.AugAssign):
            n = get_name(node.target)
            if n: local_vars.add(n)
            self.generic_visit(node)
    LocalCollector().visit(func_node)
    class GlobalUsageCollector(gast.NodeVisitor):
        def visit_Name(self, node: gast.Name):
            if isinstance(node.ctx, gast.Load):
                n = get_name(node)
                if n and n not in local_vars: used_globals.add(n)
            self.generic_visit(node)
    GlobalUsageCollector().visit(func_node)
    return used_globals

def get_module_function_usage(module_node: gast.Module): # Use gast.Module
    usage_map = {}
    for stmt in module_node.body:
        if isinstance(stmt, gast.FunctionDef):
            usage_map[stmt.name] = analyze_function_globals(stmt)
    return usage_map

# =============================================================================
# Module Statement Reordering Based on Global Usage
# =============================================================================

class ModCollector(gast.NodeVisitor):
    def __init__(self, logger=None):
        self.mods = set()
        self.logger = logger if logger else logging.getLogger(__name__)

    def _add_name_if_valid(self, name_str):
        if name_str and isinstance(name_str, str) and name_str not in dir(builtins):
            self.mods.add(name_str)

    def _process_target(self, target_node):
        if isinstance(target_node, gast.Name):
            self.logger.debug(f"ModCollector: Adding Name target: {get_name(target_node)}")
            self._add_name_if_valid(get_name(target_node))
        elif isinstance(target_node, (gast.Tuple, gast.List)):
            self.logger.debug(f"ModCollector: Processing Tuple/List target elements")
            for elt in target_node.elts: self._process_target(elt)
        elif isinstance(target_node, gast.Subscript):
            self.logger.debug(f"ModCollector: Processing Subscript target")
            current = target_node.value
            while isinstance(current, (gast.Subscript, gast.Attribute)):
                current = current.value if isinstance(current, gast.Subscript) else current.value
            if isinstance(current, gast.Name):
                self.logger.debug(f"ModCollector: Adding Subscript base Name: {get_name(current)}")
                self._add_name_if_valid(get_name(current))
        elif isinstance(target_node, gast.Attribute):
            self.logger.debug(f"ModCollector: Processing Attribute target")
            current = target_node.value
            while isinstance(current, gast.Attribute): current = current.value
            if isinstance(current, gast.Name):
                self.logger.debug(f"ModCollector: Adding Attribute base Name: {get_name(current)}")
                self._add_name_if_valid(get_name(current))

    def visit_Assign(self, node: gast.Assign):
        self.logger.debug(f"ModCollector: Visiting Assign: {astunparse.unparse(node).strip()}")
        for target in node.targets: self._process_target(target)
        self.visit(node.value)

    def visit_AugAssign(self, node: gast.AugAssign):
        self.logger.debug(f"ModCollector: Visiting AugAssign: {astunparse.unparse(node).strip()}")
        self._process_target(node.target)
        self.visit(node.value)

    def visit_FunctionDef(self, node: gast.FunctionDef):
        self.logger.debug(f"ModCollector: Adding FunctionDef: {node.name}")
        self._add_name_if_valid(node.name)
        for decorator in node.decorator_list: self.visit(decorator)

    def visit_AsyncFunctionDef(self, node: gast.AsyncFunctionDef):
        self.logger.debug(f"ModCollector: Adding AsyncFunctionDef: {node.name}")
        self._add_name_if_valid(node.name)
        for decorator in node.decorator_list: self.visit(decorator)

    def visit_ClassDef(self, node: gast.ClassDef):
        self.logger.debug(f"ModCollector: Adding ClassDef: {node.name}")
        self._add_name_if_valid(node.name)
        for decorator in node.decorator_list: self.visit(decorator)
        for base in node.bases: self.visit(base)
        for keyword in node.keywords: self.visit(keyword.value)

    def visit_Import(self, node: gast.Import):
        self.logger.debug(f"ModCollector: Visiting Import")
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            self.logger.debug(f"  Adding imported name: {local_name}")
            self._add_name_if_valid(local_name)

    def visit_ImportFrom(self, node: gast.ImportFrom):
        self.logger.debug(f"ModCollector: Visiting ImportFrom")
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            self.logger.debug(f"  Adding imported name from 'from': {local_name}")
            self._add_name_if_valid(local_name)

    def visit_Call(self, node: gast.Call):
        self.logger.debug(f"ModCollector: Visiting Call: {astunparse.unparse(node).strip()}")
        if isinstance(node.func, gast.Attribute):
            val_node = node.func.value
            if isinstance(val_node, gast.Name):
                self.logger.debug(f"  Call on Name attribute, considering '{get_name(val_node)}' modified.")
                self._add_name_if_valid(get_name(val_node))
            else: self.visit(val_node)
        else: self.visit(node.func)
        for arg in node.args: self.visit(arg)
        for kw in node.keywords: self.visit(kw.value)


class UseCollector(gast.NodeVisitor):
    def __init__(self, usage_map, logger=None):
        self.uses = set()
        self.usage_map = usage_map
        self.logger = logger if logger else logging.getLogger(__name__)
    def _add_name_if_valid(self, name_str):
        if name_str and isinstance(name_str, str) and name_str not in dir(builtins):
            self.uses.add(name_str)

    def visit_Name(self, node: gast.Name):
        self.logger.debug(f"UseCollector: Visiting Name: {get_name(node)}, ctx: {type(node.ctx)}")
        if isinstance(node.ctx, gast.Load): self._add_name_if_valid(get_name(node))

    def visit_Call(self, node: gast.Call):
        self.logger.debug(f"UseCollector: Visiting Call: {astunparse.unparse(node).strip()}")
        if isinstance(node.func, gast.Name):
            func_name = get_name(node.func)
            if func_name and func_name in self.usage_map:
                self.logger.debug(f"  Call to known func '{func_name}', adding uses: {self.usage_map[func_name]}")
                self.uses.update(self.usage_map[func_name])
        super().generic_visit(node)

    def visit_FunctionDef(self, node: gast.FunctionDef):
        self.logger.debug(f"UseCollector: Visiting FunctionDef: {node.name}")
        for decorator in node.decorator_list: self.visit(decorator)
        if node.args:
            for default_expr in node.args.defaults: self.visit(default_expr)
            for arg_node in node.args.args: 
                if arg_node.annotation: self.visit(arg_node.annotation)
            if node.args.vararg and node.args.vararg.annotation: self.visit(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation: self.visit(node.args.kwarg.annotation)
            for kwonly_arg in node.args.kwonlyargs:
                if kwonly_arg.annotation: self.visit(kwonly_arg.annotation)
            for kw_default_expr in node.args.kw_defaults:
                 if kw_default_expr: self.visit(kw_default_expr)
        if node.returns: self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: gast.AsyncFunctionDef):
        self.logger.debug(f"UseCollector: Visiting AsyncFunctionDef: {node.name}")
        # Similar to FunctionDef
        for decorator in node.decorator_list: self.visit(decorator)
        if node.args: # Same logic as FunctionDef for args
            for default_expr in node.args.defaults: self.visit(default_expr)
            # ... (rest of arg processing) ...
        if node.returns: self.visit(node.returns)

    def visit_ClassDef(self, node: gast.ClassDef):
        self.logger.debug(f"UseCollector: Visiting ClassDef: {node.name}")
        for decorator in node.decorator_list: self.visit(decorator)
        for base in node.bases: self.visit(base)
        for keyword in node.keywords: self.visit(keyword.value)

    def generic_visit(self, node):
        super().generic_visit(node)

def reorder_module_statements(module_node: gast.Module, usage_map, logger=None): 
    current_logger = logger if logger else logging.getLogger(__name__)
    stmts = module_node.body
    n = len(stmts)
    if n == 0: return module_node 

    mod_sets = []
    use_sets = []
    for idx, stmt in enumerate(stmts):
        current_logger.debug(f"Reordering Module Stmt {idx}: {astunparse.unparse(stmt).strip()[:100]}") # Log stmt
        if not isinstance(stmt, gast.AST): continue # 
        if isinstance(stmt, gast.FunctionDef):
            mod_collector = ModCollector(logger=current_logger)
            mod_collector.visit_FunctionDef(stmt) # Visit only the FunctionDef node, not its body for module-level mods
            mods = mod_collector.mods

            use_collector = UseCollector(usage_map, logger=current_logger)
            use_collector.visit_FunctionDef(stmt) # To collect uses from decorators, defaults, annotations
            uses = use_collector.uses
        else:
            mod_collector = ModCollector(logger=current_logger)
            mod_collector.visit(stmt)
            mods = mod_collector.mods

            use_collector = UseCollector(usage_map, logger=current_logger)
            use_collector.visit(stmt)
            uses = use_collector.uses
        
        current_logger.debug(f"  Stmt {idx} Mods: {mods}") # Log mods
        current_logger.debug(f"  Stmt {idx} Uses: {uses}") # Log uses
        mod_sets.append(mods)
        use_sets.append(uses)

    dep_graph = nx.DiGraph()
    dep_graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and (mod_sets[i] & use_sets[j]):
                dep_graph.add_edge(i, j)
                current_logger.debug(f"Adding module_stmt_reorder edge: Stmt {i} -> Stmt {j} due to {mod_sets[i].intersection(use_sets[j])}")
    try:
        new_order_indices = list(nx.topological_sort(dep_graph))
        # Ensure all original statements are included if graph was disconnected
        if len(new_order_indices) != n:
            # Add missing statements (those with no dependencies) preserving relative order among them
            all_indices = set(range(n))
            missing_indices = sorted(list(all_indices - set(new_order_indices)))
            # This simple append might not be ideal, ideally merge preserving original relative order of independent groups
            new_order_indices.extend(missing_indices)


    except nx.NetworkXUnfeasible: # Cycle detected
        current_logger.warning("Cycle detected in module statement dependencies. Falling back to original order for module statements.")
        new_order_indices = list(range(n)) # Fallback to original order of statements

    reordered_stmts = []
    original_indices_in_new_order = set()

    for new_idx_pos in new_order_indices:
        if 0 <= new_idx_pos < n and isinstance(stmts[new_idx_pos], gast.AST):
            reordered_stmts.append(stmts[new_idx_pos])
            original_indices_in_new_order.add(new_idx_pos)
        else:
            current_logger.warning(f"Index {new_idx_pos} from topological sort is out of bounds or not an AST node in original statements. Skipping.")

    # Add any statements that were part of original `stmts` but not in `new_order_indices` (e.g., if graph was not fully connected)
    # while preserving their relative order from original `stmts`.
    for original_idx in range(n):
        if original_idx not in original_indices_in_new_order and isinstance(stmts[original_idx], gast.AST):
            reordered_stmts.append(stmts[original_idx])
            current_logger.debug(f"Appending statement at original index {original_idx} as it was not covered by topological sort.")


    module_node.body = reordered_stmts
    return module_node

# =============================================================================
# Variable Renaming
# =============================================================================

def generate_new_name(length=8):
    first_char = random.choice(string.ascii_uppercase) # Python identifiers can't start with digits
    rest = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length-1))
    return first_char + rest

def collect_global_vars(ast_tree: gast.Module): # Use gast.Module
    global_vars = set()
    for node in ast_tree.body:
        if isinstance(node, gast.Assign):
            for target in node.targets:
                name = get_name(target) # get_name should handle gast.Name
                if name is not None: global_vars.add(name)
        elif isinstance(node, gast.AugAssign):
            name = get_name(node.target)
            if name is not None: global_vars.add(name)
    return global_vars

def variable_renaming(ast_tree: gast.AST, rename_functions=True): # Use gast.AST
    defined_names = set()
    function_renames = {}
    class DefCollector(gast.NodeVisitor):
        def visit_FunctionDef(self, node: gast.FunctionDef):
            defined_names.add(node.name)
            self.generic_visit(node)
        def visit_ClassDef(self, node: gast.ClassDef):
            defined_names.add(node.name)
            self.generic_visit(node)
    DefCollector().visit(ast_tree)
    
    # Ensure ast_tree is gast.Module for collect_global_vars if it expects .body
    global_vars = set()
    if isinstance(ast_tree, gast.Module):
        global_vars = collect_global_vars(ast_tree)

    builtin_names = set(dir(builtins)) # Renamed to avoid conflict
    global_mapping = {}
    for var in global_vars:
        if var not in defined_names and var not in builtin_names:
            global_mapping[var] = generate_new_name(length=random.randint(3, 10))

    class VNTransformer(gast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.scope_stack = [global_mapping.copy()]
        def _enter_scope(self): self.scope_stack.append({})
        def _exit_scope(self): self.scope_stack.pop()
        def _get_new_name(self, old_name, force_rename=False):
            for scope in reversed(self.scope_stack):
                if old_name in scope: return scope[old_name]
            if not force_rename and (old_name in defined_names or old_name in builtin_names):
                return old_name
            new_name = generate_new_name(length=random.randint(3,10))
            if self.scope_stack: self.scope_stack[-1][old_name] = new_name
            return new_name

        def visit_Module(self, node: gast.Module):
            self._enter_scope()
            self.generic_visit(node)
            self._exit_scope()
            return node
        def visit_FunctionDef(self, node: gast.FunctionDef):
            self._enter_scope()
            if rename_functions:
                original_name = node.name
                new_name = self._get_new_name(original_name, force_rename=True)
                node.name = new_name
                function_renames[original_name] = new_name
            for arg in node.args.args:
                # gast.arg has 'id' for name, standard ast.arg has 'arg'
                if hasattr(arg, "id") and isinstance(arg.id, str): # gast.Name can be arg
                    arg.id = self._get_new_name(arg.id)
                elif hasattr(arg, "arg") and isinstance(arg.arg, str): # ast.arg
                     arg.arg = self._get_new_name(arg.arg)

            node.body = [self.visit(stmt) for stmt in node.body]
            self._exit_scope()
            return node
        def visit_ClassDef(self, node: gast.ClassDef):
            self._enter_scope()
            node.body = [self.visit(stmt) for stmt in node.body]
            self._exit_scope()
            return node
        def visit_Name(self, node: gast.Name):
            if isinstance(node.ctx, (gast.Load, gast.Store, gast.Del)):
                if node.id in function_renames:
                    node.id = function_renames[node.id]
                else:
                    node.id = self._get_new_name(node.id)
            return node
    return VNTransformer().visit(ast_tree)

def find_cycle(graph, logger=None): # Added logger
    current_logger = logger if logger else logging.getLogger(__name__)
    G = nx.DiGraph()
    for node_id in graph.nodes: G.add_node(node_id)
    for edge in graph.edges:
        if hasattr(edge, 'id1') and hasattr(edge, 'id2'): # Basic check for edge structure
             G.add_edge(edge.id1, edge.id2)
    try:
        cycle = nx.find_cycle(G, orientation='original')
        current_logger.warning(f"Cycle found in perturbation graph: {cycle}") # Use logger
        for u, v, _ in cycle:
            u_type = graph.nodes[u].ast_node.__class__.__name__ if u in graph.nodes and hasattr(graph.nodes[u], 'ast_node') else "UnknownType"
            v_type = graph.nodes[v].ast_node.__class__.__name__ if v in graph.nodes and hasattr(graph.nodes[v], 'ast_node') else "UnknownType"
            current_logger.debug(f"Cycle edge: {u}({u_type}) -> {v}({v_type})") # Use logger (debug for details)
        return cycle
    except nx.NetworkXNoCycle:
        current_logger.debug("No cycle found in perturbation graph.") # Use logger
        return None
    except Exception as e:
        current_logger.error(f"Error during cycle detection: {e}")
        return None # Or re-raise depending on desired behavior

class SequentialJumpDependencyVisitor(gast.NodeVisitor):
    def __init__(self, graph, logger=None): # Added logger
        self.graph = graph
        self.logger = logger if logger else logging.getLogger(__name__)

    def _process_statement_list(self, stmt_list):
        if not stmt_list or not isinstance(stmt_list, list): return
        for i, current_stmt_ast in enumerate(stmt_list):
            if not isinstance(current_stmt_ast, gast.AST): continue
            if isinstance(current_stmt_ast, (gast.Break, gast.Continue)):
                current_stmt_node_obj = self.graph.get_node_by_ast_node(current_stmt_ast)
                if not current_stmt_node_obj:
                    self.logger.debug(f"AST node {type(current_stmt_ast)} not found in graph during jump dependency.")
                    continue
                current_stmt_node_id = current_stmt_node_obj.id
                for j in range(i):
                    prev_stmt_ast = stmt_list[j]
                    if not isinstance(prev_stmt_ast, gast.AST): continue
                    prev_stmt_node_obj = self.graph.get_node_by_ast_node(prev_stmt_ast)
                    if not prev_stmt_node_obj:
                        self.logger.debug(f"AST node {type(prev_stmt_ast)} (preceding jump) not found in graph.")
                        continue
                    prev_stmt_node_id = prev_stmt_node_obj.id
                    
                    edge_type_val = EdgeType.CFG_NEXT
                    new_edge = pb.Edge(id1=prev_stmt_node_id, id2=current_stmt_node_id, type=edge_type_val)
                    if prev_stmt_node_id != current_stmt_node_id and new_edge not in self.graph.edges and not would_create_cycle(new_edge.id1, new_edge.id2, self.graph):
                        self.graph.add_edge(new_edge)
                        self.logger.debug(f"Added jump dependency edge: {prev_stmt_node_id} -> {current_stmt_node_id}")

    def visit_FunctionDef(self, node: gast.FunctionDef): 
        self._process_statement_list(node.body)
        self.generic_visit(node)

    def visit_If(self, node: gast.If): 
        self._process_statement_list(node.body)
        self._process_statement_list(node.orelse)
        self.generic_visit(node)

    def visit_For(self, node: gast.For): 
        self._process_statement_list(node.body)
        self._process_statement_list(node.orelse)
        self.generic_visit(node)

    def visit_While(self, node: gast.While): 
        self._process_statement_list(node.body);
        self._process_statement_list(node.orelse)
        self.generic_visit(node)

    def visit_Try(self, node: gast.Try):
        self._process_statement_list(node.body)
        for handler in node.handlers:
            if isinstance(handler, gast.ExceptionHandler): self._process_statement_list(handler.body)
        self._process_statement_list(node.orelse)
        self._process_statement_list(node.finalbody)
        self.generic_visit(node)

    def visit_With(self, node: gast.With): 
        self._process_statement_list(node.body)
        self.generic_visit(node)

    def visit_ClassDef(self, node: gast.ClassDef): 
        self.generic_visit(node)

def add_sequential_dependencies_for_jumps(graph, ast_tree, logger=None): # Added logger
    visitor = SequentialJumpDependencyVisitor(graph, logger=logger) # Pass logger
    visitor.visit(ast_tree)

class NodeFinder(gast.NodeVisitor): # Use gast.NodeVisitor
    def __init__(self, target_unparsed_string):
        self.target_string = target_unparsed_string.strip() # Strip target string once
        self.found_node_ids = set()

    def generic_visit(self, node): # Changed from visit to generic_visit for standard traversal
        current_unparsed = ""
        try:
            if isinstance(node, gast.expr):
                current_unparsed = astunparse.unparse(node).strip()
            elif isinstance(node, gast.stmt):
                temp_module = gast.Module(body=[node])
                current_unparsed = astunparse.unparse(temp_module).strip()
            
            if self.target_string == current_unparsed:
                self.found_node_ids.add(id(node))

            if isinstance(node, gast.Expr) and isinstance(node.value, gast.expr):
                expr_val_unparsed = astunparse.unparse(node.value).strip()
                if self.target_string == expr_val_unparsed:
                    self.found_node_ids.add(id(node.value))
                    self.found_node_ids.add(id(node))
        except Exception:
            pass # Unparsing can fail for partial/malformed nodes during traversal
        super().generic_visit(node) # Standard traversal

def get_ast_nodes_for_string(root_ast_node: gast.AST, target_string: str, logger=None): # Use gast.AST, added logger
    current_logger = logger if logger else logging.getLogger(__name__)
    normalized_target_string = target_string.strip()
    try:
        gt_module = gast.parse(target_string.strip())
        if gt_module.body:
            if len(gt_module.body) == 1 and isinstance(gt_module.body[0], gast.Expr):
                normalized_target_string = astunparse.unparse(gt_module.body[0].value).strip()
            else:
                normalized_target_string = astunparse.unparse(gt_module.body).strip()
    except SyntaxError:
        current_logger.debug(f"Target string for AST node finding is not valid Python standalone: '{target_string[:50]}...'")
    except Exception as e:
        current_logger.debug(f"Exception normalizing target string for AST node finding: {e}")

    # Use ComprehensiveNodeFinder as it's more robust for collecting subtrees
    class ComprehensiveNodeFinder(gast.NodeVisitor):
        def __init__(self, target_unparsed_string_normalized):
            self.target_string_normalized = target_unparsed_string_normalized
            self.found_base_node_ids = set() # Store id() of Python AST node objects

        def visit(self, node): # Override visit to control recursion
            is_match = False
            try:
                current_unparsed = ""
                if isinstance(node, gast.expr):
                    current_unparsed = astunparse.unparse(node).strip()
                elif isinstance(node, gast.stmt):
                    temp_module = gast.Module(body=[node])
                    current_unparsed = astunparse.unparse(temp_module).strip()

                if self.target_string_normalized == current_unparsed:
                    is_match = True
                
                if not is_match and isinstance(node, gast.Expr) and isinstance(node.value, gast.expr):
                    expr_val_unparsed = astunparse.unparse(node.value).strip()
                    if self.target_string_normalized == expr_val_unparsed:
                        self.collect_subtree_ids(node.value)
                        self.found_base_node_ids.add(id(node)) 
                        return 
            except Exception: pass

            if is_match:
                self.collect_subtree_ids(node)
            else:
                super().generic_visit(node)

        def collect_subtree_ids(self, root_for_subtree):
            for sub_node in gast.walk(root_for_subtree): # Use gast.walk
                self.found_base_node_ids.add(id(sub_node))
    
    finder = ComprehensiveNodeFinder(normalized_target_string)
    finder.visit(root_ast_node)
    current_logger.debug(f"AST Node Finder: Target '{normalized_target_string[:50]}...', Found {len(finder.found_base_node_ids)} node IDs.")
    return finder.found_base_node_ids

# =============================================================================
# Main Perturbation Function
# =============================================================================

def perturbation(code, apply_perturbation=True, apply_renaming=False, show_reorderings=False, ground_truth_string=None,logger=None):
    """
    Main function to perform perturbation on the provided code.
    It builds a program graph, modifies it, reconstructs the AST, reorders statements,
    and applies variable renaming based on the specified boolean flags.

    Args:
        code (str): The original Python code.
        apply_perturbation (bool): If True, apply the perturbation transformations (graph-based AST reordering).
        apply_renaming (bool): If True, apply the variable renaming transformation.

    Returns:
        str: The transformed Python code.
    """

    current_logger = logger if logger else logging.getLogger(__name__)
    if not current_logger.hasHandlers(): # Basic config if run standalone
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # If no transformation is desired, return the original code
    if not apply_perturbation and not apply_renaming:
        return code, None

    # Apply perturbation transformations (graph-based AST reordering) if enabled
    total_reorderings = 0
    reconstructed_ast = None

    if apply_perturbation:
        current_logger.debug("Starting perturbation: Building graph and AST...")
        graph, tree = dataflow_parser.get_program_graph(code)

        if graph is None or tree is None:
            current_logger.error("Failed to build program graph or AST from code. Aborting perturbation.")
            current_logger.debug(f"Code provided to dataflow_parser:\n{code[:500]}...") # Log snippet
            return code, None # Return original code if graph/tree building fails

        ground_truth_ast_node_ids = set()

        if ground_truth_string and tree:
            current_logger.debug(f"Finding AST nodes for ground truth string: '{ground_truth_string[:50]}...'")
            ground_truth_ast_node_ids = get_ast_nodes_for_string(tree, ground_truth_string, logger=current_logger)
            current_logger.debug(f"Found {len(ground_truth_ast_node_ids)} AST node IDs to protect for ground truth.")

        current_logger.debug("Adding sequential dependencies for jumps...")
        add_sequential_dependencies_for_jumps(graph, tree, logger=current_logger)
        
        current_logger.debug("Removing next_syntax_edges_until_first_function_call...")
        remove_next_syntax_edges_until_first_function_call(graph, tree) # Ensure gast compatibility
        current_logger.debug("Removing last_reads...")
        remove_last_reads(graph, tree) # Ensure gast compatibility

        current_logger.debug("Reordering graph based on AST order...")
        ast_order = ASTOrder(graph) # Ensure gast compatibility
        ast_order.visit(tree)
        ast_order.reorder_graph()

        current_logger.debug("Removing CFG_NEXT edges between functions...")
        remove_cfg_next_edges_between_functions(graph) # Ensure gast compatibility
        current_logger.debug("Adding import dependencies...")
        add_import_dependencies(graph, tree) # Ensure gast compatibility
        current_logger.debug("Adding control block dependencies...")
        add_control_block_dependencies(graph) # Ensure gast compatibility

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(graph.nodes.keys())
        edges = [(edge.id1, edge.id2) for edge in graph.edges]
        nx_graph.add_edges_from(edges)

        current_logger.debug("Checking for cycles in the perturbation graph...")
        cycle = find_cycle(graph, logger=current_logger)
        
        if nx.is_directed_acyclic_graph(nx_graph):
            topo_sort = non_deterministic_topological_sort(nx_graph)
            reconstructed_ast, reordering_counts = graph_to_ast(graph, topo_sort, show_reorderings=show_reorderings,  ground_truth_node_ids=ground_truth_ast_node_ids)

            if show_reorderings and reordering_counts:
                print(f"Reordering counts: {reordering_counts}")
                total_reorderings = 1
                for _, count in reordering_counts.items():
                    total_reorderings *= count
                print(f"Total possible reorderings: {total_reorderings}")

            #usage_map = get_module_function_usage(reconstructed_ast)
            #reconstructed_ast = reorder_module_statements(reconstructed_ast, usage_map,logger=logger)
        else:
            print("The graph contains a cycle and cannot be topologically sorted.")
            return code, None

    else:
        # If perturbation is not enabled, simply parse the original code into an AST.
        reconstructed_ast = ast.parse(code)

    # Apply variable renaming transformation if enabled
    if apply_renaming:
        if reconstructed_ast is None: # Should only happen if apply_perturbation was false and parsing failed
            current_logger.warning("Cannot apply renaming because AST is None.")
        else:
            current_logger.debug("Applying variable renaming...")
            reconstructed_ast = variable_renaming(reconstructed_ast, rename_functions=True) # Ensure gast compatibility

    if reconstructed_ast is None:
        current_logger.error("Reconstructed AST is None before unparsing. Returning original code.")
        return code, None

    try:
        current_logger.debug("Unparsing final AST to code...")
        generated_code = astunparse.unparse(reconstructed_ast)
        # Sanity check: try to parse the generated code to ensure it's valid Python
        try:
            gast.parse(generated_code) # Use gast.parse for consistency
            current_logger.debug("Perturbation successful. Generated code is valid Python.")
        except SyntaxError as se:
            current_logger.error(f"Syntax error in the *final* generated code by perturbation module: {se}")
            current_logger.error(f"Problematic generated code snippet:\n{generated_code[:1000]}...")
            return code, None # Return original if final code is invalid
        return generated_code, total_reorderings
    except Exception as e:
        current_logger.error(f"Error during final unparsing of AST: {e}", exc_info=True)
        return code, None # Return original code on unparsing error


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Ensure logger is active
    problematic_code = """
    # ... (paste the full code that causes the "fprintln(*<gast.Name...>" error)
    # This is eval_prompt.replace("{{completion}}", ground_truth) from the failing sample
    # Example:
    # def fprintln(*objects, **kwargs):
    #     print(*objects, file=__output_file, **kwargs)
    # __output_file = None
    # # ... more code ...
    # def solve():
    #     pass # {{completion}}
    # """
