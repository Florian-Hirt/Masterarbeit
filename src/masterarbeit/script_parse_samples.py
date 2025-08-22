### MODIFIED VERSION OF THE ORIGINAL FILE 


"""Graphviz visualizations of sample Program Graphs."""

import functools
import inspect
import math
import os
from typing import Any, Union

import numpy as np
import pandas as pd
import pygraphviz
from python_graphs import program_graph
from python_graphs.program_graph_dataclasses import EdgeType
from python_graphs import program_graph_dataclasses as pb
import six

from digraph_transformer import dataflow_parser
import ogb_parser

####################################################################################################
####################################################################################################
####################################################################################################
### My modifications:
import ast
import gast
####################################################################################################
####################################################################################################
####################################################################################################


# pylint: skip-file

####################################################################################################
####################################################################################################
####################################################################################################
### My modifications:

class ASTOrder(ast.NodeVisitor):
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
            if self.node_to_order[id(self.graph.nodes[edge.id1].ast_node)] > self.node_to_order[id(self.graph.nodes[edge.id2].ast_node)]:
                if self.creates_cycle(edge):
                    edges_to_remove.append(edge)
            elif edge.id1 == edge.id2:
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            self.graph.edges.remove(edge)

    def creates_cycle(self, edge):
        visited = set()
        stack = [edge.id2]
        while stack:
            node_id = stack.pop()
            if node_id == edge.id1:
                return True
            if node_id not in visited:
                visited.add(node_id)
                stack.extend([e.id2 for e in self.graph.edges if e.id1 == node_id])
        return False


class ParentTrackingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent_map = {}

    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node  
            self.visit(child)

    def get_parents(self, node):
        parents = []
        current = self.parent_map.get(node, None)
        while current:
            parents.append(current)
            current = self.parent_map.get(current, None)
        return parents

def remove_last_reads(graph, ast_tree):
    visitor = ParentTrackingVisitor()
    visitor.visit(ast_tree)

    edges_to_remove = []
    for edge in graph.edges:
        if edge.type == EdgeType.LAST_READ:
            node1 = graph.get_node(edge.id1)
            node2 = graph.get_node(edge.id2)

            parents1 = visitor.get_parents(node1.ast_node)
            parents2 = visitor.get_parents(node2.ast_node)
            parents = parents1 + parents2
            parent_classes = [p.__class__.__name__ for p in parents]
            
            if "Call" in parent_classes:
                # built-in functions 
                white_list_builtin = ["sum", "mean", "max", "min", "len", "sorted", "reversed", "enumerate", "range", "zip", "map", "filter", "all", "any"]
                # numpy functions
                white_list_numpy = ["array", "arange", "linspace", "zeros", "ones", "empty", "full", "eye", "identity", "random", "dot", "matmul", "linalg", "fft", "mean", "median", "std", "var", "sum", "prod", "cumsum", "cumprod", "min", "max", "argmin", "argmax", "argsort", "sort", "unique", "reshape", "transpose", "concatenate", "stack", "hstack", "vstack", "split", "hsplit", "vsplit"]

                if any([isinstance(parent, gast.gast.Call) and isinstance(parent.func, gast.gast.Attribute) and (parent.func.attr in white_list_builtin or parent.func.attr in white_list_numpy) for parent in parents]):
                    edges_to_remove.append(edge)
                else:
                    pass
            else:
                edges_to_remove.append(edge)

    for edge in edges_to_remove:
        graph.edges.remove(edge)

class ImportDependencyVisitor(ast.NodeVisitor):
    """
    After the ProgramGraph is built, walk the AST again.
    Track each import as if it were a variable assignment.
    Then add edges from the import statement to every usage of that name.
    """
    def __init__(self, graph):
        self.graph = graph
        # Map a variable name to the node ID where it's "imported"
        self.import_writes = {}

    def visit_Import(self, node):
        node_id = self.graph.get_node_by_ast_node(node).id
        for alias in node.names:
            # alias.name is what's imported; alias.asname is the "local name" if any
            local_name = alias.asname if alias.asname else alias.name
            self.import_writes[local_name] = node_id
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        For `from itertools import zip_longest` or `from itertools import zip_longest as z`,
        record that 'zip_longest' or 'z' is 'written' by this node.
        """
        node_id = self.graph.get_node_by_ast_node(node).id
        for alias in node.names:
            local_name = alias.asname if alias.asname else alias.name
            self.import_writes[local_name] = node_id
        self.generic_visit(node)

    def visit_Name(self, node):
        """
        Whenever we see a usage of a name in `Load` context, check if it was imported.
        If so, add a LAST_READ edge from the import node to this usage node.
        """
        if isinstance(node.ctx, ast.Load):
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


def add_control_block_dependencies(graph):
    def get_descendants(node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited:
            return set()
        
        visited.add(node_id)
        descendants = set()
        
        for edge in graph.edges:
            if edge.id1 == node_id:  # Nachfolgerknoten
                descendants.add(edge.id2)
                descendants.update(get_descendants(edge.id2, visited))
        
        return descendants
    
    def would_create_cycle(source_id, target_id):
        """Checks if an edge from source_id to target_id would create a cycle."""
        if source_id == target_id:
            return True
            
        visited = set()
        to_visit = [target_id]
        
        while to_visit:
            current = to_visit.pop()
            if current == source_id:
                return True
                
            if current not in visited:
                visited.add(current)
                for edge in graph.edges:
                    if edge.id1 == current:
                        to_visit.append(edge.id2)
        
        return False


    for node_id in graph.nodes:
        ast_node = graph.get_node(node_id).ast_node
        node_type = ast_node.__class__.__name__

        if node_type in ["If", "For", "Try", "While", "With", "AugAssign", "Assign"]:
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

            if node_type == "AugAssign":
                block_nodes.add(graph.get_node_by_ast_node(ast_node.target).id)
                block_nodes.add(graph.get_node_by_ast_node(ast_node.value).id)

            if node_type == "Assign":
                block_nodes.add(graph.get_node_by_ast_node(ast_node.targets[0]).id)
                block_nodes.add(graph.get_node_by_ast_node(ast_node.value).id)

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
                            if new_edge not in graph.edges and not would_create_cycle(edge.id1, node_id):
                                graph.add_edge(new_edge)

            
                    
def remove_cfg_next_edges_between_functions(graph):
    ''' Remove CFG_NEXT edges between functions if they are in the same module or classo
    '''
    edges_to_remove = []
    for edge in graph.edges:
        if edge.type == EdgeType.CFG_NEXT:
            node1 = graph.get_node(edge.id1).ast_node
            node2 = graph.get_node(edge.id2).ast_node
            if node1.__class__.__name__ == "FunctionDef" or node2.__class__.__name__ == "FunctionDef":

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
        # Remove next_syntax edges until the first function call in execution mode
        edges_to_keep = []
        found_first_function_call = False

        for edge in graph.edges:
            if edge.type.value == 9:
                # print(graph.get_node(edge.id1).ast_node.__class__, graph.get_node(edge.id2).ast_node.__class__)
                if not found_first_function_call:
                    node = graph.get_node(edge.id1).ast_node
                    if isinstance(node, gast.gast.Call) or any(isinstance(child, gast.gast.Call) for child in ast.iter_child_nodes(node)):
                        visitor = ParentTrackingVisitor()
                        visitor.visit(ast_tree)
                        parents = visitor.get_parents(node)
                        parent_classes = [p.__class__.__name__ for p in parents]
                        
                        if not "FunctionDef" in parent_classes:
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
        print(found_first_function_call)

        graph.edges = edges_to_keep

def would_create_cycle(source_id, target_id, graph):
        """Checks if an edge from source_id to target_id would create a cycle."""
        if source_id == target_id:
            return True
            
        visited = set()
        to_visit = [target_id]
        
        while to_visit:
            current = to_visit.pop()
            if current == source_id:
                return True
                
            if current not in visited:
                visited.add(current)
                for edge in graph.edges:
                    if edge.id1 == current:
                        to_visit.append(edge.id2)
        
        return False

class SequentialJumpDependencyVisitor(gast.NodeVisitor):
    """
    Adds CFG_NEXT dependencies to ensure statements preceding a break or continue
    in the same syntactic block are ordered before the jump statement.
    This helps preserve the intended control flow when reordering.
    """
    def __init__(self, graph):
        self.graph = graph
        # Assumes graph has a method get_node_by_ast_node
        # and it can handle gast nodes.

    def _process_statement_list(self, stmt_list):
        if not stmt_list or not isinstance(stmt_list, list):
            return

        for i, current_stmt_ast in enumerate(stmt_list):
            # Ensure current_stmt_ast is a valid AST node
            if not isinstance(current_stmt_ast, gast.AST):
                continue

            if isinstance(current_stmt_ast, (gast.Break, gast.Continue)):
                current_stmt_node_obj = self.graph.get_node_by_ast_node(current_stmt_ast)
                if not current_stmt_node_obj:
                    # This node might not be in the graph if it was filtered or is a comment proxy, etc.
                    # print(f"Warning: AST node {type(current_stmt_ast)} not found in graph during jump dependency analysis.")
                    continue
                
                current_stmt_node_id = current_stmt_node_obj.id

                for j in range(i): # Iterate over all preceding statements in this list
                    prev_stmt_ast = stmt_list[j]
                    if not isinstance(prev_stmt_ast, gast.AST):
                        continue

                    prev_stmt_node_obj = self.graph.get_node_by_ast_node(prev_stmt_ast)
                    if not prev_stmt_node_obj:
                        # print(f"Warning: AST node {type(prev_stmt_ast)} (preceding jump) not found in graph.")
                        continue
                    
                    prev_stmt_node_id = prev_stmt_node_obj.id

                    # Add a CFG_NEXT edge from the previous statement to the break/continue statement.
                    # This signifies that prev_stmt_ast must be processed/evaluated before
                    # current_stmt_ast (the jump) in this specific syntactic sequence.
                    new_edge = pb.Edge(
                        id1=prev_stmt_node_id,
                        id2=current_stmt_node_id,
                        type=EdgeType.CFG_NEXT # Using CFG_NEXT as it represents sequential control flow
                    )

                    if prev_stmt_node_id != current_stmt_node_id and \
                       new_edge not in self.graph.edges and \
                       not would_create_cycle(new_edge.id1, new_edge.id2, self.graph):
                        self.graph.add_edge(new_edge)
            
            # Note: The actual recursive traversal into compound statements (If, For, etc.)
            # to find nested statement lists is handled by the standard NodeVisitor mechanism
            # (i.e., specific visit_XYZ methods calling _process_statement_list and self.generic_visit).

    # Override visit methods for nodes that can contain lists of statements

    def visit_FunctionDef(self, node: gast.FunctionDef):
        self._process_statement_list(node.body)
        self.generic_visit(node) # Process decorators, args, return annotations etc.

    def visit_If(self, node: gast.If):
        self._process_statement_list(node.body)
        self._process_statement_list(node.orelse)
        self.generic_visit(node) # Process test expression

    def visit_For(self, node: gast.For):
        self._process_statement_list(node.body)
        self._process_statement_list(node.orelse)
        self.generic_visit(node) # Process target, iter

    def visit_While(self, node: gast.While):
        self._process_statement_list(node.body)
        self._process_statement_list(node.orelse)
        self.generic_visit(node) # Process test

    def visit_Try(self, node: gast.Try):
        self._process_statement_list(node.body)
        for handler in node.handlers:
            # gast.Try has handlers as a list of gast.ExceptionHandler nodes
            if isinstance(handler, gast.ExceptionHandler): 
                 self._process_statement_list(handler.body)
        self._process_statement_list(node.orelse)
        self._process_statement_list(node.finalbody)
        self.generic_visit(node) # Process handlers' types/names if any, etc.

    def visit_With(self, node: gast.With):
        self._process_statement_list(node.body)
        self.generic_visit(node) # Process withitems

    def visit_ClassDef(self, node: gast.ClassDef):
        # Class bodies don't directly contain break/continue affecting other class-level statements.
        # Methods (FunctionDef) within the class will be visited by visit_FunctionDef.
        # Assignments or other statements at class level are not typically followed by jumps.
        self.generic_visit(node) # Process bases, keywords, decorators, and body statements (like methods)


def add_sequential_dependencies_for_jumps(graph, ast_tree):
    """
    Traverses the AST and adds CFG_NEXT edges to the graph
    for statements that must precede break/continue statements
    within the same syntactic block.
    """
    visitor = SequentialJumpDependencyVisitor(graph)
    visitor.visit(ast_tree)
                
####################################################################################################
####################################################################################################
####################################################################################################




def parse_sample(name, source, attr2idx, type2idx):
    """Parse source code with our graph construction procedure."""
    print(source)
    graph, tree = dataflow_parser.get_program_graph(source)

    add_sequential_dependencies_for_jumps(graph, tree)

    print("Original")
    print("Number of nodes in the graph:", len(graph.nodes))
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))


    remove_next_syntax_edges_until_first_function_call(graph, tree)

    print("After removing next syntax edges until first function call")
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))

    # # ####################################################################################################
    # # ####################################################################################################
    # # ####################################################################################################
    # # ### My modifications:
    # # Add control block dependencies
    remove_last_reads(graph, tree)

    print("After removing last reads")
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))

    ast_order = ASTOrder(graph)
    ast_order.visit(tree)
    # Reorder the graph to drop cycles
    ast_order.reorder_graph()

    print("After reordering the graph")
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))

    # # Remove CFG_NEXT edges between functions if they are in the same module or class
    remove_cfg_next_edges_between_functions(graph)
    print("After removing CFG_NEXT edges between functions")
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))

    add_import_dependencies(graph, tree)

    print("After adding import dependencies")
    print("Number of edges in the graph:", len(graph.edges))
    print("Edge types in the graph:", set(edge.type for edge in graph.edges))

    # add_control_block_dependencies(graph)

    print("After adding control block dependencies")
    print("Number of edges in the graph:", len(graph.edges))


    ogd_data, label = dataflow_parser.py2ogbgraph(source, attr2idx, type2idx)

    # The existing render call for the full program graph (from the second get_program_graph call)
    # This 'graph' is the unmodified program graph from the second call to dataflow_parser.get_program_graph
    print(f"\n--- Full Program Graph Visualization for {name} (from second parse) ---")
    render(graph, f"docs/figures/parsed_pythongraphs_{name}.pdf")
    print(f"Full program graph saved to docs/figures/parsed_pythongraphs_{name}.pdf")
    print(f"Number of nodes in full program graph: {len(graph.nodes)}")
    print(f"Number of edges in full program graph: {len(graph.edges)}")
    print(f"Edge types in full program graph: {set(edge.type for edge in graph.edges)}") # Corrected to use edge.type
    print("--------------------------------------------------\n")

    # The existing render call for the OGB graph
    print(f"\n--- OGB Graph Visualization for {name} ---")
    df_ogb_render(ogd_data, f"docs/figures/parsed_{name}.pdf")
    print(f"OGB graph saved to docs/figures/parsed_{name}.pdf")
    print("--------------------------------------------------\n")

    print("\n\n----------------------------------------------------------\n")
    print(f"Source code for {name}:") # This line was part of the original selection's context, added for clarity
    print(source)
    print(f"Data-flow centric OGB data generated for y={label}:")
    for key, value in ogd_data.items():
        if isinstance(value, np.ndarray) and key != "edge_index":
            value = value.transpose()
        print(f"{key}: {value}")


def visualize_ast_from_graph(name, base_graph):
        """Creates and renders an AST-only visualization from a ProgramGraph."""
        # Create a new graph for AST visualization based on the provided 'base_graph'
        ast_visualization_graph = program_graph.ProgramGraph()
        # Copy node-related information. Nodes are shared.
        ast_visualization_graph.nodes = base_graph.nodes
        ast_visualization_graph.root_id = base_graph.root_id
        
        # Filter for AST (FIELD) edges only
        ast_edges = []
        for edge_obj in base_graph.edges: 
            if edge_obj.type == EdgeType.FIELD:
                ast_edges.append(edge_obj)
        ast_visualization_graph.edges = ast_edges
        
        # Render the AST-only graph
        print(f"\n--- AST-only Graph Visualization for {name} ---")
        render(ast_visualization_graph, f"docs/figures/ast_only_{name}.pdf")
        print(f"AST-only graph saved to docs/figures/ast_only_{name}.pdf")
        print(f"Number of nodes in AST-only graph: {len(ast_visualization_graph.nodes)}")
        print(f"Number of edges in AST-only graph: {len(ast_visualization_graph.edges)} (FIELD edges only)")
        print("--------------------------------------------------\n")

def generate_and_render_ast_only(name, source):
    """Parses source and renders an AST-only visualization."""
    # This function is for standalone AST visualization, e.g., from __main__
    # It performs its own parse.
    graph_for_ast, _ = dataflow_parser.get_program_graph(source)
    visualize_ast_from_graph(name, graph_for_ast)


def parse_sample_python_graphs(name, source):
    graph = program_graph.get_program_graph(source)
    render(graph, f"docs/figures/parsed_pythongraphs_{name}.pdf")


def parse_sample_ogb(name, source, attr2idx, type2idx):
    """Parse source code with our graph OGB's construction procedure."""
    py2graph = functools.partial(
        ogb_parser.py2graph_helper, attr2idx=attr2idx, type2idx=type2idx
    )

    ogd_data, (ast_nodes, ast_edges) = py2graph(source)

    ogb_render(ast_nodes, ast_edges, path=f"docs/figures/ogb_{name}.pdf")

    print("\nOGB Code2 generated:")
    for key, value in ogd_data.items():
        if isinstance(value, np.ndarray) and key != "edge_index":
            value = value.transpose()
        print(f"{key}: {value}")



def test_function():
    flag = (1 > 2) or (3 < 4)
    print(flag)

# Here as string to avoid linting errors
set_author_11836 = r"""def set_author(self, *, name, url=EmptyEmbed, icon_url=EmptyEmbed):
  self._author = {
      'name': str(name)
  }

  if url is not EmptyEmbed:
    self._author['url'] = str(url) 

  if icon_url is not EmptyEmbed:
    self._author['icon_url'] = str(icon_url)

  return self
"""


async def async_print(content):
    print(content)


def try_ensure_str(value):
    try:
        return six.ensure_str(value, "utf-8")
    except:
        try:
            return str(value)
        except:
            return "<no-str>"


def to_graphviz(graph):
    """Creates a graphviz representation of a ProgramGraph.

    Args:
      graph: A ProgramGraph object to visualize.

    Returns:
      A pygraphviz object representing the ProgramGraph.
    """
    g = pygraphviz.AGraph(strict=False, directed=True)
    for unused_key, node in graph.nodes.items():
        node_attrs = {}
        if node.ast_type:
            node_attrs["label"] = six.ensure_str(node.ast_type, "utf-8")
            if hasattr(node, "ast_value") and node.ast_value:
                node_attrs["label"] += "\n" + try_ensure_str(node.ast_value)
            if hasattr(node, "fields") and isinstance(node.fields, dict):
                for field, value in node.fields.items():
                    if value:
                        node_attrs["label"] += f"\n{field}:" + try_ensure_str(value)
            node_attrs["color"] = "purple"
            node_attrs["colorscheme"] = "svg"
        elif hasattr(node, "ast_value") and node.ast_value:
            node_attrs["label"] = try_ensure_str(node.ast_value)
        else:
            node_attrs["shape"] = "point"
        # node_type_colors = {}
        # node_type = node.type if hasattr(node, type) else node.node_type
        # breakpoint()
        # if node_type in node_type_colors:
        #   node_attrs['color'] = node_type_colors[node_type]
        #   node_attrs['colorscheme'] = 'svg'

        g.add_node(node.id, **node_attrs)
    for edge in graph.edges:
        edge_attrs = {}
        edge_attrs["label"] = edge.type.name
        if hasattr(edge, "field_name") and edge.field_name:
            edge_attrs["label"] += "\n" + try_ensure_str(edge.field_name)
        edge_colors = {
            pb.EdgeType.LAST_READ: "green",
            pb.EdgeType.LAST_WRITE: "orange",
            pb.EdgeType.COMPUTED_FROM: "blue",
            pb.EdgeType.CFG_NEXT: "red",
            pb.EdgeType.NEXT_SYNTAX: "purple",
        }
        if edge.type in edge_colors:
            edge_attrs["color"] = edge_colors[edge.type]
            edge_attrs["colorscheme"] = "svg"
        g.add_edge(edge.id1, edge.id2, **edge_attrs)
    return g


def render(graph, path="/tmp/graph.png"):
    g = to_graphviz(graph)
    g.draw(path, prog="dot")


def df_ogb_to_graphviz(ogb_data):
    """Creates a graphviz representation of a OGB graph.

    Args:
      graph: DF OGB data.

    Returns:
      A pygraphviz object representing the ProgramGraph.
    """
    g = pygraphviz.AGraph(strict=False, directed=True)
    for node_id in range(ogb_data["num_nodes"]):
        node_attrs = {}
        node_attrs["label"] = try_ensure_str(ogb_data["node_feat_raw"][node_id][0])
        value = try_ensure_str(ogb_data["node_feat_raw"][node_id][1])
        if value != "__NONE__":
            node_attrs["label"] += "\n" + value
        node_attrs["color"] = "purple"
        node_attrs["colorscheme"] = "svg"

        g.add_node(node_id, **node_attrs)

    for edge_id in range(ogb_data["num_edges"]):
        edge_attrs = {}
        edge_type = pb.EdgeType(ogb_data["edge_type"][edge_id][0])
        if edge_type.name == "NEXT_SYNTAX":
            continue
        edge_attrs["label"] = edge_type.name

        value = try_ensure_str(ogb_data["edge_name"][edge_id][0])
        if value != "__NONE__":
            edge_attrs["label"] += "\n" + value
            if ogb_data["edge_order"][edge_id][0]:
                edge_attrs["label"] += ":" + str(ogb_data["edge_order"][edge_id][0])

        edge_colors = {
            pb.EdgeType.LAST_READ: "green",
            pb.EdgeType.LAST_WRITE: "orange",
            pb.EdgeType.COMPUTED_FROM: "blue",
            pb.EdgeType.CFG_NEXT: "red",
            pb.EdgeType.NEXT_SYNTAX: "purple",
        }
        if edge_type in edge_colors:
            edge_attrs["color"] = edge_colors[edge_type]
            edge_attrs["colorscheme"] = "svg"
        g.add_edge(
            ogb_data["edge_index"][0][edge_id],
            ogb_data["edge_index"][1][edge_id],
            **edge_attrs,
        )
    return g


def df_ogb_render(ogb_data, path="/tmp/graph.png"):
    g = df_ogb_to_graphviz(ogb_data)
    # g.graph_attr.update(size="3,4")
    g.draw(path, prog="dot")


def ogb_to_graphviz(ast_nodes, ast_edges):
    """Creates a graphviz representation of a ProgramGraph.

    Args:
      graph: A ProgramGraph object to visualize.

    Returns:
      A pygraphviz object representing the ProgramGraph.
    """
    g = pygraphviz.AGraph(strict=False, directed=True)
    for idx, node in ast_nodes.items():
        node_attrs = {}
        node_attrs["label"] = node["type"]
        if node["attributed"]:
            node_attrs["label"] += "\n" + node["attribute"]
        node_attrs["color"] = "purple"
        node_attrs["colorscheme"] = "svg"

        g.add_node(idx, **node_attrs)
    for src, dest in ast_edges:
        g.add_edge(src, dest)

    predecessor = -1
    edge_attrs = {}
    edge_attrs["color"] = "red"
    edge_attrs["colorscheme"] = "svg"
    for idx, node in ast_nodes.items():
        if node["attributed"]:
            if predecessor >= 0:
                g.add_edge(predecessor, idx, **edge_attrs)
            predecessor = idx
    return g


def ogb_render(ast_nodes, ast_edges, path="/tmp/graph.png"):
    g = ogb_to_graphviz(ast_nodes, ast_edges)
    g.draw(path, prog="dot")


cases = [
    (test_function.__name__, inspect.getsource(test_function)),
]

if __name__ == "__main__":
    # For the OGB parser
    mapping_dir = "data/code2/mapping"
    attr2idx_ = dict()
    type2idx_ = dict()
    for line in pd.read_csv(os.path.join(mapping_dir, "attridx2attr.csv.gz")).values:
        attr2idx_[line[1]] = int(line[0])
    for line in pd.read_csv(os.path.join(mapping_dir, "typeidx2type.csv.gz")).values:
        type2idx_[line[1]] = int(line[0])

    for name_, source_ in cases:
        generate_and_render_ast_only(name_, source_)
        parse_sample(name_, source_, attr2idx_, type2idx_)
        parse_sample_python_graphs(name_, source_)
        parse_sample_ogb(name_, source_, attr2idx_, type2idx_)

 
