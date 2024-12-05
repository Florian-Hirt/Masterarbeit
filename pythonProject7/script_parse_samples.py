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
from python_graphs import program_graph_dataclasses as pb
import six

from digraph_transformer import dataflow_parser
import ogb_parser


# pylint: skip-file


def parse_sample(name, source, attr2idx, type2idx):
    """Parse source code with our graph construction procedure."""
    graph, tree = dataflow_parser.get_program_graph(source)

    ogd_data, label = dataflow_parser.py2ogbgraph(source, attr2idx, type2idx)

    # render(graph, f"graphs/parsed_pythongraphs_{name}.pdf")
    render(graph, f"./pythonProject7/graphs/parsed_pythongraphs_{name}.pdf")

    # df_ogb_render(ogd_data, f"graphs/parsed_{name}.pdf")

    df_ogb_render(ogd_data, f"./pythonProject7/graphs/parsed_{name}.pdf")

    print("\n\n----------------------------------------------------------\n")
    print(source)
    print(f"Data-flow centric generated for y={label}:")
    for key, value in ogd_data.items():
        if isinstance(value, np.ndarray) and key != "edge_index":
            value = value.transpose()
        print(f"{key}: {value}")


def parse_sample_python_graphs(name, source):
    graph = program_graph.get_program_graph(source)
    render(graph, f"./pythonProject7/graphs/parsed_pythongraphs_{name}.pdf")


def parse_sample_ogb(name, source, attr2idx, type2idx):
    """Parse source code with our graph OGB's construction procedure."""
    py2graph = functools.partial(
        ogb_parser.py2graph_helper, attr2idx=attr2idx, type2idx=type2idx
    )

    ogd_data, (ast_nodes, ast_edges) = py2graph(source)

    ogb_render(ast_nodes, ast_edges, path=f"./pythonProject7/graphs/ogb_{name}.pdf")

    print("\nOGB Code2 generated:")
    for key, value in ogd_data.items():
        if isinstance(value, np.ndarray) and key != "edge_index":
            value = value.transpose()
        print(f"{key}: {value}")


def example(a, b):
    a = a**2
    c = math.sqrt(b)
    return c + a


def transform_add(a, b: float = 3.14):
    a = a**2
    c = math.sqrt(b)
    return c + a


def transform_add_perm(a, b: float = 3.14):
    c = math.sqrt(b)
    a = a**2
    return a + c


def compl_transform_add(a, b: int = 2):
    a = a**2
    c = math.sqrt(b)
    a = math.tanh(a) + a / 2
    return a * c + b


def compl_transform_add_perm(a, b: int = 2):
    c = math.sqrt(b)
    a = a**2
    a = math.tanh(a) + a / 2
    return b + a * c


def for_loop(n=3):
    for i in range(n):
        i = try_catch(i)
        i = i - 1
    else:
        i += n
        i = i - 1
    i += 5
    return


def for_loop_chaos(n=3):
    for i in range(n):
        k = try_catch(i)
        j = k - 1
    else:
        j += n
        p = k + j - 1
    p += 5
    return


def for_loop_noret(n=3):
    for i in range(n):
        i = try_catch(i)
        i = i - 1


def while_break_continue(n=3):
    while True:
        if n == 0:
            break
        elif n < 0 or n > 100:
            n += 25
        else:
            n = n + 1
        n = n // 2
    return n


def no_inputs():
    print("Hello")
    print("World")


def try_catch(something: Any):
    """This is a helpful docstring.

    Args:
      something: don't know either
    Returns: literally something
    """
    try:
        something *= 2
        a = 12
    except Exception as e:
        print("Does not work")
        print(f"Oh forgot to tell that the error was {e}")
    finally:
        # A super helpful comment
        a = 42
        return something


def try_if_raise(something: Any):
    try:
        if isinstance(something, str):
            raise ValueError("Should not be string")
        else:
            something *= 2
            a = 12
    except Exception as e:
        print("Does not work")
        print(f"Oh forgot to tell that the error was {e}")
    finally:
        # A super helpful comment
        a = 42
        return something


def recursion(value: Union[int, float]):
    if value > 100:
        value = math.sqrt(value)
        value = value / 2
    return recursion(value)


def write(file, content):
    with open(file, "w") as f:
        f.write(content)
        print("Wrote {0}" % content)
    return


def assert_on_none(value):
    assert value is not None
    print(value)
    return value


def raise_uncaught(value):
    if value is None:
        raise ValueError()
    print(value)
    return value


def comprehend(keys, values):
    return {key: value for key, value in zip(keys, values)}


def intermediate_args(*args, last):
    print(last)


def bare_wildcard(*, last):
    print(last)


def f1_score(pred, label):
    correct = pred == label
    for i in range(10):
        print(correct)
    tp = (correct & label).sum()
    fn = (~correct & pred).sum()
    fp = (~correct & ~pred).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (recall * precision) / (recall + precision)


def test_function():
    i = 0
    while i < 5:
        print(f"Iteration {i}")
        i += 1


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
    # (example.__name__, inspect.getsource(example)),
    # (assert_on_none.__name__, inspect.getsource(assert_on_none)),
    # (async_print.__name__, inspect.getsource(async_print)),
    # (bare_wildcard.__name__, inspect.getsource(bare_wildcard)),
    # (compl_transform_add.__name__, inspect.getsource(compl_transform_add)),
    # (compl_transform_add_perm.__name__,
    #  inspect.getsource(compl_transform_add_perm)),
    # (comprehend.__name__, inspect.getsource(comprehend)),
    # (for_loop.__name__, inspect.getsource(for_loop)),
    # (for_loop_chaos.__name__, inspect.getsource(for_loop_chaos)),
    # (for_loop_noret.__name__, inspect.getsource(for_loop_noret)),
    # (intermediate_args.__name__, inspect.getsource(intermediate_args)),
    # (no_inputs.__name__, inspect.getsource(no_inputs)),
    # (raise_uncaught.__name__, inspect.getsource(raise_uncaught)),
    # (recursion.__name__, inspect.getsource(recursion)),
    # (transform_add.__name__, inspect.getsource(transform_add)),
    # (transform_add_perm.__name__, inspect.getsource(transform_add_perm)),
    # (try_catch.__name__, inspect.getsource(try_catch)),
    # (try_if_raise.__name__, inspect.getsource(try_if_raise)),
    # (while_break_continue.__name__, inspect.getsource(while_break_continue)),
    # (write.__name__, inspect.getsource(write)),
    # ('set_author_11836', set_author_11836)  # Example of OGB function
    # (f1_score.__name__, inspect.getsource(f1_score))
    (test_function.__name__, inspect.getsource(test_function))
]
if __name__ == "__main__":
    # For the OGB parser
    mapping_dir = "~/Downloads/code2/mapping"
    attr2idx_ = dict()
    type2idx_ = dict()
    for line in pd.read_csv(os.path.join(mapping_dir, "attridx2attr.csv.gz")).values:
        attr2idx_[line[1]] = int(line[0])
    for line in pd.read_csv(os.path.join(mapping_dir, "typeidx2type.csv.gz")).values:
        type2idx_[line[1]] = int(line[0])

    for name_, source_ in cases:
        parse_sample(name_, source_, attr2idx_, type2idx_)
        parse_sample_python_graphs(name_, source_)
        parse_sample_ogb(name_, source_, attr2idx_, type2idx_)
