"""Taken from https://github.com/snap-stanford/ogb/blob/c8efe8ec99d11279c80f2bcdbe1567675c1c5666/examples/graphproppred/code2/py2graph.py (only minor changes)."""

import ast

import networkx as nx
import numpy as np
import six

MASK_TOKEN = "_mask_"

# pylint: skip-file


class OGB_ASTWalker(ast.NodeVisitor):
    def __init__(self):
        self.node_id = 0
        self.stack = []
        self.graph = nx.Graph()
        self.nodes = {}

    def generic_visit(self, node):
        node_name = self.node_id
        self.node_id += 1

        # if available, extract AST node attributes
        name = getattr(node, "name", None)
        arg = getattr(node, "arg", None)
        s = getattr(node, "s", None)
        n = getattr(node, "n", None)
        id_ = getattr(node, "id", None)
        attr = getattr(node, "attr", None)

        values = [name, arg, s, n, id_, attr]
        node_value = next((str(value) for value in values if value is not None), None)
        if isinstance(node_value, str):
            node_value = node_value.encode("utf-8", errors="surrogatepass")

        # encapsulate all node features in a dict
        self.nodes[node_name] = {
            "type": type(node).__name__,
            "attribute": node_value.decode("UTF-8")
            if node_value is not None
            else node_value,
            "attributed": True if node_value != None else False,
            "depth": len(self.stack),
            "dfs_order": node_name,
        }

        # DFS traversal logic
        parent_name = None
        if self.stack:
            parent_name = self.stack[-1]
        self.stack.append(node_name)
        self.graph.add_node(node_name)
        if parent_name != None:
            # replicate AST as NetworkX object
            self.graph.add_edge(node_name, parent_name)
        super().generic_visit(node)
        self.stack.pop()


def py2graph_helper(code, attr2idx, type2idx, mask=False):
    """Input: code: code snippet

    Mappers:
    - attr_mapping: mapping from attribute to integer idx
    - type_mapping: mapping from type to integer idx
    - mask (bool): whether to mask the method name or not.
    If we do method naming, we need to set it to True. Otherwise, there is data
    leakage.

    Output: OGB graph object
    """

    tree = ast.parse(code)
    walker = OGB_ASTWalker()
    walker.visit(tree)

    ast_nodes, ast_edges = walker.nodes, walker.graph.edges()

    if mask:
        assert "FunctionDef" in ast_nodes[1]["type"], (
            "To mask method name, 1st node in AST must be of type " "FunctionDef"
        )
        method_name = ast_nodes[1]["attribute"]
        for idx, ast_node in ast_nodes.items():
            if (
                "FunctionDef" in ast_node["type"]
                and ast_node["attribute"] == method_name
            ):
                ast_nodes[idx]["attribute"] = MASK_TOKEN

    # print(ast_nodes)

    data = dict()

    # first dim: type
    # second dim: attr

    # meta-info
    # dfs_order: integer
    # attributed: 0 or 1

    node_feat_raw = []
    node_feat = []
    dfs_order = []
    depth = []
    attributed = []
    for i in range(len(ast_nodes)):
        typ = ast_nodes[i]["type"] if ast_nodes[i]["type"] in type2idx else "__UNK__"

        if ast_nodes[i]["attributed"]:
            attr = (
                ast_nodes[i]["attribute"]
                if ast_nodes[i]["attribute"] in attr2idx
                else "__UNK__"
            )
        else:
            attr = "__NONE__"

        node_feat_raw.append([typ, attr])
        node_feat.append([type2idx[typ], attr2idx[attr]])

        dfs_order.append(ast_nodes[i]["dfs_order"])
        depth.append(ast_nodes[i]["depth"])
        attributed.append(ast_nodes[i]["attributed"])

    ### meta-information
    data["node_feat_raw"] = np.array(node_feat_raw, dtype=str)
    data["node_feat"] = np.array(node_feat, dtype=np.int64)
    data["node_dfs_order"] = np.array(dfs_order, dtype=np.int64).reshape(-1, 1)
    data["node_depth"] = np.array(depth, dtype=np.int64).reshape(-1, 1)
    data["node_is_attributed"] = np.array(attributed, dtype=np.int64).reshape(-1, 1)

    data["edge_index"] = np.array([[i, j] for i, j in ast_edges]).transpose()

    data["num_nodes"] = len(data["node_feat"])
    data["num_edges"] = len(data["edge_index"][0])

    return data, (ast_nodes, ast_edges)


def try_ensure_str(value):
    try:
        return six.ensure_str(value, "utf-8")
    except:
        try:
            return str(value)
        except:
            return "<no-str>"


def test_transform(py2graph, code):
    graph, _ = py2graph(code)
    print(graph)

    invalid_code = """
    import antigravity
    xkcd loves Python
    """

    try:
        graph = py2graph(invalid_code)
    except SyntaxError:
        print("Successfully caught syntax error")
