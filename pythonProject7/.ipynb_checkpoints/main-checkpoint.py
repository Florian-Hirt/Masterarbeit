import ast
from digraph_transformer import dataflow_parser


def print_ast_details(body, label):
    print(f"{label}:\n")
    for stmt in body:
        print(ast.dump(stmt, annotate_fields=True, include_attributes=True))
    print("\n" + "=" * 50 + "\n")


if __name__ == '__main__':
    # Code-Snippets of three f1 scores and one 'false' f1 score
    code1 = """
    def f1_score(pred, label):
        correct = pred == label
        tp = (correct & label).sum()
        fn = (~correct & pred).sum()
        fp = (~correct & ~pred).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (2 * (recall * precision) / (recall + precision))
    """

    code2 = """
    def f1_score(pred, label):
        correct = pred == label
        tp = (correct & label).sum()
        fn = (~correct & ~pred).sum()
        fp = (~correct & pred).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (2 * (precision * recall) / (precision + recall))
    """

    code3 = """
    def f1_score(pred, label):
        correct = pred == label
        tp = (correct & label).sum()
        fn = (~correct & ~pred).sum()
        recall = tp / (tp + fn)
        fp = (~correct & pred).sum()
        precision = tp / (tp + fp)
        return (2 * (precision * recall) / (precision + recall))
    """

    incorrect_code = """
    def f1_score(pred, label):
        correct = pred != label # unequal
        tp = (correct & label).sum()
        fn = (~correct & ~pred).sum()
        recall = (tp + fn) / tp # changed Zähler and Nenner
        fp = (~correct & pred).sum()
        precision = tp / (tp - fp) # minus instead of plus
        return (2 * (precision * recall) / (precision + recall))
    """

    # Parse the ASTs
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    tree3 = ast.parse(code3)
    tree_incorrect = ast.parse(incorrect_code)

    # Extract the Function Bodies
    function_body1 = tree1.body[0].body
    function_body2 = tree2.body[0].body
    function_body3 = tree3.body[0].body
    function_body_incorrect = tree_incorrect.body[0].body

    # Function for detailed output of the AST nodes

    # AST representation for each code snippet
    print_ast_details(function_body1, "Code 1")
    # print_ast_details(function_body2, "Code 2")
    # print_ast_details(function_body3, "Code 3")
    # print_ast_details(function_body_incorrect, "Incorrect Code")

    graph = dataflow_parser.get_program_graph(tree1)

    print(graph)
