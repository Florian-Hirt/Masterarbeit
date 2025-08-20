from typing import Dict, List, Tuple, Optional, Any, Callable
import ast
import gast
import networkx as nx
import sys
from networkx import DiGraph
import hashlib
import astunparse



# Import existing modules
from perturbation import (
    non_deterministic_topological_sort, neighborhood_topological_sort,
    graph_to_ast)

from digraph_transformer.dataflow_parser import (remove_next_syntax_edges_until_first_function_call, remove_last_reads,
    ASTOrder, remove_cfg_next_edges_between_functions, add_import_dependencies,
    add_control_block_dependencies)




from Python_Files.code_completion_generation import incomplete_code_transform

class Edge:
    id1: str
    id2: str

class Graph:
    nodes: Dict[str, Any]
    edges: List[Edge]

NodeID = str
TopoSort = List[NodeID]


class AdversarialReorderingSearch:
    def __init__(
        self,
        logger,
        evaluation_function_batch: Callable[[List[str], List[str], List[str]], List[Optional[int]]],
        initial_samples: int = 10,
        neighborhood_samples: int = 5,
        iterations: int = 3,
        change_probability: float = 0.3,
    ) -> None:
        self.evaluation_function_batch = evaluation_function_batch
        self.initial_samples = initial_samples
        self.neighborhood_samples = neighborhood_samples
        self.iterations = iterations
        self.change_probability = change_probability
        self.results_cache: Dict[str, Tuple[Optional[float], str]] = {}
        self.logger = logger

    def reset_cache(self) -> None:
        self.results_cache = {}

    def get_reordered_code(
        self,
        original_code_for_fallback: str,
        topo_sort: Optional[TopoSort] = None,
        base_graph: Optional[Graph] = None,
        base_ast_tree: Optional[ast.AST] = None
    ) -> Tuple[str, Optional[Graph], Optional[ast.AST], Optional[DiGraph]]:
        """Generate reordered code with better error handling."""
        if base_graph is None or base_ast_tree is None:
            try:
                from digraph_transformer import dataflow_parser
                self.logger.debug(f"Generating base graph/AST")
                current_graph, current_ast = dataflow_parser.get_program_graph(original_code_for_fallback)

                if current_graph is None or current_ast is None:
                    self.logger.warning("Failed to generate base graph/AST.")
                    return original_code_for_fallback, None, None, None

                # Apply transformations
                self._apply_graph_transformations(current_graph, current_ast)
                
                base_graph = current_graph
                base_ast_tree = current_ast
            except Exception as e:
                self.logger.error(f"Error during initial graph/AST creation: {e}")
                return original_code_for_fallback, None, None, None

        try:
            # Convert to NetworkX graph
            nx_graph = self._create_nx_graph(base_graph)
            
            if not nx.is_directed_acyclic_graph(nx_graph):
                self.logger.warning("Graph contains cycles.")
                return original_code_for_fallback, base_graph, base_ast_tree, nx_graph

            if topo_sort is None:
                topo_sort = non_deterministic_topological_sort(nx_graph)

            # Reconstruct code
            reordered_code = self._reconstruct_code(base_graph, topo_sort, original_code_for_fallback)
            return reordered_code, base_graph, base_ast_tree, nx_graph
            
        except Exception as e:
            self.logger.error(f"Error during reordering: {e}")
            return original_code_for_fallback, base_graph, base_ast_tree, None

    def _apply_graph_transformations(self, graph: Graph, ast_tree: ast.AST) -> None:
        """Apply all necessary graph transformations."""
        remove_next_syntax_edges_until_first_function_call(graph, ast_tree)
        remove_last_reads(graph, ast_tree)
        ast_order = ASTOrder(graph)
        ast_order.visit(ast_tree)
        ast_order.reorder_graph()
        remove_cfg_next_edges_between_functions(graph)
        add_import_dependencies(graph, ast_tree)
        add_control_block_dependencies(graph)

    def _create_nx_graph(self, base_graph: Graph) -> nx.DiGraph:
        """Create NetworkX graph from custom Graph object."""
        nx_graph = nx.DiGraph()
        if base_graph and base_graph.nodes:
            nx_graph.add_nodes_from(base_graph.nodes.keys())
        if base_graph and base_graph.edges:
            edges = [(edge.id1, edge.id2) for edge in base_graph.edges]
            nx_graph.add_edges_from(edges)
        return nx_graph

    def _reconstruct_code(self, base_graph: Graph, topo_sort: TopoSort, original_code: str) -> str:
        """Reconstruct code from graph and topological sort."""
        try:
            reconstructed_gast_ast_obj, _ = graph_to_ast(base_graph, topo_sort, show_reorderings=False)  
            self.logger.info(f"Reconstructed GAST AST: {reconstructed_gast_ast_obj}")  

            ast_for_unparse_obj = gast.gast_to_ast(reconstructed_gast_ast_obj)

            ast_for_unparse_obj = gast.gast_to_ast(reconstructed_gast_ast_obj)
            self.logger.info(f"AST for unparse: <ast.Module object at {hex(id(ast_for_unparse_obj))}>")
            ast.fix_missing_locations(ast_for_unparse_obj) # Keep this
      
            if ast_for_unparse_obj:
                reordered_code = astunparse.unparse(ast_for_unparse_obj)
            
            # Validate syntax
            ast.parse(reordered_code)
            self.logger.info("Reordered code is valid Python syntax.")
            return reordered_code
            
        except (SyntaxError, KeyError) as e:
            self.logger.warning(f"Failed to reconstruct valid code: {e}")
            self.logger.warning(f"Returning original code as fallback: {original_code}")
            import traceback
            self.logger.error("Full traceback for reconstruction failure:")
            traceback.print_exc(file=sys.stderr) # Print to stderr to ensure it's visible
            return original_code

    def search(
        self,
        original_code_param: str,
        source_code_for_eval: str,
        summarization_for_eval: str
    ) -> Tuple[str, float, List[int], Dict[str, int]]:
        """Simplified search method with better structure."""
        self.reset_cache()
        
        # Initialize search state
        search_state = {
            "best_score": float('inf'),
            "best_code": original_code_param,
            "best_topo_sort": None,
            "all_scores_valid": [],
            "stats": self._initialize_stats()
        }
        
        # Setup base structures
        _, base_graph, base_ast_tree, base_nx_graph = self.get_reordered_code(original_code_param)
        if not all([base_graph, base_ast_tree, base_nx_graph]):
            self.logger.warning("Failed to initialize base structures")
            return original_code_param, float('inf'), [], search_state["stats"]
        
        # Phase 1: Initial sampling
        self._phase1_initial_sampling(search_state, base_nx_graph, base_graph, base_ast_tree, 
                                     original_code_param, source_code_for_eval, summarization_for_eval)
        
        # Phase 2: Neighborhood search
        if search_state["best_topo_sort"] is not None:
            self._phase2_neighborhood_search(search_state, base_nx_graph, base_graph, base_ast_tree,
                                           original_code_param, source_code_for_eval, summarization_for_eval)
        
        return (search_state["best_code"], search_state["best_score"], 
                search_state["all_scores_valid"], search_state["stats"])

    def _initialize_stats(self) -> Dict[str, int]:
        """Initialize statistics dictionary."""
        return {
            "total_reordering_attempts": 0,
            "unique_reorderings_generated": 0,
            "failed_reorderings": 0,
            "transform_to_incomplete_failed": 0,
            "evaluation_attempts_for_new_reorderings": 0,
            "successful_llm_evaluations": 0,
            "failed_llm_evaluations": 0,
            "cache_hits_reordering_score": 0,
            "cache_hits_reordering_failed_eval": 0,
            "total_topo_sorts_generated": 0,
            "unique_perturbations": 0
        }

    def _phase1_initial_sampling(self, search_state, base_nx_graph, base_graph, base_ast_tree,
                                original_code_param, source_code_for_eval, summarization_for_eval):
        """Phase 1: Initial random sampling with batched evaluation."""
        self.logger.info("Phase 1: Initial random sampling (batched)...")
        
        # Generate topological sorts
        topo_sorts = []
        for _ in range(self.initial_samples):
            search_state["stats"]["total_topo_sorts_generated"] += 1
            topo_sorts.append(non_deterministic_topological_sort(base_nx_graph))
        
        # Process and evaluate
        self._process_topo_sorts_batch(topo_sorts, search_state, base_graph, base_ast_tree,
                                      original_code_param, source_code_for_eval, summarization_for_eval)

    def _phase2_neighborhood_search(self, search_state, base_nx_graph, base_graph, base_ast_tree,
                                   original_code_param, source_code_for_eval, summarization_for_eval):
        """Phase 2: Neighborhood search with adaptive change probability."""
        self.logger.info("Phase 2: Neighborhood search (batched)...")
        change_prob = self.change_probability
        
        for iteration in range(self.iterations):
            self.logger.info(f"Iteration {iteration+1}/{self.iterations}, change probability: {change_prob:.2f}")
            
            # Generate neighbor sorts
            neighbor_sorts = []
            for _ in range(self.neighborhood_samples):
                search_state["stats"]["total_topo_sorts_generated"] += 1
                neighbor_sorts.append(
                    neighborhood_topological_sort(base_nx_graph, search_state["best_topo_sort"], change_prob)
                )
            
            # Process and evaluate
            improved = self._process_topo_sorts_batch(
                neighbor_sorts, search_state, base_graph, base_ast_tree,
                original_code_param, source_code_for_eval, summarization_for_eval
            )
            
            # Adaptive change probability
            if not improved:
                change_prob = min(0.8, change_prob * 1.5)
            else:
                change_prob = max(0.1, change_prob * 0.9)

    def _process_topo_sorts_batch(self, topo_sorts, search_state, base_graph, base_ast_tree,
                                 original_code_param, source_code_for_eval, summarization_for_eval):
        """Process a batch of topological sorts and update search state."""
        batch_data = []
        unique_hashes = set()
        
        for topo_sort in topo_sorts:
            search_state["stats"]["total_reordering_attempts"] += 1
            reordered_code, _, _, _ = self.get_reordered_code(
                original_code_param, topo_sort, base_graph, base_ast_tree
            )
            
            if reordered_code is None or reordered_code == original_code_param:
                search_state["stats"]["failed_reorderings"] += 1
                continue
            
            code_hash = hashlib.md5(reordered_code.encode()).hexdigest()[:16]
            
            if code_hash not in unique_hashes:
                unique_hashes.add(code_hash)
                search_state["stats"]["unique_reorderings_generated"] += 1
            
            # Check cache
            if code_hash in self.results_cache:
                self._handle_cached_result(code_hash, topo_sort, search_state)
                continue
            
            # Prepare for evaluation
            incomplete_code = incomplete_code_transform(reordered_code)
            if incomplete_code is None:
                search_state["stats"]["transform_to_incomplete_failed"] += 1
                self.results_cache[code_hash] = (None, reordered_code)
                continue
            
            batch_data.append({
                "incomplete_code": incomplete_code,
                "reordered_code": reordered_code,
                "topo_sort": topo_sort,
                "code_hash": code_hash
            })
        
        # Batch evaluation
        if batch_data:
            self._evaluate_batch(batch_data, search_state, source_code_for_eval, summarization_for_eval)
        
        search_state["stats"]["unique_perturbations"] = len(unique_hashes)
        return any(item.get("improved", False) for item in batch_data)

    def _handle_cached_result(self, code_hash, topo_sort, search_state):
        """Handle cached evaluation result."""
        cached_score, cached_code = self.results_cache[code_hash]
        if cached_score is not None:
            search_state["stats"]["cache_hits_reordering_score"] += 1
            score_int = int(cached_score)
            search_state["all_scores_valid"].append(score_int)
            if score_int < search_state["best_score"]:
                search_state["best_score"] = score_int
                search_state["best_code"] = cached_code
                search_state["best_topo_sort"] = topo_sort
        else:
            search_state["stats"]["cache_hits_reordering_failed_eval"] += 1

    def _evaluate_batch(self, batch_data, search_state, source_code_for_eval, summarization_for_eval):
        """Evaluate a batch of reordered codes."""
        search_state["stats"]["evaluation_attempts_for_new_reorderings"] += len(batch_data)
        
        incomplete_codes = [item["incomplete_code"] for item in batch_data]
        source_codes = [source_code_for_eval] * len(batch_data)
        summarizations = [summarization_for_eval] * len(batch_data)
        
        scores = self.evaluation_function_batch(incomplete_codes, source_codes, summarizations)
        
        for item, score in zip(batch_data, scores):
            self.results_cache[item["code_hash"]] = (score, item["reordered_code"])
            
            if score is None:
                search_state["stats"]["failed_llm_evaluations"] += 1
                continue
            
            search_state["stats"]["successful_llm_evaluations"] += 1
            search_state["all_scores_valid"].append(score)
            
            if score < search_state["best_score"]:
                search_state["best_score"] = score
                search_state["best_code"] = item["reordered_code"]
                search_state["best_topo_sort"] = item["topo_sort"]
                item["improved"] = True
                self.logger.info(f"New best adversarial score: {score}")