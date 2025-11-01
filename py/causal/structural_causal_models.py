"""
Structural Causal Models (SCM)

Define and work with causal DAGs for identification.
"""

import networkx as nx
from typing import List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CausalDAG:
    """Directed Acyclic Graph for causal inference"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_edge(self, cause: str, effect: str):
        """Add causal edge"""
        self.graph.add_edge(cause, effect)

    def find_confounders(
        self,
        treatment: str,
        outcome: str
    ) -> Set[str]:
        """
        Find confounders using backdoor criterion.

        A confounder Z satisfies:
        1. Z is on a backdoor path from treatment to outcome
        2. Z blocks all backdoor paths when conditioned on
        """
        # Find all paths from treatment to outcome
        all_paths = list(nx.all_simple_paths(self.graph.to_undirected(), treatment, outcome))

        # Find backdoor paths (those that start with an arrow into treatment)
        backdoor_paths = []
        for path in all_paths:
            if len(path) > 2:
                # Check if first edge points INTO treatment
                if self.graph.has_edge(path[1], path[0]):
                    backdoor_paths.append(path)

        # Find nodes that appear in backdoor paths
        confounders = set()
        for path in backdoor_paths:
            # Exclude treatment and outcome
            confounders.update([n for n in path if n not in {treatment, outcome}])

        logger.info(f"Found {len(confounders)} potential confounders: {confounders}")

        return confounders

    def check_d_separation(
        self,
        x: str,
        y: str,
        z: Set[str]
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        Returns:
            True if X _||_ Y | Z
        """
        return nx.d_separated(self.graph, {x}, {y}, z)

    def find_instrumental_variables(
        self,
        treatment: str,
        outcome: str
    ) -> List[str]:
        """
        Find potential instrumental variables.

        An IV Z must:
        1. Affect treatment
        2. Not directly affect outcome
        3. Not share confounders with outcome
        """
        ivs = []

        for node in self.graph.nodes():
            if node in {treatment, outcome}:
                continue

            # Check if affects treatment
            if not self.graph.has_edge(node, treatment):
                continue

            # Check if does NOT directly affect outcome
            if self.graph.has_edge(node, outcome):
                continue

            # Check if d-separated from outcome given treatment
            if self.check_d_separation(node, outcome, {treatment}):
                ivs.append(node)

        logger.info(f"Found {len(ivs)} instrumental variables: {ivs}")

        return ivs
