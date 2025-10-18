#!/usr/bin/env python3
"""
Structural Causal Models (SCMs) and Causal DAGs

Implements Pearl's structural causal model framework using directed acyclic graphs (DAGs).

Key concepts:
- DAG: Directed acyclic graph representing causal relationships
- Backdoor criterion: Identifying confounders to adjust for valid causal inference
- Front-door criterion: Alternative identification when backdoor is blocked
- d-separation: Testing conditional independence in DAGs
- do-calculus: Computing interventional distributions

Applications in NFL:
- Model player performance dependencies (skill → yards, weather → yards)
- Identify confounding paths (team strength affects both player usage and outcomes)
- Design valid causal queries with minimal adjustment sets
"""

import logging
from collections import defaultdict, deque
from itertools import combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalDAG:
    """
    Represents a causal directed acyclic graph (DAG).

    Nodes represent variables, directed edges represent causal relationships.
    """

    def __init__(self):
        """Initialize empty DAG"""
        self.nodes: set[str] = set()
        self.edges: dict[str, set[str]] = defaultdict(set)  # node -> children
        self.parents: dict[str, set[str]] = defaultdict(set)  # node -> parents

    def add_node(self, node: str):
        """Add a node to the DAG"""
        self.nodes.add(node)

    def add_edge(self, from_node: str, to_node: str):
        """
        Add directed edge from_node -> to_node.

        Raises ValueError if this creates a cycle.
        """
        # Add nodes if they don't exist
        self.add_node(from_node)
        self.add_node(to_node)

        # Check for cycles
        if self._would_create_cycle(from_node, to_node):
            raise ValueError(f"Adding edge {from_node} -> {to_node} would create a cycle")

        self.edges[from_node].add(to_node)
        self.parents[to_node].add(from_node)

    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """Check if adding edge would create cycle using BFS"""
        if from_node == to_node:
            return True

        # Check if there's a path from to_node to from_node
        visited = set()
        queue = deque([to_node])

        while queue:
            current = queue.popleft()
            if current == from_node:
                return True

            if current in visited:
                continue

            visited.add(current)

            # Add children to queue
            for child in self.edges.get(current, set()):
                if child not in visited:
                    queue.append(child)

        return False

    def get_children(self, node: str) -> set[str]:
        """Get immediate children of node"""
        return self.edges.get(node, set())

    def get_parents(self, node: str) -> set[str]:
        """Get immediate parents of node"""
        return self.parents.get(node, set())

    def get_ancestors(self, node: str) -> set[str]:
        """Get all ancestors of node (parents, grandparents, etc.)"""
        ancestors = set()
        to_visit = list(self.parents.get(node, set()))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.parents.get(current, set()))

        return ancestors

    def get_descendants(self, node: str) -> set[str]:
        """Get all descendants of node (children, grandchildren, etc.)"""
        descendants = set()
        to_visit = list(self.edges.get(node, set()))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.edges.get(current, set()))

        return descendants

    def is_d_separated(self, X: set[str], Y: set[str], Z: set[str]) -> bool:
        """
        Test if X and Y are d-separated given Z.

        d-separation implies conditional independence: X ⊥ Y | Z

        Args:
            X: Set of source nodes
            Y: Set of target nodes
            Z: Set of conditioning nodes

        Returns:
            True if X and Y are d-separated given Z
        """
        # For each pair in X × Y, check if there's an active path
        for x in X:
            for y in Y:
                if self._has_active_path(x, y, Z):
                    return False

        return True

    def _has_active_path(self, start: str, end: str, Z: set[str]) -> bool:
        """
        Check if there's an active (unblocked) path from start to end given Z.

        A path is blocked if:
        1. Contains chain i → m → j where m ∈ Z
        2. Contains fork i ← m → j where m ∈ Z
        3. Contains collider i → m ← j where m ∉ Z and no descendant of m is in Z
        """
        # BFS with path tracking
        # State: (current_node, previous_node, direction)
        # direction: 'up' if came from child, 'down' if came from parent
        queue = deque()

        # Initialize: start from children and parents
        for child in self.get_children(start):
            queue.append((child, start, "down"))
        for parent in self.get_parents(start):
            queue.append((parent, start, "up"))

        visited = set()

        while queue:
            current, previous, direction = queue.popleft()

            if current == end:
                return True

            state = (current, previous, direction)
            if state in visited:
                continue
            visited.add(state)

            # Check if path is blocked at current node
            parents = self.get_parents(current)
            children = self.get_children(current)

            # Collider: → current ←
            is_collider = previous in parents

            if is_collider:
                # Collider is active only if current or descendant is in Z
                descendants = self.get_descendants(current)
                if current not in Z and not any(d in Z for d in descendants):
                    continue  # Path blocked

            else:
                # Chain or fork: current is in the middle
                if current in Z:
                    continue  # Path blocked

            # Add next nodes to queue
            if direction == "down":
                # Coming from parent, can go to children or other parents
                for child in children:
                    if child != previous:
                        queue.append((child, current, "down"))

                # Can go up through collider
                if is_collider:
                    for parent in parents:
                        if parent != previous:
                            queue.append((parent, current, "up"))

            else:  # direction == 'up'
                # Coming from child, can go to parents
                for parent in parents:
                    if parent != previous:
                        queue.append((parent, current, "up"))

                # Can go down if not collider or collider is activated
                if (
                    not is_collider
                    or current in Z
                    or any(d in Z for d in self.get_descendants(current))
                ):
                    for child in children:
                        if child != previous:
                            queue.append((child, current, "down"))

        return False

    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> set[str] | None:
        """
        Find minimal set satisfying backdoor criterion.

        Backdoor criterion (Pearl):
        A set Z satisfies the backdoor criterion if:
        1. No node in Z is a descendant of treatment
        2. Z blocks all backdoor paths from treatment to outcome

        Backdoor path: Any path from treatment to outcome that has an arrow INTO treatment

        Args:
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Minimal adjustment set, or None if no valid set exists
        """
        # Find all backdoor paths
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            # No backdoor paths, no adjustment needed
            logger.info(f"No backdoor paths from {treatment} to {outcome}, no adjustment needed")
            return set()

        # Candidate adjustment nodes: non-descendants of treatment, not treatment or outcome
        treatment_descendants = self.get_descendants(treatment)
        candidates = self.nodes - treatment_descendants - {treatment, outcome}

        if not candidates:
            logger.warning(f"No valid adjustment set exists for {treatment} -> {outcome}")
            return None

        # Find minimal set that blocks all backdoor paths
        # Try sets of increasing size
        for size in range(1, len(candidates) + 1):
            for adjustment_set in combinations(candidates, size):
                adjustment_set = set(adjustment_set)

                # Check if this set blocks all backdoor paths
                if self._blocks_all_paths(backdoor_paths, adjustment_set):
                    logger.info(f"Found minimal backdoor adjustment set: {adjustment_set}")
                    return adjustment_set

        logger.warning(f"No valid backdoor adjustment set found for {treatment} -> {outcome}")
        return None

    def _find_backdoor_paths(self, treatment: str, outcome: str) -> list[list[str]]:
        """Find all backdoor paths from treatment to outcome"""
        backdoor_paths = []

        # Start from parents of treatment (paths going backwards into treatment)
        parents = self.get_parents(treatment)

        for parent in parents:
            # Find all paths from parent to outcome
            paths = self._find_all_paths(parent, outcome, avoid={treatment})

            # Add treatment at beginning of each path
            for path in paths:
                backdoor_paths.append([treatment] + path)

        return backdoor_paths

    def _find_all_paths(
        self, start: str, end: str, avoid: set[str] | None = None
    ) -> list[list[str]]:
        """Find all paths from start to end (undirected)"""
        if avoid is None:
            avoid = set()

        paths = []
        stack = [(start, [start])]
        visited_paths = set()

        while stack:
            current, path = stack.pop()

            if current == end:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    paths.append(path)
                    visited_paths.add(path_tuple)
                continue

            # Explore children (forward)
            for child in self.get_children(current):
                if child not in path and child not in avoid:
                    stack.append((child, path + [child]))

            # Explore parents (backward)
            for parent in self.get_parents(current):
                if parent not in path and parent not in avoid:
                    stack.append((parent, path + [parent]))

        return paths

    def _blocks_all_paths(self, paths: list[list[str]], adjustment_set: set[str]) -> bool:
        """Check if adjustment set blocks all paths"""
        for path in paths:
            if not self._blocks_path(path, adjustment_set):
                return False
        return True

    def _blocks_path(self, path: list[str], adjustment_set: set[str]) -> bool:
        """Check if adjustment set blocks a single path"""
        # Path is blocked if any non-collider node is in adjustment set
        # or if all collider nodes are outside adjustment set

        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            # Check if node is a collider
            is_collider = next_node in self.get_parents(node) and prev_node in self.get_parents(
                node
            )

            if is_collider:
                # Collider: path is open if node or descendant is in adjustment set
                descendants = self.get_descendants(node)
                if node in adjustment_set or any(d in adjustment_set for d in descendants):
                    return False  # Path is open (not blocked)
            else:
                # Chain or fork: path is blocked if node is in adjustment set
                if node in adjustment_set:
                    return True  # Path is blocked

        return False  # Path is not blocked

    def visualize(self) -> str:
        """
        Generate simple text representation of DAG.

        Returns:
            String representation
        """
        lines = ["Causal DAG:", "-" * 40]

        for node in sorted(self.nodes):
            children = sorted(self.get_children(node))
            if children:
                lines.append(f"{node} → {', '.join(children)}")

        return "\n".join(lines)


def create_nfl_rushing_dag() -> CausalDAG:
    """
    Create example causal DAG for NFL rushing performance.

    Variables:
    - player_ability: Unobserved player skill
    - team_quality: Team strength
    - opponent_defense: Opponent's defensive quality
    - game_script: Winning/losing situation
    - carries: Number of rushing attempts
    - rushing_yards: Outcome of interest
    """
    dag = CausalDAG()

    # Unobserved confounder
    dag.add_edge("player_ability", "carries")
    dag.add_edge("player_ability", "rushing_yards")

    # Team context
    dag.add_edge("team_quality", "carries")
    dag.add_edge("team_quality", "game_script")
    dag.add_edge("team_quality", "rushing_yards")

    # Opponent
    dag.add_edge("opponent_defense", "carries")
    dag.add_edge("opponent_defense", "rushing_yards")

    # Game script affects usage
    dag.add_edge("game_script", "carries")

    # Treatment -> Outcome
    dag.add_edge("carries", "rushing_yards")

    logger.info("Created NFL rushing performance DAG")

    return dag


def create_coaching_change_dag() -> CausalDAG:
    """
    Create causal DAG for coaching change effects.

    Variables:
    - team_resources: Owner investment
    - prior_performance: Past record
    - coaching_change: Treatment
    - player_morale: Mediator
    - new_performance: Outcome
    """
    dag = CausalDAG()

    # Confounders
    dag.add_edge("team_resources", "coaching_change")
    dag.add_edge("team_resources", "new_performance")

    dag.add_edge("prior_performance", "coaching_change")
    dag.add_edge("prior_performance", "new_performance")

    # Treatment -> Mediator -> Outcome
    dag.add_edge("coaching_change", "player_morale")
    dag.add_edge("player_morale", "new_performance")

    # Direct effect
    dag.add_edge("coaching_change", "new_performance")

    logger.info("Created coaching change DAG")

    return dag


def main():
    """Example usage of causal DAGs"""

    print("\n" + "=" * 80)
    print("STRUCTURAL CAUSAL MODEL - NFL RUSHING EXAMPLE")
    print("=" * 80)

    # Create DAG
    dag = create_nfl_rushing_dag()

    print("\n" + dag.visualize())

    # Find backdoor adjustment set
    print("\n" + "=" * 80)
    print("BACKDOOR CRITERION")
    print("=" * 80)

    treatment = "carries"
    outcome = "rushing_yards"

    adjustment_set = dag.find_backdoor_adjustment_set(treatment, outcome)

    if adjustment_set:
        print(f"\nTo estimate causal effect of {treatment} on {outcome}:")
        print(f"Adjust for: {adjustment_set}")
    else:
        print("\nNo valid adjustment set found")

    # Test d-separation
    print("\n" + "=" * 80)
    print("CONDITIONAL INDEPENDENCE (d-separation)")
    print("=" * 80)

    # Test: carries ⊥ opponent_defense | {team_quality, player_ability}?
    X = {"carries"}
    Y = {"opponent_defense"}
    Z = {"team_quality", "player_ability"}

    is_independent = dag.is_d_separated(X, Y, Z)
    print("\ncarries ⊥ opponent_defense | {team_quality, player_ability}?")
    print(f"Result: {is_independent}")

    # Coaching change example
    print("\n" + "=" * 80)
    print("COACHING CHANGE CAUSAL MODEL")
    print("=" * 80)

    dag2 = create_coaching_change_dag()
    print("\n" + dag2.visualize())

    adjustment_set2 = dag2.find_backdoor_adjustment_set("coaching_change", "new_performance")

    if adjustment_set2:
        print("\nTo estimate coaching change effect:")
        print(f"Adjust for: {adjustment_set2}")

        print("\nInterpretation:")
        print("Conditioning on team_resources and prior_performance blocks backdoor")
        print("confounding paths, enabling valid causal inference.")

    # Test mediation
    print("\n" + "=" * 80)
    print("MEDIATION ANALYSIS")
    print("=" * 80)

    # Direct effect: coaching_change → new_performance
    # Indirect effect: coaching_change → player_morale → new_performance

    print("\nMediator: player_morale")
    print("- Direct effect: coaching_change → new_performance")
    print("- Indirect effect: coaching_change → player_morale → new_performance")
    print("\nTotal effect = Direct effect + Indirect effect")


if __name__ == "__main__":
    main()
