#!/usr/bin/env python3
"""Quick debug script to test GNN graph building and expose the edge remapping bug."""

import sys
sys.path.insert(0, '/Users/dro/rice/nfl-analytics/py/models')

from gnn_graph_builder import NFLDatabase, NFLGraphBuilder

# Build a multi-season graph
db = NFLDatabase()
builder = NFLGraphBuilder(db)

print("Building multi-season graph for 2024-2025...")
graph = builder.build_multi_season_graph(
    seasons=[2024, 2025],
    include_chemistry=False,
    include_matchups=False
)

print("\nGraph structure:")
print(f"  Total nodes: {graph.num_nodes}")
print(f"  Player nodes: {len(graph.node_type_to_ids['player'])}")
print(f"  Position nodes: {len(graph.node_type_to_ids['position'])}")
print(f"  Team nodes: {len(graph.node_type_to_ids['team'])}")

print("\nNode type to IDs (first 10 of each):")
print(f"  Players: {graph.node_type_to_ids['player'][:10]}")
print(f"  Positions: {graph.node_type_to_ids['position']}")
print(f"  Teams: {graph.node_type_to_ids['team'][:10]}")

print("\nEdges:")
print(f"  Player→Position: {len(graph.edges['player_to_position'])} edges")
if graph.edges['player_to_position']:
    print(f"    Sample edges: {graph.edges['player_to_position'][:10]}")
    print(f"    Max src ID: {max(src for src, _ in graph.edges['player_to_position'])}")
    print(f"    Max dst ID: {max(dst for _, dst in graph.edges['player_to_position'])}")

print("\nPlayer ID map (first 10):")
items = list(graph.player_id_map.items())[:10]
for ext_id, int_id in items:
    print(f"  {ext_id} → {int_id}")

print("\nPosition ID map:")
for ext_id, int_id in graph.position_id_map.items():
    print(f"  {ext_id} → {int_id}")
