# Create a graph using a dictionary (Adjacency List)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C'],
}

# Print the graph
for node, neighbors in graph.items():
    print(f'{node}: {neighbors}')