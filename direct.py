#contains the implementation of Directed graphs

graph = {
    'A': ['B', 'C'],  # A points to B and C
    'B': ['D'],       # B points to D
    'C': ['D'],       # C points to D
    'D': []           # D points to no one
}
#to add a new node
graph['E'] = ['C', 'C']

#add an edge
graph['A'].append('E') 

#remove a node
del graph['E']

#remove an edge
graph['A'].remove('E')

for node, neighbors in graph.items():
    print(f'${node} : {neighbors}');