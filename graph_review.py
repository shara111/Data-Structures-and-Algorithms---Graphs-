class Vertex:
    def __init__(self, data):
        self.data = data
        self.adjacancies = []
    
    def add_adjacency(self, other, weight):
        if other not in self.adjacancies:
            self.adjacancies.append((other, weight))
    
class Graph:
    #Initialize an empty graph
    def __init__(self):
        self.verticies = []
    
    def add_edge(self, u, v, weight):
        if u not in self.verticies:
            self.verticies.append(u)
        if v not in self.verticies:
            self.verticies.append(v)
        
        #Since this is an undirected graph, both edges will be added
        u.add_adjacency(v, weight)
        v.add_adjacency(u, weight)
# Create vertices
v1 = Vertex("A")
v2 = Vertex("B")

# Create a graph
g = Graph()

# Add an edge between v1 (A) and v2 (B) with weight 3
g.add_edge(v1, v2, 3)