import pandas as pd 

class Node:
    def __init__(self, name: str, df=None):
        self.name = name
        self.df = df
        
    def getName(self):
        return self.name
        
    def getData(self):
        return self.df
    
    def __str__(self):
        return f"Node({self.name})"
    
    def __repr__(self):
        return self.__str__()

class Edge:
    def __init__(self, node1, node2, name, quant, directed=False):
        self._node1 = node1
        self._node2 = node2
        self._name = name
        self._quant = quant
        self._directed = directed  # True for directed (Ligand-Receptor), False for undirected (adhesion-adhesion)
    
    @property
    def node1(self):
        return self._node1
    
    @property
    def node2(self):
        return self._node2
    
    @property
    def name(self):
        return self._name
    
    @property
    def quant(self):
        return self._quant
    
    @property
    def directed(self):
        return self._directed
    
    @property
    def weight(self):
        return (self._name, self._quant)
    
    def __str__(self):
        if self._directed:
            return f"Edge({self.node1.getName()} → {self.node2.getName()}, {self.name}({self.quant}))"
        else:
            return f"Edge({self.node1.getName()} ↔ {self.node2.getName()}, {self.name}({self.quant}))"
    
    def __repr__(self):
        return self.__str__()

class Graph:
    def __init__(self):
        self._nodes = []
        self._edges = []
        
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def edges(self):
        return self._edges
    
    def add_node(self, name: str, df=None):
        for node in self._nodes:
            if node.getName() == name:
                return node
        node = Node(name, df)
        self._nodes.append(node)
        return node
    
    def add_edge(self, node1, node2, name, quant, directed=False):
        """
        Create an edge between node1 and node2.
        :param directed: If True, the edge is directional (node1 → node2).
                         Otherwise, it is undirected (node1 ↔ node2).
        """
        edge = Edge(node1, node2, name, quant, directed)
        self._edges.append(edge)
        return edge
    
    def get_node_by_name(self, name):
        for node in self._nodes:
            if node.getName() == name:
                return node
        return None
    
    def get_edges_between(self, node1, node2, directed_only=False):
        """
        Get all edges connecting node1 and node2.
        If directed_only is True, only return edges where node1 -> node2.
        """
        if directed_only:
            return [edge for edge in self._edges 
                    if edge.directed and edge.node1 == node1 and edge.node2 == node2]
        else:
            return [edge for edge in self._edges 
                    if (edge.node1 == node1 and edge.node2 == node2) or 
                       (edge.node1 == node2 and edge.node2 == node1)]
    
    def get_out_neighbors(self, node):
        """
        For directed edges, return all nodes that node has an outgoing edge to.
        For undirected edges, the connection is symmetric.
        """
        neighbors = set()
        for edge in self._edges:
            if edge.directed:
                if edge.node1 == node:
                    neighbors.add(edge.node2)
            else:
                if edge.node1 == node:
                    neighbors.add(edge.node2)
                elif edge.node2 == node:
                    neighbors.add(edge.node1)
        return list(neighbors)
    
    def get_in_neighbors(self, node):
        """
        For directed edges, return all nodes that have an outgoing edge to node.
        For undirected edges, the connection is symmetric.
        """
        neighbors = set()
        for edge in self._edges:
            if edge.directed:
                if edge.node2 == node:
                    neighbors.add(edge.node1)
            else:
                if edge.node1 == node:
                    neighbors.add(edge.node2)
                elif edge.node2 == node:
                    neighbors.add(edge.node1)
        return list(neighbors)
    
    def __str__(self):
        return f"Graph with {len(self._nodes)} nodes and {len(self._edges)} edges"