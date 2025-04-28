from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt

def _trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def DrawGraph(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})  # Left to right

    nodes, edges = _trace(root)
    for n in nodes:
        uid = str(id(n))
        # Add node: show data and grad
        dot.node(name=uid, label=repr(n), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def DrawGraph2(output_node):
    G = nx.DiGraph()
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            label = repr(v)
            G.add_node(v, label=label)
            for child in v._prev:
                G.add_edge(child, v)
                build(child)

    build(output_node)

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color="lightblue", font_size=8)
    plt.title("Computation Graph")
    plt.show()

