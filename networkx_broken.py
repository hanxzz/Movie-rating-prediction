import networkx as nx
import matplotlib.pyplot as plt

# Initialize graph
G = nx.Graph()

# Add nodes (characters)
characters = ['A', 'B', 'C']
G.add_nodes_from(characters)

# Add edges (interactions)
interactions = [('A', 'B', 10), ('A', 'C', 5), ('B', 'C', 7)]
G.add_weighted_edges_from(interactions)

# Draw the graph
pos = nx.spring_layout(G)  # Positioning the nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15)

# Add edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Character Interaction Network")
plt.show()
