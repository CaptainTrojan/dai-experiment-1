import networkx as nx
import matplotlib.pyplot as plt
from utils import KeypointGraph

# Define the edges
edges = KeypointGraph.POSE_PAIRS

# Define the labels
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
keypointsMapping = [f"{i}: {keypointsMapping[i]}" for i in range(len(keypointsMapping))]

# Create a graph
G = nx.Graph()

# Add edges to the graph
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G, k=0.2)  # positions for all nodes, k parameter for spacing

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, width=6)

# labels
nx.draw_networkx_labels(G, pos, dict(enumerate(keypointsMapping)), font_size=16, font_color='orange')

plt.axis('off')
plt.show()
