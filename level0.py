#"Level 0"
import pandas as pd
import networkx as nx


dataset = r"Mock-Hackathon\Input data\level0.json"

data = pd.read_json(dataset)
neighbourhoods_df = pd.DataFrame(data['neighbourhoods'])
restaurants_df = pd.DataFrame(data['restaurants'])

neighbourhoods_df.dropna(inplace=True)
restaurants_df.dropna(inplace=True)

neighbours = []
restaurant_distances = []

for index, row in neighbourhoods_df.iterrows():
    neighbours.append(row['neighbourhoods']['distances'])
for index, row in restaurants_df.iterrows():
    restaurant_distances.extend(row['restaurants']['neighbourhood_distance'])

#naming for mapping it into list
names = ['n' + str(i) for i in range(len(neighbours))]

G = nx.Graph()
num_houses = len(neighbours)
G.add_nodes_from(range(num_houses))

restaurant_node = 'restaurant'
G.add_node(restaurant_node)


# Connect the restaurant node to all neighborhood house nodes
for i in range(num_houses):
    G.add_edge(restaurant_node, i, weight=restaurant_distances[i])

for i in range(num_houses):
    for j in range(i + 1, num_houses):
        G.add_edge(i, j, weight=neighbours[i][j])

start_node = restaurant_node

# Manually reorder nodes based on the start node
nodes_ordered = [start_node] + [node for node in G.nodes if node != start_node]
print(nodes_ordered)
subgraph = G.subgraph(nodes_ordered)


tsp_path = nx.approximation.traveling_salesman_problem(subgraph, weight='weight')


print("TSP Path:", tsp_path)

import matplotlib.pyplot as plt

subgraph = G.subgraph([restaurant_node] + list(G.neighbors(restaurant_node)))
pos = nx.spring_layout(subgraph)  
nx.draw(subgraph, pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(subgraph, 'weight')
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=labels)
plt.show()

print()






