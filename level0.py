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

print(restaurant_distances)

#naming for mapping it into list
names = ['n' + str(i) for i in range(len(neighbours))]

G = nx.Graph()
num_houses = len(neighbours)
G.add_nodes_from(range(num_houses))

restaurant_node = 'restaurant'
G.add_node(restaurant_node)

# Connect the restaurant node to all neighborhood house nodes
"""for i in range(num_houses):
    G.add_edge(restaurant_node, i, weight=restaurant_distances[i])"""

for i in range(num_houses):
    for j in range(i + 1, num_houses):
        G.add_edge(i, j, weight=neighbours[i][j])







