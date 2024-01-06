import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import json


dataset = r"Mock-Hackathon\Input data\level1a.json"

data = pd.read_json(dataset)
neighbourhoods_df = pd.DataFrame(data['neighbourhoods'])
restaurants_df = pd.DataFrame(data['restaurants'])
vechiles_df = pd.DataFrame(data['vehicles'])


neighbourhoods_df.dropna(inplace=True)
restaurants_df.dropna(inplace=True)
vechiles_df.dropna(inplace=True)

neighbours = []
restaurant_distances = []
vechile_capacity = 0

for index, row in neighbourhoods_df.iterrows():
    neighbours.append(row['neighbourhoods']['distances'])
for index, row in restaurants_df.iterrows():
    restaurant_distances.extend(row['restaurants']['neighbourhood_distance'])
for index, row in vechiles_df.iterrows():
    vechile_capacity = row['vehicles']['capacity']

def opt_path():
    G = nx.Graph()
    restaurant_node = 'r0'
    G.add_node(restaurant_node)
    names = ['n' + str(i) for i in range(len(neighbours))]

    num_houses = len(neighbours)
    G.add_nodes_from(names)

    # Connect the restaurant node to all neighborhood house nodes
    for i in range(num_houses):
        G.add_edge(restaurant_node, i, weight=restaurant_distances[i])

    # Connect houses with correct weights
    for i in range(num_houses):
        for j in range(i + 1, num_houses):
            G.add_edge(i, j, weight=neighbours[i][j])

    start_node = restaurant_node
    nodes_ordered = [start_node]+[node for node in G.nodes if node != start_node] 
    

    nodes_ordered = list(nx.dfs_preorder_nodes(G, source=start_node))
    G_reordered = G.subgraph(nodes_ordered)
    tsp_path = nx.approximation.traveling_salesman_problem(G_reordered)


    #print("TSP Path:", tsp_path)
    total_cost = sum(G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    print("Total Cost:", total_cost)

    return tsp_path
#--------------------------------------------------------------------------------------------------------------

tsp_path = opt_path()
print(tsp_path)


