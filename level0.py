#"Level 0"
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import json


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




#--------------------------------------------------------------------------------------------


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
#print(nodes_ordered)

nodes_ordered = list(nx.dfs_preorder_nodes(G, source=start_node))
G_reordered = G.subgraph(nodes_ordered)
tsp_path = nx.approximation.traveling_salesman_problem(G_reordered)


print("TSP Path:", tsp_path)
total_cost = sum(G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
print("Total Cost:", total_cost)

def visual():
    subgraph = G.subgraph([restaurant_node] + list(G.neighbors(restaurant_node)))
    pos = nx.spring_layout(subgraph)  
    nx.draw(subgraph, pos, with_labels=True, font_weight='bold')
    labels = nx.get_edge_attributes(subgraph, 'weight')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=labels)
    edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)]
    #plt.savefig("fig1.png")
    plt.show()

    nx.draw_networkx_edges(G_reordered, pos, edgelist=edges, edge_color='r', width=2)
    #plt.savefig("fig2.png")
    plt.show()

    subgraph = G.subgraph([restaurant_node] + list(G.neighbors(restaurant_node)))
    pos = nx.spring_layout(subgraph)  
    nx.draw(subgraph, pos, with_labels=True, font_weight='bold')
    labels = nx.get_edge_attributes(subgraph, 'weight')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=labels)
    edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)]
    nx.draw_networkx_edges(G_reordered, pos, edgelist=edges, edge_color='r', width=2)
    #plt.savefig("fig3.png")
    plt.show()





greedy_tsp_path = nx.approximation.greedy_tsp(G_reordered, weight='weight', source=start_node)
# Calculate total cost
#print(greedy_tsp_path)
total_cost_greedy_tsp = sum(G_reordered[greedy_tsp_path[i]][greedy_tsp_path[i + 1]]['weight'] for i in range(len(greedy_tsp_path) - 1))


#print("Christofides")
greedy_tsp_path = nx.approximation.christofides(G_reordered, weight='weight', tree=None)
total_cost_greedy_tsp = sum(G_reordered[greedy_tsp_path[i]][greedy_tsp_path[i + 1]]['weight'] for i in range(len(greedy_tsp_path) - 1))




#---------------------------------------------------------------------------------------------------------------
#Output file
visual()

def total_cost(graph, path):
    return sum(graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

def two_opt(graph, path):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1:
                    continue # Changes nothing, skip then
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                if total_cost(graph, new_path) < total_cost(graph, path):
                    path = new_path
                    improved = True
        return path

# Assuming tsp_path is the initial solution obtained
improved_path = two_opt(G_reordered, tsp_path)

print("Improved TSP Path:", improved_path)
print("Improved Total Cost:", total_cost(G_reordered, improved_path))

tsp_path = improved_path

out = [tsp_path[0]]
for i in tsp_path[1:-1]:
    if(type(i) != 'str'):
        out.append("n"+str(i))
out.append(tsp_path[-1])

output_file_path = "level0_output.json"
output_data = {
    "v0": {"path": out}
}
with open(output_file_path, 'w') as json_file:
    json.dump(output_data, json_file)