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
orders = []

for index, row in neighbourhoods_df.iterrows():
    neighbours.append(row['neighbourhoods']['distances'])
    orders.append(row['neighbourhoods']['order_quantity'])
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
    global Adj 
    Adj = nx.adjacency_matrix(G_reordered)

    #print("TSP Path:", tsp_path)
    total_cost = sum(G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    print("Total Cost:", total_cost)

    return tsp_path
#--------------------------------------------------------------------------------------------------------------

tsp_path = opt_path()
print(tsp_path)




from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
def ort():
    def create_data_model(neighbours, restaurant_capacity):
        data = {}
        data['distance_matrix'] = neighbours
        data['num_vehicles'] = 1
        data['depot'] = 0  # Index of the restaurant in the distance matrix
        data['demands'] = [0] + [demand for demand in neighbours]  # 0 demand for the restaurant

        # The capacity of the restaurant's scooter
        data['vehicle_capacities'] = [restaurant_capacity]

        return data

    def main(neighbours, restaurant_capacity):
        data = create_data_model(neighbours, restaurant_capacity)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            return data['demands'][manager.IndexToNode(from_index)]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Set 10 seconds as the maximum time allowed per each vehicle.
        routing.SetFixedCostOfAllVehicles(100)

        # Solve the problem.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 100

        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            print_solution(manager, routing, solution)

    def print_solution(manager, routing, solution):

        index = routing.Start(0)
        plan_output = 'Routes for scooter:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += 'Distance of the route: {} units\n'.format(route_distance)
        print(plan_output)




    restaurant_capacity = 600

    main(neighbours, restaurant_capacity)



#neighbours
#capacity
#neighbour capcity

    
adj = list(Adj)
neighbours.insert(0,[0]+restaurant_distances)
for i in range(1,21):
    neighbours[i].insert(0,restaurant_distances[i-1])



enumerated_list = list(enumerate(restaurant_distances))
sorted_list = sorted(enumerated_list, key=lambda x: x[1])
sorted_values = [item[1] for item in sorted_list]
sorted_indices = [item[0] for item in sorted_list]


l = []
temp = 0
templ = []

for i in range(len(sorted_values)):
    
    if(orders[sorted_indices[i]] + temp <= 600):
        temp+=orders[sorted_indices[i]]
        templ.append([sorted_indices[i],restaurant_distances[sorted_indices[i]]])
    else:
        l.append(templ)
        templ = []
        temp = 0
l.append(templ)
out = []
t = 0
for i in l:
    g = list()
    for j in i:
        g.append(j[0])
    G = nx.Graph()
    restaurant_node = 'r0'
    G.add_node(restaurant_node)
    names = [g[i] for i in range(len(g))]
    out.append(["r0"]+names+["r0"])
    

    num_houses = len(g)
    G.add_nodes_from(names)

    # Connect the restaurant node to all neighborhood house nodes
    for i in range(num_houses):
        G.add_edge(restaurant_node, g[i], weight=restaurant_distances[g[i]])

    # Connect houses with correct weights
    for i in range(num_houses):
        for j in range(i + 1, num_houses):
            x=g[i]
            y = g[j]
            G.add_edge(x, y, weight=neighbours[x][y])
            
    tsp_path = nx.approximation.traveling_salesman_problem(G)
    #print("TSP Path:", tsp_path)
    total_cost = sum(G[tsp_path[i]][tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    t += total_cost
    

print(t)

for i in range(len(out)):
    for j in range(1,len(out[i])-1):
        out[i][j] = "n"+str(out[i][j])
print(out)

output_file_path = "level1_output.json"
output_data = {"v0": {"path1": out[0], "path2": out[1], "path3": out[2]}}

with open(output_file_path, 'w') as json_file:
    json.dump(output_data, json_file)