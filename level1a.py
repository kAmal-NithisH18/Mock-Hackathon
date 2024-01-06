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
orders.insert(0,0)

"""Capacited Vehicles Routing Problem (CVRP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = neighbours
    data["demands"] = orders
    data["vehicle_capacities"] = [600]
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(100)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)








def create_data_model(neighbours, restaurant_capacity):
    data = {}
    data['distance_matrix'] = neighbours
    data['num_vehicles'] = 1
    data['depot'] = 0 # The restaurant is the depot
    data['demands'] = orders # Demand includes the restaurant with 0 demand
    data['vehicle_capacities'] = [restaurant_capacity]

    return data

def main1(neighbours, restaurant_capacity):
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
        0, # null capacity slack
        data['vehicle_capacities'], # vehicle maximum capacities
        True, # start cumul to zero
        'Capacity')

    # Set 10 seconds as the maximum time allowed per each vehicle.
    routing.SetFixedCostOfAllVehicles(10)

    # Solve the problem.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)

def print_solution(manager, routing, solution):
    print('Objective: {}'.format(solution.ObjectiveValue()))
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

# Sample data
restaurant_capacity = 600

main1(neighbours, restaurant_capacity)