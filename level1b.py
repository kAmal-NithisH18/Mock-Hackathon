import heapq
import json
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming

def optimize_delivery_slots(orders, scooter_capacity):
    orders.sort(key=lambda x: x['quantity'], reverse=True)

    delivery_slots = []
    total_distance = 0
    path_locations = []

    for order in orders:
        quantity = order['quantity']
        location = order['location']

        if delivery_slots and delivery_slots[0][0] + quantity <= scooter_capacity:
            current_slot = heapq.heappop(delivery_slots)
            current_slot[0] += quantity
            current_slot[1].append(location)  # Append location to the path
            heapq.heappush(delivery_slots, current_slot)
        else:
            heapq.heappush(delivery_slots, [quantity, [location]])  # Start a new delivery slot

    # Extract locations from each path
    for slot in delivery_slots:
        path_locations.append(slot[1])


    return path_locations, total_distance, delivery_slots

# Example orders
json_file_path = 'Mock-Hackathon\Input data\level1b.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

name = ['n'+str(i) for i in range(50)]
print(name)

orders = [{'quantity': data['neighbourhoods'][name[i]]['order_quantity'], 'location': i} for i in range(len(name))]

scooter_capacity = data['vehicles']['v0']['capacity']

optimized_paths, total_distance, slots = optimize_delivery_slots(orders, scooter_capacity)

print("Optimized Delivery Paths:", optimized_paths)
print("Total Distance Traveled:", total_distance)


for i in optimized_paths:
    i.insert(0, -1)


tsp_g = []

json_file_path = 'Mock-Hackathon\Input data\level1a.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

r = data['restaurants']['r0']['neighbourhood_distance']

for i in range(len(name)+1):
    tsp_g.append([0])

for i in range(len(r)):
    tsp_g[0].append(r[i])

for i in range(1, len(name)+1):
    d = data['neighbourhoods'][name[i-1]]['distances']
    for j in d:
        tsp_g[i].append(j)

for i in range(1, 50):
    tsp_g[i][0] = r[i-1]

slo = []
for i in range(len(slots)):
    for j in optimized_paths:
        temp = []
        for loc in range(len(j)):
            s = []
            for loc_1 in range(len(j)):
                s.append(tsp_g[j[loc]+1][j[loc_1]+1])
            temp.append(s)
        slo.append(np.array(temp))
    break

min_slot_dis = []

print(slo[0])

for i in range(len(slo)):
    permutation, distance = solve_tsp_dynamic_programming(slo[i])
    min_slot_dis.append([permutation,distance])

print(min_slot_dis)

print(min_slot_dis[0][1] + min_slot_dis[1][1] + min_slot_dis[2][1])

d = {'v0': {}}

m = {-1: 'r0'}
for i in len(name):
    m[i] = name[i]


for i in range(len(min_slot_dis)):
    w = []
    for j in min_slot_dis[i][0]:
        q = optimized_paths[i][j]
        w.append(m[q])
    w.append('r0')
    st = 'path'+str(i+1)        
    d['v0'][st] = w

print(d)

json_file_path = 'level1a_output.json'

with open(json_file_path, 'w') as json_file:
    json.dump(d, json_file, indent=2)