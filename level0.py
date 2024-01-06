#"Level 0"
import pandas as pd
import json


dataset = r"Mock-Hackathon\Input data\level0.json"

data = pd.read_json(dataset)
neighbourhoods_df = pd.DataFrame(data['neighbourhoods'])
restaurants_df = pd.DataFrame(data['restaurants'])

neighbourhoods_df.dropna(inplace=True)

neighbours = []

for index, row in neighbourhoods_df.iterrows():
    neighbours.append(row['neighbourhoods']['distances'])
names = ['n' + str(i) for i in range(len(neighbours))]
print(names) 









