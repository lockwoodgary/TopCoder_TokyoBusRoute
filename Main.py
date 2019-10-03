import json
import pip
import sys
import subprocess

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install('numpy')
install('pandas')
install('matplotlib')

import numpy as np 
import pandas as pd 
from matplotlib import path as mplPath
import matplotlib.pyplot as plt 


class Route:
    def __init__(self, route_id, coords, cost, tz_max_buses_dict):
        self.route_id = route_id
        self.coords = np.array(self.flattenCoords(coords))
        self.cost = float(cost)
        self.tz_max_buses_dict = self.convertToMaxPerTz(tz_max_buses_dict)
        self.pop_per_tz_agegroup_array = np.zeros((6,4))
        
    def flattenCoords(self, coords):
        return [ coord for sublist in coords for coord in sublist ]

    def convertToMaxPerTz(self, tz_max_buses_dict):
        # Multiply the max value per hour by the number of hours in the tz
        tz_max_buses_dict['TZ1_Max'] = int(tz_max_buses_dict['TZ1_Max'] * 10)
        tz_max_buses_dict['TZ2_Max'] = int(tz_max_buses_dict['TZ2_Max'] *2)
        tz_max_buses_dict['TZ3_Max'] = int(tz_max_buses_dict['TZ3_Max'] * 8)
        tz_max_buses_dict['TZ4_Max'] = int(tz_max_buses_dict['TZ4_Max'] * 4)
        return tz_max_buses_dict
    
    def updateGridsInside(self, grids_dict):
        self.grid_ids = []
        for key,grid in grids_dict.items():
            if any(grid.containsPoints(self.coords)):
                self.grid_ids.append(key)
    
    def updatePopPerTZVector(self, grids_dict):
        assert(len(self.grid_ids) > 0)
        
        for grid_id in self.grid_ids:
            self.pop_per_tz_agegroup_array += grids_dict[grid_id].agegroup_tz_array
        
        self.pop_per_tz_vector = self.pop_per_tz_agegroup_array.sum(axis=0)

class Grid:
    def __init__(self, mesh_id, path, agegroup_tz_array, tz_ratio_array, agegroups_one_hot):
        self.mesh_id = mesh_id
        self.path = self.convertPath(path)
        self.agegroup_tz_array = agegroup_tz_array*tz_ratio_array*agegroups_one_hot
    
    def convertPath(self, path):
        # Flatten first
        while True:
            if len(path) == 1:
                path = path[0]
            else:
                break
        return mplPath.Path(np.array(path), closed=True) # Convert to mpl Path object
    
    def containsPoint(self, point):
        return self.path.contains_point(point)
    
    def containsPoints(self, points):
        return self.path.contains_points(points)


def run():
    input_params = [ s.rstrip() for s in sys.stdin.readlines() ]

    agegroups = np.array([int(x) for x in input_params[0].split(',')])
    total_budget = float(input_params[1])
    popgeo_fn = input_params[2].rstrip().replace('\\', '/')
    routegeo_fn = input_params[3].rstrip().replace('\\', '/')
    active_fn = input_params[4].rstrip().replace('\\', '/')

    # print([agegroups,total_budget,popgeo_fn,routegeo_fn,active_fn])
    # Convert age groups to one-hot encoding
    num_classes = 6
    agegroups_one_hot = np.eye(num_classes)[agegroups]
    agegroups_one_hot = agegroups_one_hot.sum(axis=0).reshape(num_classes,1)

    for v in agegroups_one_hot:
        assert(v[0] <= 1)

    del agegroups

    # Get timezone ratio values
    df = pd.read_csv(f'{active_fn}')
    tz_ratio_array = df.values

    del df

    with open(f'{routegeo_fn}', 'rb') as f:
        route_data = json.load(f)
    
    with open(f'{popgeo_fn}', 'rb') as f:
        pop_data = json.load(f)
    
    routes = []
    for feature in route_data['features']:
        route_id = feature['properties']['RouteID']
        coords = feature['geometry']['coordinates']
        cost = feature['properties']['Cost']
        tz_max_buses_dict = { k:feature['properties'][k] for k in ['TZ1_Max','TZ2_Max','TZ3_Max','TZ4_Max'] }
        routes.append(Route(route_id, coords, cost, tz_max_buses_dict))
        
    del route_data

    grids_dict = {}
    for feature in pop_data['features']:
        mesh_id = feature['properties']['MESH_ID']
        path = feature['geometry']['coordinates']
        agegroup_tz_array = np.array([[ feature['properties']['G1_TZ1'], feature['properties']['G1_TZ2'], feature['properties']['G1_TZ3'], feature['properties']['G1_TZ4'] ],
                                    [ feature['properties']['G2_TZ1'], feature['properties']['G2_TZ2'], feature['properties']['G2_TZ3'], feature['properties']['G2_TZ4'] ],
                                    [ feature['properties']['G3_TZ1'], feature['properties']['G3_TZ2'], feature['properties']['G3_TZ3'], feature['properties']['G3_TZ4'] ],
                                    [ feature['properties']['G4_TZ1'], feature['properties']['G4_TZ2'], feature['properties']['G4_TZ3'], feature['properties']['G4_TZ4'] ],
                                    [ feature['properties']['G5_TZ1'], feature['properties']['G5_TZ2'], feature['properties']['G5_TZ3'], feature['properties']['G5_TZ4'] ],
                                    [ feature['properties']['G6_TZ1'], feature['properties']['G6_TZ2'], feature['properties']['G6_TZ3'], feature['properties']['G6_TZ4'] ]])
        grids_dict[str(mesh_id)] = Grid(mesh_id, path, agegroup_tz_array, tz_ratio_array, agegroups_one_hot)

    for route in routes:
        route.updateGridsInside(grids_dict)
    
    for route in routes:
        route.updatePopPerTZVector(grids_dict)

    del grids_dict

    # Combine all route and tz data into a DataFrame for easy analysis
    routes_by_tz_df = pd.DataFrame()
    for route in routes:
        for tz_idx,total_pop in enumerate(route.pop_per_tz_vector):
            if total_pop > 0:
                routes_by_tz_df = routes_by_tz_df.append({'route_id':route.route_id, 
                                                        'tz':tz_idx+1,
                                                        'cost':route.cost,
                                                        'total_pop':total_pop,
                                                        'max_buses':route.tz_max_buses_dict[f'TZ{tz_idx+1}_Max']}, ignore_index=True)

    routes_by_tz_df['pop_per_cost'] = routes_by_tz_df['total_pop'] / routes_by_tz_df['cost']
    routes_by_tz_df = routes_by_tz_df.sort_values('pop_per_cost', ascending=False).reset_index(drop=True)

    min_cost = min(routes_by_tz_df.cost)

    budget_left = total_budget

    rte_ids, tzs = [],[]
    for i,row in routes_by_tz_df.iterrows():
        if budget_left < min_cost:
            break
            
        max_buses = row['max_buses']
        while (budget_left > row['cost'] and max_buses > 0):
            rte_ids.append(int(row['route_id']))
            tzs.append(int(row['tz']))
            budget_left -= row['cost']
            max_buses -= 1
    
    for i in range(len(rte_ids)):
        print(f'{rte_ids[i]}, {tzs[i]}')
    

if __name__ == '__main__':
    run()