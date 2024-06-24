from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.spatial.distance import cdist
from itertools import permutations
import random

app = Flask(__name__)

# 測試數據：初始節點位置
locations = np.array([
    [22.997585, 120.212661] #台南車站
])

# 計算距離矩陣，使用 haversine 距離計算
def compute_distance_matrix(locations):
    from geopy.distance import distance

    n = len(locations)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                coords_i = (locations[i][0], locations[i][1])
                coords_j = (locations[j][0], locations[j][1])
                distance_matrix[i][j] = distance(coords_i, coords_j).km

    return distance_matrix

#讓VRP路線的顏色不一樣
def generate_random_color(): 
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())

@app.route('/')
def index():
    return render_template('index.html', locations=locations.tolist())

#增加新的站點
@app.route('/add_location', methods=['POST'])
def add_location():
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    global locations
    locations = np.append(locations, [[lat, lng]], axis=0)
    return jsonify({'locations': locations.tolist()})

#刪除站點
@app.route('/delete_location/<int:index>', methods=['DELETE'])
def delete_location(index):
    global locations
    if 0 < index < len(locations):
        locations = np.delete(locations, index, axis=0)
        return jsonify({'locations': locations.tolist()})
    else:
        return jsonify({'error': '無效的索引'}), 400

# TSP算法，使用修正的距離矩陣
def tsp_brute_force(distance_matrix):
    n = len(distance_matrix)
    min_cost = float('inf')
    best_route = None

    for perm in permutations(range(n)):
        cost = sum(distance_matrix[perm[i]][perm[i+1]] for i in range(n-1))
        cost += distance_matrix[perm[-1]][perm[0]]

        if cost < min_cost:
            min_cost = cost
            best_route = perm

    return best_route, min_cost

# VRP 算法，使用啟發式方法
def vrp_heuristic(distance_matrix, num_vehicles):
    n = len(distance_matrix)
    depot = 0  # 假設第一個節點為中心

    # 每輛車的初始路徑
    routes = [[] for _ in range(num_vehicles)]

    # 初始化節點列表（除了中心點）
    nodes = list(range(1, n))

    # 將節點按照距離中心點的距離排序
    nodes.sort(key=lambda x: distance_matrix[depot][x])

    # 將節點分配給每輛車
    for i, node in enumerate(nodes):
        routes[i % num_vehicles].append(node)

    # 計算各車輛的總行駛成本
    total_cost = 0
    for route in routes:
        route.insert(0, depot)
        route.append(depot)
        route_cost = sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))
        total_cost += route_cost

    return routes, total_cost

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    method = data.get('method', 'tsp')
    num_vehicles = int(data.get('num_vehicles', 1))

    if method == 'tsp':
        if len(locations) < 2:
            return jsonify({'error': '至少需要兩個站點來執行TSP'}), 400
        distance_matrix = compute_distance_matrix(locations)
        route, cost = tsp_brute_force(distance_matrix)
        return jsonify({'route': list(route), 'cost': cost})
    elif method == 'vrp':
        if len(locations) < num_vehicles:
            return jsonify({'error': '站點數量不足以支持指定的車輛數量'}), 400
        distance_matrix = compute_distance_matrix(locations)
        routes, total_cost = vrp_heuristic(distance_matrix, num_vehicles)
        colors = [generate_random_color() for _ in range(num_vehicles)]
        return jsonify({'routes': routes, 'cost': total_cost, 'colors': colors})

    return jsonify({'error': '無效的方法'}), 400

if __name__ == '__main__':
    app.run(debug=True)
