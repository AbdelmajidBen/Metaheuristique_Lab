import numpy as np
import random
from tabulate import tabulate
import time

# Define cities with fixed coordinates
cities = {
    'Oujda': (0, 0),
    'Tanger': (5, 4),
    'Ahfir': (1, 3),
    'Dakhla': (6, 1),
    'Casa': (2, 2),
    'Rabat': (3, 5),
    'Boujdor': (7, 5)
}

city_names = list(cities.keys())

# Calculate the distance matrix
def calculate_distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i, city1 in enumerate(cities.values()):
        for j, city2 in enumerate(cities.values()):
            dist_matrix[i, j] = np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    return dist_matrix

dist_matrix = calculate_distance_matrix(cities)

# Print distance matrix
print("Distance Matrix:")
print(tabulate(dist_matrix, headers=city_names, showindex=city_names, floatfmt=".2f"))

# Define the fitness function
def fitness(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i + 1]]
    distance += dist_matrix[route[-1], route[0]]  # return to start
    return distance

# Camel Algorithm for TSP
def camel_algorithm_tsp(cities, dist_matrix, max_iter, num_camels, start_city, use_nn=False):
    num_cities = len(cities)
    cities_list = list(cities.keys())
    start_city_index = cities_list.index(start_city)
    best_route = None
    best_fitness = float('inf')

    for iteration in range(max_iter):
        camels = []
        for _ in range(num_camels):
            current_city = start_city_index
            unvisited_cities = set(cities_list)
            unvisited_cities.remove(start_city)
            tour = [current_city]

            supply = random.uniform(5, 10)  # Higher supply means more exploratory
            if use_nn or supply < 7.5:  # Use Nearest Neighbor if supply is low
                while unvisited_cities:
                    nearest_city = min(unvisited_cities, key=lambda city: dist_matrix[current_city, cities_list.index(city)])
                    tour.append(cities_list.index(nearest_city))
                    unvisited_cities.remove(nearest_city)
                    current_city = cities_list.index(nearest_city)
            else:
                for _ in range(num_cities - 1):
                    next_city = random.choice(list(unvisited_cities))
                    tour.append(cities_list.index(next_city))
                    unvisited_cities.remove(next_city)
                    current_city = cities_list.index(next_city)

            # Complete the tour
            tour.append(start_city_index)
            camels.append({
                "route": tour,
                "endurance": random.uniform(0.5, 1.5),
                "temperature": random.uniform(30, 50),
                "supply": supply
            })

        # Evaluate fitness for all camels
        for camel in camels:
            camel["fitness"] = fitness(camel["route"])

        # Select the best camel
        best_camel = min(camels, key=lambda c: c["fitness"])

        # Update best solution found so far
        if best_camel["fitness"] < best_fitness:
            best_route = best_camel["route"]
            best_fitness = best_camel["fitness"]

    return best_route, best_fitness, camels

# Print the route and details of each camel
def print_camel_details(route, camels):
    cities_list = list(cities.keys())
    for camel in camels:
        route_cities = [cities_list[i] for i in camel["route"]]
        print(f"Route: {' -> '.join(route_cities)}")
        print(f"Fitness: {camel['fitness']:.2f}, Endurance: {camel['endurance']:.2f}, Temperature: {camel['temperature']:.2f}, Supply: {camel['supply']:.2f}\n")

# Example usage
max_iterations = 1000
num_camels = 50  # Number of camels (solution candidates)
start_city = 'Oujda'  # Fixed starting city

best_route, best_fitness, camels = camel_algorithm_tsp(cities, dist_matrix, max_iterations, num_camels, start_city, use_nn=False)

print("\nBest Route:")
best_route_cities = [city_names[i] for i in best_route]
print(" -> ".join(best_route_cities))
print(f"Total Distance: {best_fitness:.2f}\n")

print("Details of each camel:")
print_camel_details(best_route, camels)

print(f"Perfect Fitness (Minimum Distance): {best_fitness:.2f}")
print(f"Best Route Path: {' -> '.join(best_route_cities)}")
