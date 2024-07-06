import numpy as np
import random
from tabulate import tabulate
import tsplib95
import time

# Define cities with fixed coordinates
cities = {
    'Oujda': (0, 0),
    'Tanger': (5, 4),
    'Ahfir': (1, 3),
    'Dakhla': (6, 1),
    'Casa': (2, 2),
    'Rabat': (3, 5)
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

# Bonobo Optimizer
def bonobo_optimizer(num_cities, num_bonobos, iterations, start_city_idx):
    print(f"\nBonobo Optimizer\nStart and End city: {city_names[start_city_idx]}")

    # Initialize bonobos
    bonobos = [list(np.random.permutation(range(1, num_cities))) for _ in range(num_bonobos)]

    # Iteration loop
    for iter in range(iterations):
        print(f"\nIteration {iter + 1}:")
        for i, bonobo in enumerate(bonobos):
            bonobo = [start_city_idx] + bonobo + [start_city_idx]  # Ensure start and end at start_city_idx
            current_fitness = fitness(bonobo)
            new_bonobo = bonobo[:]
            swap_indices = np.random.choice(range(1, num_cities), 2, replace=False)  # Avoid swapping start/end city
            new_bonobo[swap_indices[0]], new_bonobo[swap_indices[1]] = new_bonobo[swap_indices[1]], new_bonobo[swap_indices[0]]
            new_fitness = fitness(new_bonobo)
            if new_fitness < current_fitness:
                bonobos[i] = new_bonobo[1:-1]  # Remove start/end city before updating bonobo
                print(f"Bonobo {i + 1} path: {city_names[start_city_idx]} -> {' -> '.join(city_names[idx] for idx in new_bonobo)} -> {city_names[start_city_idx]}, Fitness: {new_fitness}")
            else:
                print(f"Bonobo {i + 1} path: {city_names[start_city_idx]} -> {' -> '.join(city_names[idx] for idx in bonobo)} -> {city_names[start_city_idx]}, Fitness: {current_fitness}")

    # Select the best bonobo route
    best_bonobo_index = np.argmin([fitness([start_city_idx] + bonobo + [start_city_idx]) for bonobo in bonobos])
    best_bonobo_route = bonobos[best_bonobo_index]

    # Ensure start and end cities are correct
    best_bonobo_route = [start_city_idx] + best_bonobo_route + [start_city_idx]

    # Convert route indices to city names
    best_bonobo_route_names = [city_names[idx] for idx in best_bonobo_route]

    # Display the best route and its fitness
    best_bonobo_route_fitness = fitness(best_bonobo_route)
    print("\nBest route:", " -> ".join(best_bonobo_route_names))
    print("Total distance (fitness):", best_bonobo_route_fitness)
    return best_bonobo_route_fitness

# Benchmarking function
def benchmark_bonobo_optimizer(problem_name, num_cities, num_bonobos, iterations, start_city_idx):
    print(f"\nBenchmarking Bonobo Optimizer on problem {problem_name}")

    # Record start time
    start_time = time.time()

    # Run Bonobo Optimizer
    bonobo_distance = bonobo_optimizer(num_cities, num_bonobos, iterations, start_city_idx)

    # Record end time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time

    # Display results
    print("\nBonobo Optimizer Result:")
    print(f"Total distance (fitness): {bonobo_distance}")
    print(f"Execution time: {execution_time:.2f} seconds")

# Parameters for benchmarking
num_cities = len(cities)
num_bonobos = 50  # Number of bonobos
iterations = 1000  # Number of iterations
start_city_idx = 0  # Index of the starting city in city_names

# Run benchmarking
benchmark_bonobo_optimizer('Custom TSP Problem', num_cities, num_bonobos, iterations, start_city_idx)
