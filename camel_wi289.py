import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Define eil51 cities with fixed coordinates
eil51_coordinates = {
    '1': (37, 52), '2': (49, 49), '3': (52, 64), '4': (20, 26), '5': (40, 30),
    '6': (21, 47), '7': (17, 63), '8': (31, 62), '9': (52, 33), '10': (51, 21),
    '11': (42, 41), '12': (31, 32), '13': (5, 25), '14': (12, 42), '15': (36, 16),
    '16': (52, 41), '17': (27, 23), '18': (17, 33), '19': (13, 13), '20': (57, 58),
    '21': (62, 42), '22': (42, 57), '23': (16, 57), '24': (8, 52), '25': (7, 38),
    '26': (27, 68), '27': (30, 48), '28': (43, 67), '29': (58, 48), '30': (58, 27),
    '31': (37, 69), '32': (38, 46), '33': (46, 10), '34': (61, 33), '35': (62, 63),
    '36': (63, 69), '37': (32, 22), '38': (45, 35), '39': (59, 15), '40': (5, 6),
    '41': (10, 17), '42': (21, 10), '43': (5, 64), '44': (30, 15), '45': (39, 10),
    '46': (32, 39), '47': (25, 32), '48': (25, 55), '49': (48, 28), '50': (56, 37),
    '51': (30, 40)
}

# Calculate the distance matrix for eil51
def calculate_distance_matrix(coordinates):
    num_cities = len(coordinates)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i, coord1 in enumerate(coordinates.values()):
        for j, coord2 in enumerate(coordinates.values()):
            dist_matrix[i, j] = np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return dist_matrix

dist_matrix_eil51 = calculate_distance_matrix(eil51_coordinates)

# Define the fitness function for TSP
def fitness(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i + 1]]
    distance += dist_matrix[route[-1], route[0]]  # return to start
    return distance

# Camel Algorithm for TSP on eil51
# Camel Algorithm for TSP on eil51
def camel_algorithm_tsp(cities, dist_matrix, max_iter, num_camels, use_nn=False):
    num_cities = len(cities)
    cities_list = list(cities.keys())
    best_route = None
    best_fitness = float('inf')

    for _ in range(max_iter):
        camels = []
        for _ in range(num_camels):
            start_city = random.choice(cities_list)
            current_city = cities_list.index(start_city)
            unvisited_cities = set(cities_list)
            unvisited_cities.remove(start_city)
            tour = [current_city]

            if use_nn:
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
            tour.append(cities_list.index(start_city))
            camels.append(tour)

        # Evaluate fitness for all camels
        camels_fitness = [fitness(route, dist_matrix) for route in camels]

        # Select the best camel
        best_camel_index = np.argmin(camels_fitness)
        best_camel = camels[best_camel_index]
        best_camel_fitness = camels_fitness[best_camel_index]

        # Update best solution found so far
        if best_camel_fitness < best_fitness:
            best_route = best_camel
            best_fitness = best_camel_fitness

    return best_route, best_fitness

# Benchmarking function for eil51
def benchmark_algorithm(algorithm_func, cities, dist_matrix, max_iter, num_camels, use_nn, num_trials=10):
    total_distances = []
    total_time = 0

    for _ in range(num_trials):
        start_time = time.time()
        _, best_fitness = algorithm_func(cities, dist_matrix, max_iter, num_camels, use_nn)
        end_time = time.time()
        execution_time = end_time - start_time
        total_time += execution_time
        total_distances.append(best_fitness)

    best_distance = min(total_distances)
    worst_distance = max(total_distances)
    avg_distance = np.mean(total_distances)
    std_distance = np.std(total_distances)
    avg_time = total_time / num_trials

    return best_distance, worst_distance, avg_distance, std_distance, avg_time

# Example usage of benchmarking Camel Algorithm with and without NN on eil51
max_iterations = 1000
num_camels = 50  # Number of camels (solution candidates)
num_trials = 10

# Benchmark Camel Algorithm without NN on eil51
best_no_nn, worst_no_nn, avg_no_nn, std_no_nn, avg_time_no_nn = benchmark_algorithm(camel_algorithm_tsp, eil51_coordinates, dist_matrix_eil51, max_iterations, num_camels, use_nn=False, num_trials=num_trials)

print("\nBenchmarking Camel Algorithm without NN on eil51:")
print(f"Best Distance: {best_no_nn:.2f}")
print(f"Worst Distance: {worst_no_nn:.2f}")
print(f"Average Distance: {avg_no_nn:.2f}")
print(f"Standard Deviation: {std_no_nn:.2f}")
print(f"Average Time: {avg_time_no_nn:.4f} seconds")

# Benchmark Camel Algorithm with NN on eil51
best_nn, worst_nn, avg_nn, std_nn, avg_time_nn = benchmark_algorithm(camel_algorithm_tsp, eil51_coordinates, dist_matrix_eil51, max_iterations, num_camels, use_nn=True, num_trials=num_trials)

print("\nBenchmarking Camel Algorithm with NN on eil51:")
print(f"Best Distance: {best_nn:.2f}")
print(f"Worst Distance: {worst_nn:.2f}")
print(f"Average Distance: {avg_nn:.2f}")
print(f"Standard Deviation: {std_nn:.2f}")
print(f"Average Time: {avg_time_nn:.4f} seconds")

# Visualize the best route found by the Camel Algorithm (best_no_nn or best_nn)
def plot_tour(coordinates, route, title):
    x = [coordinates[str(city)][0] for city in route]
    y = [coordinates[str(city)][1] for city in route]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.plot([x[-1], x[0]], [y[-1], y[0]], linestyle='-', color='b')  # connect back to the start
    plt.title(title)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()

# Plot the best route found by Camel Algorithm without NN on eil51
plot_tour(eil51_coordinates, best_no_nn, 'Best Tour found by Camel Algorithm without NN (eil51)')

# Plot the best route found by Camel Algorithm with NN on eil51
plot_tour(eil51_coordinates, best_nn, 'Best Tour found by Camel Algorithm with NN (eil51)')
