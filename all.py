import numpy as np
import random
from tabulate import tabulate

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

# Genetic Algorithm with "camel" behavior
def genetic_algorithm_with_camels(num_cities, num_camels, generations, mutation_rate, start_city, end_city):
    print(f"\nGenetic Algorithm with Camels\nStart city: {city_names[start_city]}, End city: {city_names[end_city]}")

    # Initialize population
    population = [list(np.random.permutation(num_cities)) for _ in range(num_camels)]

    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_values = [fitness(route) for route in population]
        
        # Select parents (tournament selection)
        parents = []
        for _ in range(num_camels):
            tournament = random.sample(population, 5)
            tournament_fitness = [fitness(route) for route in tournament]
            parents.append(tournament[np.argmin(tournament_fitness)])
        
        # Generate offspring (crossover and mutation)
        offspring = []
        for _ in range(num_camels):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = np.random.randint(1, num_cities - 1)
            child = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
            if np.random.rand() < mutation_rate:
                swap_indices = np.random.choice(num_cities, 2, replace=False)
                child[swap_indices[0]], child[swap_indices[1]] = child[swap_indices[1]], child[swap_indices[0]]
            offspring.append(child)
        
        population = offspring

    # Select the best route
    best_route_index = np.argmin([fitness(route) for route in population])
    best_route = population[best_route_index]

    # Convert route indices to city names
    best_route_names = [city_names[i] for i in best_route]

    # Ensure start and end cities are correct
    best_route_names = [city_names[start_city]] + [city for city in best_route_names if city != city_names[start_city] and city != city_names[end_city]] + [city_names[end_city]]

    # Display the best route and its fitness
    best_route_fitness = fitness([city_names.index(city) for city in best_route_names])
    print("Best route:", " -> ".join(best_route_names))
    print("Total distance (fitness):", best_route_fitness)
    return best_route_fitness

# Team Game Algorithm
def team_game_algorithm(num_cities, num_teams, iterations, start_city, end_city):
    print(f"\nTeam Game Algorithm\nStart city: {city_names[start_city]}, End city: {city_names[end_city]}")

    # Initialize teams
    teams = [list(np.random.permutation(num_cities)) for _ in range(num_teams)]

    # Iteration loop
    for _ in range(iterations):
        for team in teams:
            for i in range(num_cities - 1):
                current_fitness = fitness(team)
                new_team = team[:]
                swap_indices = np.random.choice(num_cities, 2, replace=False)
                new_team[swap_indices[0]], new_team[swap_indices[1]] = new_team[swap_indices[1]], new_team[swap_indices[0]]
                new_fitness = fitness(new_team)
                if new_fitness < current_fitness:
                    team[:] = new_team

    # Select the best team route
    best_team_index = np.argmin([fitness(team) for team in teams])
    best_team_route = teams[best_team_index]

    # Convert route indices to city names
    best_team_route_names = [city_names[i] for i in best_team_route]

    # Ensure start and end cities are correct
    best_team_route_names = [city_names[start_city]] + [city for city in best_team_route_names if city != city_names[start_city] and city != city_names[end_city]] + [city_names[end_city]]

    # Display the best route and its fitness
    best_team_route_fitness = fitness([city_names.index(city) for city in best_team_route_names])
    print("Best route:", " -> ".join(best_team_route_names))
    print("Total distance (fitness):", best_team_route_fitness)
    return best_team_route_fitness

# Bonobo Optimizer
def bonobo_optimizer(num_cities, num_bonobos, iterations, start_city, end_city):
    print(f"\nBonobo Optimizer\nStart city: {city_names[start_city]}, End city: {city_names[end_city]}")

    # Initialize bonobos
    bonobos = [list(np.random.permutation(num_cities)) for _ in range(num_bonobos)]

    # Iteration loop
    for _ in range(iterations):
        for bonobo in bonobos:
            for i in range(num_cities - 1):
                current_fitness = fitness(bonobo)
                new_bonobo = bonobo[:]
                swap_indices = np.random.choice(num_cities, 2, replace=False)
                new_bonobo[swap_indices[0]], new_bonobo[swap_indices[1]] = new_bonobo[swap_indices[1]], new_bonobo[swap_indices[0]]
                new_fitness = fitness(new_bonobo)
                if new_fitness < current_fitness:
                    bonobo[:] = new_bonobo

    # Select the best bonobo route
    best_bonobo_index = np.argmin([fitness(bonobo) for bonobo in bonobos])
    best_bonobo_route = bonobos[best_bonobo_index]

    # Convert route indices to city names
    best_bonobo_route_names = [city_names[i] for i in best_bonobo_route]

    # Ensure start and end cities are correct
    best_bonobo_route_names = [city_names[start_city]] + [city for city in best_bonobo_route_names if city != city_names[start_city] and city != city_names[end_city]] + [city_names[end_city]]

    # Display the best route and its fitness
    best_bonobo_route_fitness = fitness([city_names.index(city) for city in best_bonobo_route_names])
    print("Best route:", " -> ".join(best_bonobo_route_names))
    print("Total distance (fitness):", best_bonobo_route_fitness)
    return best_bonobo_route_fitness

# Parameters
num_cities = len(cities)
num_camels = 50  # Population size (number of camels)
generations = 1000  # Number of generations
mutation_rate = 0.01  # Mutation rate
num_teams = 50  # Number of teams
iterations = 1000  # Number of iterations
num_bonobos = 50  # Number of bonobos

# Randomly select start and end cities once
start_city, end_city = np.random.choice(num_cities, 2, replace=False)

# Run Genetic Algorithm with Camels
ga_distance = genetic_algorithm_with_camels(num_cities, num_camels, generations, mutation_rate, start_city, end_city)

# Run Team Game Algorithm
tg_distance = team_game_algorithm(num_cities, num_teams, iterations, start_city, end_city)

# Run Bonobo Optimizer
bo_distance = bonobo_optimizer(num_cities, num_bonobos, iterations, start_city, end_city)

# Display comparison
print("\nComparison:")
print(f"Genetic Algorithm with Camels Distance: {ga_distance}")
print(f"Team Game Algorithm Distance: {tg_distance}")
print(f"Bonobo Optimizer Distance: {bo_distance}")
