import numpy as np
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

# Passing operator
def passing_operator(player_position, captain_position, random_player_position, r=0.1):
    new_position = player_position + r * (2 * captain_position - player_position - random_player_position)
    return new_position

# Mistake operator
def mistake_operator(player_position, opponent_player_position):
    dim_to_change = np.random.randint(len(player_position))  # Randomly select a dimension to change
    player_position[dim_to_change] = opponent_player_position[dim_to_change]
    return player_position

# Substitution operator
def substitution_operator(team, iter_idx, max_iterations_without_improvement=50):
    fitnesses = [fitness(player) for player in team]
    best_fitness = min(fitnesses)
    best_index = np.argmin(fitnesses)
    
    if iter_idx % max_iterations_without_improvement == 0:
        worst_index = np.argmax(fitnesses)
        team[worst_index] = list(np.random.permutation(len(team[worst_index])))  # Substitute with a random new player
    
    return team

# Team Game Algorithm with operators
def team_game_algorithm(num_cities, num_players_per_team, iterations, start_city, end_city):
    print(f"\nTeam Game Algorithm\nStart city: {city_names[start_city]}, End city: {city_names[end_city]}")

    # Initialize teams
    team1 = [list(np.random.permutation(num_cities)) for _ in range(num_players_per_team)]
    team2 = [list(np.random.permutation(num_cities)) for _ in range(num_players_per_team)]

    # Captain positions (for passing operator)
    captain1 = np.mean(team1, axis=0)
    captain2 = np.mean(team2, axis=0)

    # Iteration loop
    for iter_idx in range(iterations):
        print(f"Iteration {iter_idx + 1}:")
        
        # Team 1 move
        for player_idx in range(num_players_per_team):
            current_fitness_team1 = fitness(team1[player_idx])
            print(f"Team 1 Player {player_idx + 1} route:", [city_names[i] for i in team1[player_idx]])
            print(f"Fitness: {current_fitness_team1:.2f}")
            
            # Perform passing operator
            random_player_idx = np.random.randint(num_players_per_team)
            new_position = passing_operator(team1[player_idx], captain1, team1[random_player_idx])
            team1[player_idx] = new_position.astype(int)
            
            # Perform mistake operator with opponent team
            opponent_player_idx = np.random.randint(num_players_per_team)
            team1[player_idx] = mistake_operator(team1[player_idx], team2[opponent_player_idx])
            
            # Check for substitution
            team1 = substitution_operator(team1, iter_idx)

            new_fitness_team1 = fitness(team1[player_idx])
            print(f"New route after operators:", [city_names[i] for i in team1[player_idx]])
            print(f"New fitness: {new_fitness_team1:.2f}")
            print()  # Blank line for readability
        
        # Team 2 move
        for player_idx in range(num_players_per_team):
            current_fitness_team2 = fitness(team2[player_idx])
            print(f"Team 2 Player {player_idx + 1} route:", [city_names[i] for i in team2[player_idx]])
            print(f"Fitness: {current_fitness_team2:.2f}")
            
            # Perform passing operator
            random_player_idx = np.random.randint(num_players_per_team)
            new_position = passing_operator(team2[player_idx], captain2, team2[random_player_idx])
            team2[player_idx] = new_position.astype(int)
            
            # Perform mistake operator with opponent team
            opponent_player_idx = np.random.randint(num_players_per_team)
            team2[player_idx] = mistake_operator(team2[player_idx], team1[opponent_player_idx])
            
            # Check for substitution
            team2 = substitution_operator(team2, iter_idx)

            new_fitness_team2 = fitness(team2[player_idx])
            print(f"New route after operators:", [city_names[i] for i in team2[player_idx]])
            print(f"New fitness: {new_fitness_team2:.2f}")
            print()  # Blank line for readability

    # Select the best route from both teams
    best_team1_fitness = min([fitness(player) for player in team1])
    best_team2_fitness = min([fitness(player) for player in team2])
    
    if best_team1_fitness < best_team2_fitness:
        best_route = team1[np.argmin([fitness(player) for player in team1])]
        best_team_name = "Team 1"
    else:
        best_route = team2[np.argmin([fitness(player) for player in team2])]
        best_team_name = "Team 2"

    # Convert route indices to city names
    best_route_names = [city_names[start_city]] + [city_names[i] for i in best_route if i != start_city and i != end_city] + [city_names[end_city]]

    # Display the best route and its fitness
    best_fitness = fitness(best_route)
    print(f"Best route ({best_team_name}):", " -> ".join(best_route_names))
    print("Total distance (fitness):", best_fitness)
    return best_fitness

# Parameters
num_cities = len(cities)
num_players_per_team = 3  # Number of players per team
iterations = 3  # Number of iterations

# Randomly select start and end cities once
start_city = 0  # Start and end at the same city (index 0)
end_city = 0

# Run Team Game Algorithm
tg_distance = team_game_algorithm(num_cities, num_players_per_team, iterations, start_city, end_city)

# Display results
print("\nTeam Game Algorithm Result:")
print(f"Total distance (fitness): {tg_distance:.2f}")
