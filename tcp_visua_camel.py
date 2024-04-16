#to execute  /usr/local/bin/python3 /Users/mac/Desktop/Programation/tcp_visua_camel.py   

import random
import math
import matplotlib.pyplot as plt
import numpy as np

NUM_CITIES = 10
POPULATION_SIZE = 50
MAX_ITERATIONS = 1000

class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

def generate_cities(num_cities):
    cities = []
    for i in range(num_cities):
        name = "City" + str(i+1)
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append(City(name, x, y))
    return cities

def calculate_distance(city1, city2):
    return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

def initialize_population(population_size, cities):
    population = []
    for _ in range(population_size):
        route = random.sample(cities, len(cities))
        population.append(route)
    return population

def adjust_route(route):
    new_route = route[:]
    index1, index2 = random.sample(range(1, len(route)-1), 2)
    new_route[index1], new_route[index2] = new_route[index2], new_route[index1]
    return new_route

def objective_function(route):
    total_distance = 0
    for i in range(len(route) - 1):
        distance = calculate_distance(route[i], route[i+1])
        total_distance += distance
    return total_distance

def camel_optimization(num_camels, cities, max_iterations, destination):
    population = initialize_population(num_camels, cities)
    
    best_route = []
    worst_route = []
    
    for _ in range(max_iterations):
        fitness = [objective_function(route) for route in population]
        
        best_camel_index = fitness.index(min(fitness))
        best_route = population[best_camel_index]
        
        for i in range(num_camels):
            population[i] = adjust_route(population[i])

    for route in [best_route, worst_route]:
        if route and route[-1] != destination:
            route[-1], route[route.index(destination)] = route[route.index(destination)], route[-1]
    
    worst_camel_index = fitness.index(max(fitness))
    worst_route = population[worst_camel_index]
    
    return best_route, worst_route

def plot_cities_and_route(cities, route, color, start_color='g', end_color='r'):
    plt.figure(figsize=(10, 6))
    
    for city in cities:
        if city == route[0]:
            plt.plot(city.x, city.y, start_color, marker='o', markersize=8)
            plt.text(city.x, city.y, city.name, fontsize=9, ha='right', va='bottom')
        elif city == route[-1]:
            plt.plot(city.x, city.y, end_color, marker='o', markersize=8)
            plt.text(city.x, city.y, city.name, fontsize=9, ha='right', va='bottom')
        else:
            plt.plot(city.x, city.y, 'bo')
            plt.text(city.x, city.y, city.name, fontsize=9, ha='right', va='bottom')
    
    route_x = [city.x for city in route]
    route_y = [city.y for city in route]
    route_x.append(route[0].x)
    route_y.append(route[0].y)
    plt.plot(route_x, route_y, color)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cities and Route')
    plt.grid(True)
    plt.show()

def display_distance_table(cities):
    distances = np.zeros((len(cities), len(cities)))
    for i in range(len(cities)):
        for j in range(len(cities)):
            distances[i][j] = calculate_distance(cities[i], cities[j])
    
    print("Distance Table:")
    print("{:<8}".format(''), end='')
    for city in cities:
        print("{:<8}".format(city.name), end='')
    print()
    for i, city in enumerate(cities):
        print("{:<8}".format(city.name), end='')
        for j, other_city in enumerate(cities):
            if i != j:
                print("{:<8.2f}".format(distances[i][j]), end='')
            else:
                print("{:<8}".format('-'), end='')
        print()

def main():
    cities = generate_cities(NUM_CITIES)
    
    display_distance_table(cities)
    
    destination = random.choice(cities)
    print("\nYour destination is:", destination.name)
    
    starting_city = random.choice(cities)
    print("Your starting city is:", starting_city.name)
    
    best_route, worst_route = camel_optimization(POPULATION_SIZE, cities, MAX_ITERATIONS, destination)
    
    best_route.remove(starting_city)
    best_route.insert(0, starting_city)
    best_route.append(destination)
    
    worst_route.remove(starting_city)
    worst_route.insert(0, starting_city)
    worst_route.append(destination)
    
    print("\nThe best road is:", end=" ")
    for city in best_route:
        print(city.name, "->", end=" ")
    print("Total distance for the best road:", objective_function(best_route))
    
    print("\nThe worst road is:", end=" ")
    for city in worst_route:
        print(city.name, "->", end=" ")
    print("Total distance for the worst road:", objective_function(worst_route))
    
    plot_cities_and_route(cities, best_route, 'b-')
    plot_cities_and_route(cities, worst_route, 'r-')

if __name__ == "__main__":
    main()
