"""
Traveling Salesman Problem (TSP) solved using a Genetic Algorithm (GA)
Author: Your Name
"""

import matplotlib.pyplot as plt
from itertools import permutations
from random import shuffle
import random
import numpy as np
import seaborn as sns

# ----------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------
x = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
cities_names = [
    "Gliwice", "Cairo", "Rome", "Krakow", "Paris",
    "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"
]
city_coords = dict(zip(cities_names, zip(x, y)))

n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# ----------------------------------------------------------
# VISUALIZE INITIAL MAP
# ----------------------------------------------------------
colors = sns.color_palette("pastel", len(cities_names))
city_icons = {
    "Gliwice": "♕", "Cairo": "♖", "Rome": "♗", "Krakow": "♘",
    "Paris": "♙", "Alexandria": "♔", "Berlin": "♚",
    "Tokyo": "♛", "Rio": "♜", "Budapest": "♝"
}

fig, ax = plt.subplots()
ax.grid(False)

for i, (city, (cx, cy)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(cx, cy, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (cx, cy), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (cx, cy), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                textcoords='offset points')
    for j, (other_city, (ox, oy)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([cx, ox], [cy, oy], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
plt.show()

# ----------------------------------------------------------
# GENETIC ALGORITHM FUNCTIONS
# ----------------------------------------------------------
def initial_population(cities_list, n_population=250):
    """Generate random initial population from all permutations"""
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    return [list(possible_perms[i]) for i in random_ids]

def dist_two_cities(city_1, city_2):
    """Euclidean distance between two cities"""
    c1, c2 = city_coords[city_1], city_coords[city_2]
    return np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2))

def total_dist_individual(individual):
    """Total travel distance for one tour"""
    return sum(dist_two_cities(individual[i], individual[(i + 1) % len(individual)])
               for i in range(len(individual)))

def fitness_prob(population):
    """Compute fitness probabilities for roulette wheel selection"""
    distances = np.array([total_dist_individual(ind) for ind in population])
    max_cost = max(distances)
    fitness = max_cost - distances
    probs = fitness / sum(fitness)
    return probs

def roulette_wheel(population, fitness_probs):
    """Select one individual proportionally to fitness"""
    cumsum = fitness_probs.cumsum()
    index = np.searchsorted(cumsum, np.random.rand())
    return population[index]

def crossover(p1, p2):
    """Single-point crossover"""
    cut = random.randint(1, len(cities_names) - 1)
    o1 = p1[:cut] + [c for c in p2 if c not in p1[:cut]]
    o2 = p2[:cut] + [c for c in p1 if c not in p2[:cut]]
    return o1, o2

def mutation(offspring):
    """Swap two random cities"""
    i1, i2 = random.sample(range(len(cities_names)), 2)
    offspring[i1], offspring[i2] = offspring[i2], offspring[i1]
    return offspring

# ----------------------------------------------------------
# MAIN GA LOOP
# ----------------------------------------------------------
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for gen in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents = [roulette_wheel(population, fitness_probs)
                   for _ in range(int(crossover_per * n_population))]
        offspring = []
        for i in range(0, len(parents), 2):
            o1, o2 = crossover(parents[i], parents[(i + 1) % len(parents)])
            if random.random() < mutation_per: o1 = mutation(o1)
            if random.random() < mutation_per: o2 = mutation(o2)
            offspring.extend([o1, o2])
        combined = population + offspring
        fitness_probs = fitness_prob(combined)
        sorted_idx = np.argsort(fitness_probs)[::-1]
        population = [combined[i] for i in sorted_idx[:n_population]]
    return population

best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
distances = [total_dist_individual(p) for p in best_population]
best_path = best_population[np.argmin(distances)]
min_dist = min(distances)

# ----------------------------------------------------------
# PLOT BEST ROUTE
# ----------------------------------------------------------
x_best = [city_coords[c][0] for c in best_path] + [city_coords[best_path[0]][0]]
y_best = [city_coords[c][1] for c in best_path] + [city_coords[best_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_best, y_best, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

plt.title("TSP Best Route Using Genetic Algorithm", fontsize=22, color="k")
str_params = f"\n{n_generations} Generations\n{n_population} Population\n{crossover_per} Crossover\n{mutation_per} Mutation"
plt.suptitle(f"Total Distance: {round(min_dist, 3)}{str_params}", fontsize=16, y=1.04)

for i, city in enumerate(best_path):
    ax.annotate(f"{i+1}- {city}", (x_best[i], y_best[i]), fontsize=16)

fig.set_size_inches(16, 12)
plt.show()
