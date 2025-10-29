import random

# Parameters
POP_SIZE = 500  # Number of chromosomes
MUT_RATE = 0.2  # Mutation rate
TARGET = 'naruto'  # Goal string
GENES = ' abcdefghijklmnopqrstuvwxyz'  # Gene pool

# Initialize population
def initialize_pop(TARGET):
    population = []
    tar_len = len(TARGET)

    for _ in range(POP_SIZE):
        temp = [random.choice(GENES) for _ in range(tar_len)]
        population.append(temp)

    return population

# Fitness calculation (0 = perfect)
def fitness_cal(TARGET, chromo_from_pop):
    difference = sum(t1 != t2 for t1, t2 in zip(TARGET, chromo_from_pop))
    return [chromo_from_pop, difference]

# Selection (top 50%)
def selection(population):
    sorted_chromo_pop = sorted(population, key=lambda x: x[1])
    return sorted_chromo_pop[:int(0.5 * POP_SIZE)]

# Crossover
def crossover(selected_chromo, CHROMO_LEN, population):
    offspring_cross = []
    for _ in range(POP_SIZE):
        parent1 = random.choice(selected_chromo)
        parent2 = random.choice(population[:50])  # small sample

        p1 = parent1[0]
        p2 = parent2[0]

        crossover_point = random.randint(1, CHROMO_LEN - 1)
        child = p1[:crossover_point] + p2[crossover_point:]
        offspring_cross.append(child)
    return offspring_cross

# Mutation
def mutate(offspring, MUT_RATE):
    mutated_offspring = []
    for arr in offspring:
        for i in range(len(arr)):
            if random.random() < MUT_RATE:
                arr[i] = random.choice(GENES)
        mutated_offspring.append(arr)
    return mutated_offspring

# Replacement
def replace(new_gen, population):
    for i in range(len(population)):
        if population[i][1] > new_gen[i][1]:
            population[i] = new_gen[i]
    return population

# Main GA loop
def main():
    population = [fitness_cal(TARGET, chromo) for chromo in initialize_pop(TARGET)]
    generation = 1

    while True:
        population.sort(key=lambda x: x[1])
        best = population[0]

        print(f"Generation: {generation}, String: {''.join(best[0])}, Fitness: {best[1]}")

        if best[1] == 0:
            print(f"\nTarget found in {generation} generations!")
            break

        selected = selection(population)
        crossovered = crossover(selected, len(TARGET), population)
        mutated = mutate(crossovered, MUT_RATE)
        new_gen = [fitness_cal(TARGET, chromo) for chromo in mutated]
        population = replace(new_gen, population)

        generation += 1


if __name__ == "__main__":
    main()
