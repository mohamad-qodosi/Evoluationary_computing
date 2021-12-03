import numpy as np
import matplotlib.pyplot as plt
import random
import string

target_str = 'Mohammad Ghoddosi - SB university, 98443182. Gmail:mohamadqodosi1996@gmail.com'
char_set = string.ascii_letters + '1234567890' + ' !@#$%^&*()_+-=,./<>?":;'
population_size = 50

def fitness(chromosome):
    matche = [1 if chromosome[i] == ord(target_str[i]) else 0 for i in range(len(target_str))]
    return sum(matche) / len(matche)

def generate_first_generation(population_size, genotype_len):
    # returns binary array of size population_size * genotype_len
    population = np.random.randint(0, 256, (population_size, genotype_len))
    return population.tolist()


def print_chr(chromosome):
    ord_str = ''.join([chr(i) for i in chromosome])
    print(ord_str)


def roletroll(cumulative_chance):
    p = np.random.random()
    for i in range(len(cumulative_chance)):
        if cumulative_chance[i] >= p:
            return i


def select_parents(population):
    population_fitness = np.array(list(map(lambda x: x['fitness'], population)))
    sum_fitness = np.sum(population_fitness)
    population_chance = population_fitness / sum_fitness
    cumulative_chance = np.cumsum(population_chance)

    parents = []
    for i in range(population_size * 4):
        index = roletroll(cumulative_chance)
        parents.append(population[index])

    parents = [(parents[i], parents[i + 1]) for i in range(0, len(parents), 2)]
    return parents


def recombination(parents):
    parent1 = parents[0]['chromosome']
    parent2 = parents[1]['chromosome']

    cutoff = np.random.randint(0, len(target_str))
    offspring1 = parent1[:cutoff] + parent2[cutoff:]
    offspring2 = parent2[:cutoff] + parent1[cutoff:]

    return offspring1, offspring2


def mutation(offspring):
    chance = np.random.rand(len(offspring))
    for i in np.where(chance)[0]:
        offspring[i] = ord(random.sample(char_set, 1)[0])

    return offspring


def generate_offsprings(parents_list):
    offsprings = []
    for parents in parents_list:
        offspring1, offspring2 = recombination(parents)

        offspring1 = mutation(offspring1)
        offspring2 = mutation(offspring2)

        offsprings = offsprings + [{'chromosome': offspring1, 'fitness': fitness(offspring1)},
                                   {'chromosome': offspring2, 'fitness': fitness(offspring2)}]
    return offsprings


population = list(map(lambda x: {'chromosome': x, 'fitness': fitness(x)},generate_first_generation(population_size, len(target_str))))
population = sorted(population, key=lambda x: x['fitness'], reverse=True)

best_fitness = 0
all_fitness = []
all_fitness.append(list(map(lambda x: x['fitness'], population)))

for generation in range(1000):
    print(generation)
    parents = select_parents(population)
    offsprings = generate_offsprings(parents)

    population = offsprings + population

    population = sorted(population, key=lambda x: x['fitness'], reverse=True)[:population_size]
    best_fitness = population[0]['fitness']

    all_fitness.append(list(map(lambda x: x['fitness'], population)))
    if best_fitness == 1:
        break

best_solution = population[0]['chromosome']

all_fitness = np.array(all_fitness)
plt.plot(all_fitness.mean(axis=1), 'ro-', label='mean fitness')
plt.plot(all_fitness.max(axis=1), 'bo--', label='max fitness')
plt.legend()

print_chr(best_solution)

plt.show()