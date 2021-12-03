import numpy as np
import time


class EvolutionaryAlgorithm:
    def __init__(self, benchmark_function, population_size, offspring_size,
                 genotype_len):

        self.objective_function = benchmark_function['function']
        data_range = benchmark_function['initialization_range']
        self.data_range = (min(data_range), max(data_range))
        self.in_range = lambda x: (x >= self.data_range[0]) * (x <= self.data_range[1])
        self.global_optimum = benchmark_function['global_min']

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.genotype_len = genotype_len


class EvolutionStrategies:
    def __init__(self, EA, recombination_type='intermediate'):

        self.fitness = EA.objective_function
        self.global_optimum = EA.global_optimum
        self.population_size = EA.population_size
        self.offspring_size = EA.offspring_size
        self.genotype_len = EA.genotype_len
        self.data_range = EA.data_range
        self.in_range = EA.in_range

        if recombination_type == 'intermediate':
            self.recombination = self.intermediate_recombination
        elif recombination_type == 'discrete':
            self.recombination = self.discrete_recombination
        else:
            raise ValueError('recombination type should be "intermediate" or "discrete".')

    def sort_population(self):
        idx = np.argsort(self.fitness_matrix)
        self.fitness_matrix = self.fitness_matrix[idx]
        self.population = self.population[idx]

    def generate_first_generation(self):
        # returns binary array of size population_size * genotype_len
        range_mean = (self.data_range[1] + self.data_range[0]) / 2
        range_width = self.data_range[1] - range_mean
        population = (np.random.random((self.population_size, self.genotype_len)) - range_mean) * range_width
        self.fitness_matrix = self.fitness(population)
        self.population = population

        self.sort_population()

    def select_parents(self):
        parents = np.random.choice(np.arange(0, self.population_size), self.offspring_size * 2)
        self.parents = self.population[parents, :].reshape((self.offspring_size, 2, -1))

    def intermediate_recombination(self):
        self.offsprings = (self.parents[:, 0, :] + self.parents[:, 1, :]) / 2

    def discrete_recombination(self):
        mask = np.random.randint(0, 2, (self.parents.shape[0], self.parents.shape[1]))
        self.offsprings = (self.parents[:, 0, :] * mask) + (self.parents[:, 1, :] * (1 - mask))

    def mutation(self):
        p = np.random.normal(0, 1, self.offsprings.shape)
        self.offsprings = self.offsprings + p

    def generate_offsprings(self):
        self.recombination()
        self.mutation()
        self.offsprings_fitness_matrix = self.fitness(self.offsprings)

    def survival_selection(self):
        self.population = np.vstack((self.offsprings, self.population))
        self.fitness_matrix = np.hstack((self.offsprings_fitness_matrix, self.fitness_matrix))
        self.sort_population()
        self.population = self.population[:self.population_size]
        self.fitness_matrix = self.fitness_matrix[:self.population_size]

    def run(self, itteration, repeat=1):

        history = {'best_fitness': np.zeros(itteration),
                   'mean_fitness': np.zeros(itteration),
                   'answer_dist': np.zeros(repeat)}

        for r in range(repeat):
            self.generate_first_generation()

            for i in range(itteration):
                self.select_parents()
                self.generate_offsprings()
                self.survival_selection()

                history['best_fitness'][i] += self.fitness_matrix[0]
                history['mean_fitness'][i] += np.mean(self.fitness_matrix)

            if 'answer' not in history.keys() or self.fitness(history['answer']) > self.fitness_matrix[0]:
                history['answer'] = self.population[0]
            history['answer_dist'][r] = np.sum((self.population[0] - self.global_optimum) ** 2)

        history['best_fitness'] = history['best_fitness'] / repeat
        history['mean_fitness'] = history['mean_fitness'] / repeat

        return history


class AdaptiveEvolutionStrategies(EvolutionStrategies):
    def __init__(self, EA, recombination_type='intermediate',
                 sigma0=1, k=5, C=0.9):

        super().__init__(EA, recombination_type)

        self.sigma0 = sigma0
        self.k = k
        self.C = C

    def mutation(self):
        p = np.random.normal(0, self.sigma, self.offsprings.shape)
        self.offsprings = self.offsprings + p

    def generate_offsprings(self):

        self.recombination()
        before_mutation_fitness = self.fitness(self.offsprings)
        self.mutation()
        after_mutation_fitness = self.fitness(self.offsprings)
        self.offsprings_fitness_matrix = self.fitness(self.offsprings)

        self.successful_mutation += sum(after_mutation_fitness < before_mutation_fitness)
        self.all_mutation += self.offsprings.shape[0]

    def update_sigma(self, ps):
        if ps > 0.2:
            self.sigma = self.sigma / self.C
        elif ps < 0.2:
            self.sigma = self.sigma * self.C

        self.successful_mutation = 0
        self.all_mutation = 0


    def run(self, itteration, repeat=1):

        history = {'best_fitness': np.zeros(itteration),
                   'mean_fitness': np.zeros(itteration),
                   'sigma': np.zeros(int(itteration / self.k)),
                   'ps': np.zeros(int(itteration / self.k)),
                   'answer_dist': np.zeros(repeat)}

        for r in range(repeat):
            self.successful_mutation = 0
            self.all_mutation = 0
            self.sigma = self.sigma0
            self.generate_first_generation()

            for i in range(itteration):
                self.select_parents()
                self.generate_offsprings()
                self.survival_selection()
                if i % self.k == 0 and i != 0:
                    ps = self.successful_mutation / self.all_mutation
                    self.update_sigma(ps)
                    history['sigma'][int(i / self.k) - 1] += self.sigma
                    history['ps'][int(i / self.k) - 1] += ps

                history['best_fitness'][i] += self.fitness_matrix[0]
                history['mean_fitness'][i] += np.mean(self.fitness_matrix)

            if 'answer' not in history.keys() or self.fitness(history['answer']) > self.fitness_matrix[0]:
                history['answer'] = self.population[0]
            history['answer_dist'][r] = np.sum((self.population[0] - self.global_optimum) ** 2)

        history['best_fitness'] = history['best_fitness'] / repeat
        history['mean_fitness'] = history['mean_fitness'] / repeat
        history['sigma'] = history['sigma'] / repeat
        history['ps'] = history['ps'] / repeat

        return history


class SelfAdaptiveEvolutionStrategies(EvolutionStrategies):
    def __init__(self, EA, recombination_type='intermediate',
                 tau=-1, tau2=-1, eps=0.001):

        super().__init__(EA, recombination_type)

        self.base_fitness = EA.objective_function
        self.fitness = self.adaptive_fitness

        self.tau2 = (1 / ((2 * self.genotype_len) ** 0.5)) if tau2 == -1 else tau2
        self.tau = (1 / ((2 * (self.genotype_len ** 0.5)) ** 0.5)) if tau == -1 else tau
        self.eps = eps

    def adaptive_fitness(self, x):
        if len(x.shape) == 1:
            return self.base_fitness(x[:-1])
        elif len(x.shape) == 2:
            return self.base_fitness(x[:, :-1])

    def generate_first_generation(self):
        # returns binary array of size population_size * genotype_len
        range_mean = (self.data_range[1] + self.data_range[0]) / 2
        range_width = self.data_range[1] - range_mean
        population = (np.random.random((self.population_size, self.genotype_len)) - range_mean) * range_width
        sigmas = np.random.randn((self.population_size)).reshape(-1, 1)

        self.population = np.hstack((population, sigmas))
        self.fitness_matrix = self.fitness(self.pofoulation)

        self.sort_population()

    def mutation(self):
        self.offsprings[:, -1] = self.offsprings[:, -1] * \
                                 np.exp(self.tau2 * np.random.randn(self.offsprings.shape[0]) +
                                        self.tau * np.random.randn(self.offsprings.shape[0]))

        self.offsprings[:, -1][self.offsprings[:, -1] < self.eps] = self.eps
        sigma = self.offsprings[:, -1]

        p = np.random.normal(0, 1, self.offsprings[:, :-1].shape) * np.tile(sigma, [self.genotype_len, 1]).T
        self.offsprings[:, :-1] = self.offsprings[:, :-1] + p

    def run(self, itteration, repeat=1):

        history = {'best_fitness': np.zeros(itteration),
                   'mean_fitness': np.zeros(itteration),
                   'answer_dist': np.zeros(repeat)}

        for r in range(repeat):
            self.generate_first_generation()

            for i in range(itteration):
                self.select_parents()
                self.generate_offsprings()
                self.survival_selection()

                history['best_fitness'][i] += self.fitness_matrix[0]
                history['mean_fitness'][i] += np.mean(self.fitness_matrix)

            if 'answer' not in history.keys() or self.fitness(history['answer']) > self.fitness_matrix[0]:
                history['answer'] = self.population[0][:-1]
            history['answer_dist'][r] = np.sum((self.population[0][:-1] - self.global_optimum) ** 2)

        history['best_fitness'] = history['best_fitness'] / repeat
        history['mean_fitness'] = history['mean_fitness'] / repeat
        history['answer'] = history['answer'][:-1]

        return history


class DifferentialEvolution:
    def __init__(self, EA, F=0.1, Cr=0.1):
        self.fitness = EA.objective_function
        self.global_optimum = EA.global_optimum
        self.population_size = EA.population_size
        self.genotype_len = EA.genotype_len
        self.data_range = EA.data_range
        self.in_range = EA.in_range

        self.F = F
        self.Cr = Cr

    def sort_population(self):
        idx = np.argsort(self.fitness_matrix)
        self.fitness_matrix = self.fitness_matrix[idx]
        self.population = self.population[idx]

    def generate_first_generation(self):
        # returns binary array of size population_size * genotype_len
        range_mean = (self.data_range[1] + self.data_range[0]) / 2
        range_width = self.data_range[1] - range_mean
        population = (np.random.random((self.population_size, self.genotype_len)) - range_mean) * range_width
        self.fitness_matrix = self.fitness(population)
        self.population = population

        self.sort_population()

    def select_parents(self):
        # self.parents = np.zeros((self.population_size, 3, self.genotype_len))
        # for i in range(self.population_size):
        #     parents = np.random.choice(np.arange(0, self.population_size), 4, replace=False)
        #     parents = parents[parents != i][:3]
        #     self.parents[i] = np.array(list(self.population[parents]))
        parents = np.random.choice(np.arange(0, self.population_size), self.population_size * 3)
        self.parents = self.population[parents, :].reshape(self.population_size, 3, self.genotype_len)

    def get_mutants(self):
        mutants = self.parents[:, 0, :] + (self.F * (self.parents[:, 1, :] - self.parents[:, 2, :]))
        return mutants

    def generate_offsprings(self):
        v = self.get_mutants()

        r = np.random.random((self.population_size, self.genotype_len))
        forced_crossover = np.random.randint(0, self.genotype_len, self.population_size)
        for i in range(forced_crossover.shape[0]):
            r[i, forced_crossover[i]] = 0

        u = v * (r < self.Cr) + self.population * (r >= self.Cr)
        self.offsprings = u
        self.offspring_fitness = self.fitness(self.offsprings)

    def survival_selection(self):
        self.population[self.offspring_fitness < self.fitness_matrix] = self.offsprings[self.offspring_fitness < self.fitness_matrix]
        self.fitness_matrix[self.offspring_fitness < self.fitness_matrix] = self.offspring_fitness[self.offspring_fitness < self.fitness_matrix]

    def run(self, itteration, repeat=1):

        history = {'best_fitness': np.zeros(itteration),
                   'mean_fitness': np.zeros(itteration),
                   'answer_dist': np.zeros(repeat)}

        for r in range(repeat):
            self.generate_first_generation()

            for i in range(itteration):
                self.select_parents()
                self.generate_offsprings()
                self.survival_selection()
                self.sort_population()

                history['best_fitness'][i] += self.fitness_matrix[0]
                print(self.fitness_matrix[0])
                history['mean_fitness'][i] += np.mean(self.fitness_matrix)

            history['answer'] = self.population[0]
            history['answer_dist'][r] = np.sum((self.population[0] - self.global_optimum) ** 2)

        history['best_fitness'] = history['best_fitness'] / repeat
        history['mean_fitness'] = history['mean_fitness'] / repeat

        return history


class PSO:
    def __init__(self, EA, w=0.7, phi1=1.5, phi2=1.5):
        self.fitness = EA.objective_function
        self.global_optimum = EA.global_optimum
        self.population_size = EA.population_size
        self.genotype_len = EA.genotype_len
        self.data_range = EA.data_range
        self.in_range = EA.in_range

        self.W = w
        self.phi1 = phi1
        self.phi2 = phi2

    def generate_first_generation(self):
        # returns binary array of size population_size * genotype_len
        range_mean = (self.data_range[1] + self.data_range[0]) / 2
        range_width = self.data_range[1] - range_mean
        population = (np.random.random((self.population_size, self.genotype_len)) - range_mean) * range_width
        self.fitness_matrix = self.fitness(population)
        self.population = population

        self.global_best = self.population[np.argmin(self.fitness_matrix)]
        self.local_best = self.population
        self.velocity = np.random.random((self.population_size, self.genotype_len)) * 2 - 1

    def update_velocity(self):
        u1 = np.random.random((self.population_size, self.genotype_len))
        u2 = np.random.random((self.population_size, self.genotype_len))
        x = self.population
        y = self.local_best
        z = self.global_best
        self.velocity = (self.W * self.velocity) + (self.phi1 * u1 * (y - x)) + (self.phi2 * u2 * (z - x))

    def update_position(self):
        self.population += self.velocity
        # print(self.data_range[0], self.data_range[1])
        # self.population = np.clip(self.population, self.data_range[0], self.data_range[1])
        # print(self.population)

    def update_swarm(self):
        self.update_velocity()
        self.update_position()
        self.fitness_matrix = self.fitness(self.population)

    def update_bests(self):
        if self.fitness(self.global_best) > np.min(self.fitness_matrix):
            self.global_best = self.population[np.argmin(self.fitness_matrix)]

        best_changed = self.fitness(self.local_best) > self.fitness_matrix
        self.local_best[best_changed] = self.population[best_changed]

    def run(self, itteration, repeat=1):

        history = {'best_fitness': np.zeros(itteration),
                   'mean_fitness': np.zeros(itteration),
                   'answer_dist': np.zeros(repeat)}

        for r in range(repeat):
            self.generate_first_generation()

            for i in range(itteration):
                self.update_swarm()
                self.update_bests()

                history['best_fitness'][i] += np.min(self.fitness_matrix)
                history['mean_fitness'][i] += np.mean(self.fitness_matrix)

            if 'answer' not in history.keys() or self.fitness(history['answer']) > self.fitness(self.global_best):
                history['answer'] = self.global_best
            history['answer_dist'][r] = np.sum((self.global_best - self.global_optimum) ** 2)


        history['best_fitness'] = history['best_fitness'] / repeat
        history['mean_fitness'] = history['mean_fitness'] / repeat
        return history


if __name__ == "__main__":
    from benchmark_functions import benchmark_functions
    import matplotlib.pyplot as plt

    EA = EvolutionaryAlgorithm(benchmark_functions['quadratic'], 10, 10, 30)

    # adaptiveES = AdaptiveEvolutionStrategies(EA)
    #
    # history = adaptiveES.run(500, 10)
    # plt.figure('Adaptive ES')
    # plt.subplot(2, 1, 1)
    # plt.title('fitness')
    # plt.plot(history['best_fitness'], 'r-')
    # plt.plot(history['mean_fitness'], 'b-')
    # plt.subplot(2, 2, 3)
    # plt.title('sigma')
    # plt.plot(history['sigma'], 'k-')
    # plt.subplot(2, 2, 4)
    # plt.title('ps')
    # plt.plot(history['ps'], 'g-')
    # print('Adaptive ES :', benchmark_functions['quadratic']['function'](history['answer']))
    # print('Adaptive ES :', history['answer_dist'])
    #
    # selfAdaptiveES = SelfAdaptiveEvolutionStrategies(EA)
    #
    # history = selfAdaptiveES.run(500, 10)
    # plt.figure('self-adaptive ES')
    # plt.title('fitness')
    # plt.plot(history['best_fitness'], 'r-')
    # plt.plot(history['mean_fitness'], 'b-')
    # print('Self-daptive ES :', benchmark_functions['quadratic']['function'](history['answer']))
    # print('Self-daptive ES :', history['answer_dist'])
    #
    DE = DifferentialEvolution(EA)
    history = DE.run(5000, 1)
    plt.figure('DE')
    plt.title('fitness')
    plt.plot(history['best_fitness'], 'r-')
    plt.plot(history['mean_fitness'], 'b-')
    print('DE :', benchmark_functions['quadratic']['function'](history['answer']))
    print('DE :', history['answer_dist'])

    EA = EvolutionaryAlgorithm(benchmark_functions['ackley'], 1000, 50, 30)

    # pso = PSO(EA)
    # history = pso.run(1000, 5)
    # plt.figure('PSO')
    # plt.title('fitness')
    # plt.plot(history['best_fitness'], 'r-')
    # plt.plot(history['mean_fitness'], 'b-')
    # print('PSO :', benchmark_functions['quadratic']['function'](history['answer']))
    # print('PSO :', history['answer_dist'])

    plt.show()
