from benchmark_functions import benchmark_functions
from EvolutionaryAlgorithm import *

import matplotlib.pyplot as plt

for function in benchmark_functions:
    EA = EvolutionaryAlgorithm(benchmark_functions[function], 1000, 1000, 30)

    plt.figure()
    plt.title(function)

    adaptiveES = AdaptiveEvolutionStrategies(EA)
    history = adaptiveES.run(1000, 10)
    print('Adaptive ES (' + function + ')', history['answer_dist'])
    plt.plot(history['best_fitness'], 'r-', label='Adaptive ES')

    selfAdaptiveES = SelfAdaptiveEvolutionStrategies(EA)
    history = selfAdaptiveES.run(1000, 10)
    print('Self-Adaptive ES (' + function + ')', history['answer_dist'])
    plt.plot(history['best_fitness'], 'g-', label='Self-adaptive ES')

    DE = DifferentialEvolution(EA)
    history = DE.run(1000, 10)
    print('DE (' + function + ')', history['answer_dist'])
    plt.plot(history['best_fitness'], 'b-', label='DE')

    pso = PSO(EA)
    history = pso.run(1000, 10)
    print('PSO (' + function + ')', history['answer_dist'])
    plt.plot(history['best_fitness'], 'k-', label='PSO')

    print('-------------------------------')

    plt.legend()

plt.show()