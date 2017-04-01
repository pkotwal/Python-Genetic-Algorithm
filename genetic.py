import numpy as np
import math

MAX_MUTATION = 5
MUTATION_RATE = 0.3
POPULATION_SIZE = 2
CROSSOVER_RATE = 0.9
ITERATIONS = 1000
NUM_INPUTS = 7
NUM_OUTPUTS = 3
LAYOUT = [NUM_INPUTS, 20, NUM_OUTPUTS]
START_GENERATION = 0

def init_network(layout):
    weights = list()
    for i in range(0, len(layout) - 1):
        temp_weight = (np.random.rand(layout[i + 1], layout[i] + 1) * 2) - 1
        weights.append(temp_weight)
    return weights

def predict(x, weights):
    temp = x
    for i in range(0, len(weights)):
        temp = add_bias(temp)
        temp = weights[i].dot(temp)
        temp = sigmoid(temp)
    return np.argmax(temp)

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

def add_bias(x):
    x = np.append([1], x)
    return x

def select(population, fitness, algorithm='roulette'):
    def roulette(population, fitness):
        eCount = []
        aCount = []
        prob = []
        total = sum(fitness)
        avg = np.mean(fitness)

        fitness = np.array(fitness)
        population = np.array(population)

        inds = fitness.argsort()
        inds[:] = inds[::-1]
        fitness = fitness[inds]
        population = population[inds]

        for i in range(0, len(population)):
            prob.append(fitness[i] / total)
            eCount.append(fitness[i] / avg)
            aCount.append(int(round(eCount[i])))
            print(round(fitness[i], 3), "\t", round(prob[i], 3), "\t\t", round(eCount[i], 3), "\t\t", aCount[i],
                  population[i].tolist())

        print("Fitness is: ", sum(fitness), "\n")
        temp = []

        for i in range(0, len(population)):
            for j in range(0, aCount[i]):
                temp.append(population[i].tolist())
        population = temp

        if (sum(aCount) > len(population)):
            population = population[:len(population)]
        if (sum(aCount) < len(population)):
            diff = len(population) - sum(aCount)
            for i in range(0, diff):
                population.append(population[i].tolist())
        return population

    if algorithm == 'roulette':
        return roulette(population, fitness)


def crossover(initial_population):
    population = []
    np.random.shuffle(initial_population)

    for i in range(0, int(POPULATION_SIZE / 2)):
        if (np.random.randint(0, 100) < CROSSOVER_RATE * 100):
            # print(len(initial_population[0]))
            crossover_point = np.random.randint(0, len(initial_population[0]))
            chr1 = []
            chr2 = []
            for j in range(0, crossover_point):
                chr1.append(initial_population[i * 2][j])
                chr2.append(initial_population[(i * 2) + 1][j])
            for j in range(crossover_point, len(initial_population[0])):
                chr1.append(initial_population[(i * 2) + 1][j])
                chr2.append(initial_population[(i * 2)][j])
            population.append(chr1)
            population.append(chr2)
        else:
            # print("Dont Crossover")
            population.append(initial_population[i * 2])
            population.append(initial_population[(i * 2) + 1])
    # print(population)
    return population

def mutation(initial_population):
    population = []
    # print(initial_population)
    for i in range(0, POPULATION_SIZE):
        if (np.random.randint(0, 100) < MUTATION_RATE * 100):
            # print("Mutate")
            ch = []
            mutation_point = []
            num_mutations = np.random.randint(1, MAX_MUTATION)
            for k in range(0, num_mutations):
                mutation_point.append(np.random.randint(1, len(initial_population[0])))
            # print(mutation_point)
            # print(len(initial_population[0]))
            for j in range(0, len(initial_population[0])):
                if (not (j in mutation_point)):

                    ch.append(initial_population[i][j])
                else:
                    # print((np.random.randn()*2)-1)
                    ch.append((np.random.randn() * 2) - 1)
            # print(ch)
            # print(ch)
            population.append(ch)
        else:
            # print("Dont Mutate")
            # print(initial_population[i])
            population.append(initial_population[i])
    # print(population)
    return population

def getFitness(attributes):
    return attributes

def start(function_to_run, max_mutations, mutation_rate, population_size, cressover_rate, iterations, num_inputs, num_outputs, layout=False, start_generation=0, fittness_function=getFitness):

    global MAX_MUTATION, MUTATION_RATE, POPULATION_SIZE, CROSSOVER_RATE, ITERATIONS, NUM_OUTPUTS, NUM_INPUTS, LAYOUT, START_GENERATION

    MAX_MUTATION = max_mutations
    MUTATION_RATE = mutation_rate
    POPULATION_SIZE = population_size
    CROSSOVER_RATE = cressover_rate
    ITERATIONS = iterations
    NUM_INPUTS = num_inputs
    NUM_OUTPUTS = num_outputs
    LAYOUT = layout
    START_GENERATION = start_generation
    if not layout:
        LAYOUT = [NUM_INPUTS, NUM_OUTPUTS]

    weights = []
    scores = []
    fitness = []
    timePlayed = []
    dim = (3, 8)
    allweights = []
    population = []
    current_fitess = 0
    attributes = []

    for i in range(START_GENERATION, ITERATIONS):
        print("\n\nGeneration ", i)

        for j in range(0, POPULATION_SIZE):

            if i == START_GENERATION:
                weights = init_network(LAYOUT)

            else:
                weights = allweights[j]

            currently_used_weights = []
            currently_used_weights_flattened = []
            for layer in weights:
                temp = layer.tolist()
                currently_used_weights.append(temp)
            print("Using: ", currently_used_weights)

            current_attributes = function_to_run(weights=weights, iteration=i, individual=j)
            attributes.append(attributes)
            fitness.append(fittness_function(current_attributes))
            list_weights = []
            for layer in weights:
                for neuron_weights in layer:
                    for individual_weight in neuron_weights:
                        list_weights.append(individual_weight)
            population.append(list_weights)

        allweights.clear()

        print("Before Selection\n", population)

        population = select(population, fitness)
        print("After Selection\n", population)

        population = crossover(population)
        print("After Crossover\n", population)

        population = mutation(population)
        print("After Mutation\n", population)

        allweights.clear()
        for weights in population:
            current_weights = []
            start = 0
            for j in range(0, len(LAYOUT) - 1):
                num_weights = (LAYOUT[j] + 1) * LAYOUT[j + 1]
                weight = np.array(weights[start:start + num_weights])
                weight = weight.reshape(num_weights, 1)
                weight = weight.reshape(LAYOUT[j + 1], (LAYOUT[j] + 1))
                current_weights.append(weight)
                start += num_weights
            allweights.append(current_weights)

        population.clear()
        fitness.clear()
        scores.clear()
