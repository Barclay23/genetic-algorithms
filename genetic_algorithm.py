import random
import matplotlib.pyplot as plt
import math


POPULATION_SIZE = 10
NUM_GENERATIONS = 50
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.05
OFFSET = 70
DOMAIN = list(range(-1, 22))

def fitness_function(x):
    return -0.4 * x**2 + 4 * x + 6


SHIFT = -min(DOMAIN)
ENCODE_MAX = max(DOMAIN) + SHIFT
BIT_LENGTH = math.ceil(math.log2(ENCODE_MAX + 1))

def encode(x):
    shifted = x + SHIFT
    return format(shifted, f'0{BIT_LENGTH}b')

def decode(bits):
    shifted_value = int(bits, 2)
    return shifted_value - SHIFT


def initialize_population():
    return [encode(random.choice(DOMAIN)) for _ in range(POPULATION_SIZE)]


def selection(population, fitness_values):
    decoded = [decode(ind) for ind in population]
    sorted_pop = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    ranks = list(range(len(sorted_pop), 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]
    selected = random.choices([x[0] for x in sorted_pop], weights=probs, k=POPULATION_SIZE)
    return selected


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROB:
        point = random.randint(1, BIT_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual):
    while True:
        mutated = ""
        for bit in individual:
            if random.random() < MUTATION_PROB:
                mutated += '0' if bit == '1' else '1'
            else:
                mutated += bit

        decoded_value = decode(mutated)

        if min(DOMAIN) <= decoded_value <= max(DOMAIN):
            return encode(decoded_value)


def genetic_algorithm():
    population = initialize_population()
    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    min_fitness_per_gen = []

    for generation in range(NUM_GENERATIONS):
        decoded = [decode(ind) for ind in population]
        fitness_values = [fitness_function(x) for x in decoded]

        best_fitness_per_gen.append(max(max(fitness_values)+OFFSET ,0))
        avg_fitness_per_gen.append(max(sum(fitness_values) / len(fitness_values)+OFFSET ,0))
        min_fitness_per_gen.append(max(min(fitness_values)+OFFSET, 0))

        best_idx = fitness_values.index(max(fitness_values))
        best_x = decoded[best_idx]
        best_fit = fitness_values[best_idx]
        print(f"Generacja {generation+1:02d}: najlepszy x = {best_x}, fitness = {best_fit:.4f}")

        selected = selection(population, fitness_values)
        next_population = []

        for i in range(0, POPULATION_SIZE, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % POPULATION_SIZE]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_population.extend([c1, c2])

        population = next_population[:POPULATION_SIZE]

    return best_fitness_per_gen, avg_fitness_per_gen, min_fitness_per_gen


def plot_results(best, avg, min_):
    generations = list(range(NUM_GENERATIONS))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(generations, best, label="Max Fitness", color="green")
    plt.plot(generations, avg, label="Avg Fitness", color="blue")
    plt.plot(generations, min_, label="Min Fitness", color="red")
    plt.xlabel("Generacja")
    plt.ylabel("Dopasowanie")
    plt.title("Postęp dopasowania (binarne GA)")
    plt.legend()


    plt.subplot(1, 2, 2)
    xs = DOMAIN
    ys = [fitness_function(x) for x in xs]
    plt.plot(xs, ys, marker='o')
    plt.title("Funkcja dopasowania")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.tight_layout()
    plt.show()
    plt.savefig("wykres.png")


if __name__ == "__main__":
    best, avg, min_ = genetic_algorithm()
    plot_results(best, avg, min_)
