"""
tsp_ga.py

Algorytm genetyczny dla problemu komiwojażera (TSP) z:
- selekcją: nieproporcjonalna ruletka (rank-based roulette),
- krzyżowaniem: CX (cycle crossover),
- mutacją: mutacja równomierna (swap z losową pozycją dla każdego genu z prawdop. pm_gene).

Uruchomienie:
    python tsp_ga.py

Autor: wygenerowane automatycznie
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# -----------------------
# Dane testowe (10 miast)
# -----------------------
CITY_NAMES = ["A","B","C","D","E","F","G","H","I","J"]
CITIES = {
    "A": (2, 1),
    "B": (9, 7),
    "C": (6, 5),
    "D": (1, 7),
    "E": (3, 6),
    "F": (5, 6),
    "G": (4, 2),
    "H": (10, 4),
    "I": (7, 3),
    "J": (8, 10),
}
# For convenience: ordered coordinate list
COORDS = [CITIES[name] for name in CITY_NAMES]


# -----------------------
# Funkcje pomocnicze
# -----------------------
def euclidean(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def tour_length(tour: List[int], coords: List[Tuple[float,float]]) -> float:
    """Długość cyklicznej trasy permutacji indeksów"""
    L = 0.0
    n = len(tour)
    for i in range(n):
        a = coords[tour[i]]
        b = coords[tour[(i+1)%n]]
        L += euclidean(a,b)
    return L

# -----------------------
# Operatory GA
# -----------------------
def init_population(n_pop: int, n_cities: int) -> List[List[int]]:
    pop = []
    base = list(range(n_cities))
    for _ in range(n_pop):
        ind = base[:]
        random.shuffle(ind)
        pop.append(ind)
    return pop

def rank_based_roulette_selection(pop: List[List[int]], fitness_values: List[float], n_select: int) -> List[List[int]]:
    """
    Nieproporcjonalna ruletka: rank-based roulette.
    - fitness_values: wartości dopasowania (niższe lepsze - długości tras)
    - przypisujemy rangę 1..N (1=best), a prawdopodobieństwo proporcjonalne do (N - rank + 1)
    Zwraca wybrane osobniki (z powtórzeniami) - tzw. selection with replacement.
    """
    N = len(pop)
    # sortujemy po rosnącym fitness (mniejsze = lepsze)
    order = sorted(range(N), key=lambda i: fitness_values[i])
    ranks = [0]*N
    for rank_pos, idx in enumerate(order, start=1):
        ranks[idx] = N - rank_pos + 1  # większa liczba -> większe prawdopodobieństwo
    weights = np.array(ranks, dtype=float)
    probs = weights / weights.sum()
    chosen_indices = np.random.choice(N, size=n_select, replace=True, p=probs)
    return [pop[i][:] for i in chosen_indices]

def cycle_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    CX - Cycle Crossover dla permutacji.
    Zwraca dwa potomne.
    """
    n = len(parent1)
    child1 = [-1]*n
    child2 = [-1]*n
    # znajdź cykle
    visited = [False]*n
    for start in range(n):
        if visited[start]:
            continue
        # utwórz cykl zaczynając od start
        idx = start
        cycle_indices = []
        while not visited[idx]:
            visited[idx] = True
            cycle_indices.append(idx)
            # znajdź wartość w parent2, która odpowiada parent1[idx]
            val = parent1[idx]
            idx = parent2.index(val)
        # dla indeksów w cyklu - przypisz je z parent1 do child1, z parent2 do child2
        if len(cycle_indices) > 0:
            for i in cycle_indices:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
    # pozostałe pozycje uzupełnij zamiennie (standardowe CX)
    for i in range(n):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]
    return child1, child2

def uniform_mutation(individual: List[int], pm_gene: float) -> List[int]:
    """
    Mutacja równomierna dla permutacji:
    Dla każdego indeksu i z prawdop. pm_gene wybieramy losowy indeks j i swap(i,j).
    (czyli dla każdego genu mamy równą szansę zostać przemieszczonym)
    """
    ind = individual[:]
    n = len(ind)
    for i in range(n):
        if random.random() < pm_gene:
            j = random.randrange(n)
            ind[i], ind[j] = ind[j], ind[i]
    return ind

# -----------------------
# Główna pętla GA
# -----------------------
def run_ga(coords: List[Tuple[float,float]],
           pop_size: int = 100,
           n_generations: int = 200,
           pm_gene: float = 0.05,
           elitism: int = 2,
           verbose: bool = False):
    """
    Uruchamia GA i zwraca historię statystyk oraz końcową populację.
    - pm_gene: prawdopodobieństwo zamiany każdego genu (swap) w mutacji równomiernej
    """
    n_cities = len(coords)
    pop = init_population(pop_size, n_cities)

    # statystyki
    hist_avg = []
    hist_min = []
    hist_max = []

    for gen in range(1, n_generations+1):
        # oblicz fitness (długość trasy)
        fitness = [tour_length(ind, coords) for ind in pop]
        avg = float(np.mean(fitness))
        mn = float(np.min(fitness))
        mx = float(np.max(fitness))
        hist_avg.append(1/avg)
        hist_min.append(1/mn)
        hist_max.append(1/mx)

        if verbose and gen % max(1, n_generations//10) == 0:
            print(f"Gen {gen}/{n_generations}: avg={avg:.3f}, min={mn:.3f}, max={mx:.3f}")

        # elitism: zachowaj najlepsze indywidua bezpośrednio
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitness[i])
        new_pop = [pop[i][:] for i in sorted_idx[:elitism]]

        # selekcja (nieproporcjonalna ruletka)
        n_to_select = (pop_size - elitism)
        selected = rank_based_roulette_selection(pop, fitness, n_to_select)

        # krzyżowanie: paruj kolejnych wybranych w pary, stosuj CX
        children = []
        for i in range(0, len(selected)-1, 2):
            p1 = selected[i]
            p2 = selected[i+1]
            c1, c2 = cycle_crossover(p1, p2)
            children.append(c1)
            children.append(c2)
        # jeśli nieparzysta liczba - skopiuj ostatniego
        if len(selected) % 2 == 1:
            children.append(selected[-1][:])
        # jeżeli powstało za dużo dzieci (np. n_to_select < len(children)), obetnij
        children = children[:n_to_select]

        # mutacja
        mutated_children = []
        for child in children:
            mutated = uniform_mutation(child, pm_gene)
            mutated_children.append(mutated)
        # nowa populacja
        new_pop.extend(mutated_children)
        pop = new_pop

    # końcowe fitness i sortowanie
    final_fitness = [tour_length(ind, coords) for ind in pop]
    final_sorted_idx = sorted(range(len(pop)), key=lambda i: final_fitness[i])
    best_idx = final_sorted_idx[0]
    worst_idx = final_sorted_idx[-1]

    result = {
        "population": pop,
        "fitness": final_fitness,
        "history": {
            "avg": hist_avg,
            "min": hist_min,
            "max": hist_max,
        },
        "best": {
            "index": best_idx,
            "tour": pop[best_idx],
            "length": final_fitness[best_idx],
        },
        "worst": {
            "index": worst_idx,
            "tour": pop[worst_idx],
            "length": final_fitness[worst_idx],
        }
    }
    return result

# -----------------------
# Rysowanie / Wizualizacja
# -----------------------
def plot_history(history: dict):
    gens = range(1, len(history["avg"])+1)
    plt.figure(figsize=(8,4))
    plt.plot(gens, history["avg"], label="Średnie")
    plt.plot(gens, history["min"], label="Najlepsze (min)")
    plt.plot(gens, history["max"], label="Najgorsze (max)")
    plt.xlabel("Generacja")
    plt.ylabel("Fitness")
    plt.title("Historie dopasowania populacji")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tour(tour: List[int], coords: List[Tuple[float,float]], title: str):
    x = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    y = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
    plt.figure(figsize=(5,5))
    # rysuj krawędzie
    plt.plot(x, y, marker='o')
    # rysuj miasta i etykiety
    for idx, i in enumerate(tour):
        cx, cy = coords[i]
        label = CITY_NAMES[i] if idx < len(CITY_NAMES) else str(i)
        plt.scatter([cx], [cy])
        plt.text(cx + 0.15, cy + 0.15, label)
    plt.xlim(0, 10.5)
    plt.ylim(0, 10.5)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_two_maps(best_tour: List[int], worst_tour: List[int], coords: List[Tuple[float,float]]):
    fig, axes = plt.subplots(1,2, figsize=(11,5))
    # Best
    ax = axes[0]
    x = [coords[i][0] for i in best_tour] + [coords[best_tour[0]][0]]
    y = [coords[i][1] for i in best_tour] + [coords[best_tour[0]][1]]
    ax.plot(x,y, marker='o')
    for idx, i in enumerate(best_tour):
        cx, cy = coords[i]
        ax.scatter([cx],[cy])
        label = CITY_NAMES[i] if idx < len(CITY_NAMES) else str(i)
        ax.text(cx + 0.15, cy + 0.15, label)
    ax.set_title("Najkrótsza trasa (finalna populacja)")
    ax.set_xlim(0,10.5)
    ax.set_ylim(0,10.5)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Worst
    ax = axes[1]
    x = [coords[i][0] for i in worst_tour] + [coords[worst_tour[0]][0]]
    y = [coords[i][1] for i in worst_tour] + [coords[worst_tour[0]][1]]
    ax.plot(x,y, marker='o')
    for idx, i in enumerate(worst_tour):
        cx, cy = coords[i]
        ax.scatter([cx],[cy])
        label = CITY_NAMES[i] if idx < len(CITY_NAMES) else str(i)
        ax.text(cx + 0.15, cy + 0.15, label)
    ax.set_title("Najdłuższa trasa (finalna populacja)")
    ax.set_xlim(0,10.5)
    ax.set_ylim(0,10.5)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

# -----------------------
# Uruchomienie przykładowe
# -----------------------
if __name__ == "__main__":

    POP_SIZE = 100
    GENERATIONS = 100
    PM_GENE = 0.02
    ELITISM = 0

    print("Uruchamiam GA dla TSP - dane testowe 10 miast (mapa 10x10).")
    result = run_ga(COORDS,
                    pop_size=POP_SIZE,
                    n_generations=GENERATIONS,
                    pm_gene=PM_GENE,
                    elitism=ELITISM,
                    verbose=True)

    print("\nWyniki końcowe:")
    print(f"Najlepsza długość: {result['best']['length']:.3f}")
    print("Najlepsza trasa (kolejność miast):", " -> ".join(CITY_NAMES[i] for i in result['best']['tour']))
    print(f"Najgorsza długość: {result['worst']['length']:.3f}")
    print("Najdłuższa trasa (kolejność miast):", " -> ".join(CITY_NAMES[i] for i in result['worst']['tour']))

    plot_history(result['history'])

    plot_two_maps(result['best']['tour'], result['worst']['tour'], COORDS)
