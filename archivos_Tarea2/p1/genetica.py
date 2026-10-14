import numpy as np
from p1_main import LOW, HIGH, N, transmittance, objective
import matplotlib.pyplot as plt
import os
import sys


def blended_crossover(p1, p2, alpha=0.3):

   
    s1 = p1 - alpha * (p1 - p2)
    s2 = p2 + alpha * (p1 - p2)

    s1 = np.clip(s1, LOW, HIGH)
    s2 = np.clip(s2, LOW, HIGH)

    return s1, s2



def two_point_crossover(p1, p2):
    size = p1.size
    point1, point2 = np.sort(np.random.choice(size, 2, replace=False))

    s1 = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
    s2 = np.concatenate([p2[:point1], p1[point1:point2], p2[point2:]])

    return s1, s2



def mutation(individual, mutation_rate=0.2, sigma=10.0):
    mutated_individual = []
    for gene in individual:
        if np.random.rand() < mutation_rate:
            delta = np.random.normal(0.0, sigma)
            mutated_individual.append(np.clip(gene + delta, LOW, HIGH))
        else:
            mutated_individual.append(gene)
    return np.array(mutated_individual)




def tournament(population, fitness, k=3):
    idx = np.random.choice(len(population), k, replace=False)   
    winner = idx[np.argmin(fitness[idx])]                         
    return population[winner]





def next_generation(population, fitness_scores, objective, crossover=two_point_crossover):
    N = len(population)

    children = []

    while len(children) < N:
        operador = np.random.choice(['crossover', 'mutation'], p=[0.7, 0.3])
        if operador == 'crossover':
            # Select two parents by tournament for crossover
            parent1 = tournament(population, fitness_scores)
            parent2 = tournament(population, fitness_scores)
            s1, s2 = crossover(parent1, parent2)
            children.append(s1)
            if len(children) < N:
                children.append(s2)
        elif operador == 'mutation':
            # Select one parent by tournament for mutation
            parent1 = tournament(population, fitness_scores)
            mutated_child = mutation(parent1)
            children.append(mutated_child)

    children = np.array(children)
    children_fitness = np.array([objective(t) for t in children])

    
    all_population = np.vstack([population, children])
    all_fitness = np.concatenate([fitness_scores, children_fitness])
    best = np.argsort(all_fitness)[:N]

    return all_population[best], all_fitness[best]



def run_ga(N, d, seed, budget=60000, crossover=two_point_crossover):
    np.random.seed(seed)

    # Initial population (N evaluations)
    population = np.random.uniform(LOW, HIGH, (N, d))
    fitness = np.array([objective(t) for t in population])
    evaluations = N

    eval_history = [evaluations]
    best_history = [fitness.min()]

    # Each generation uses N evaluations (one per child)
    while evaluations + N <= budget:
        population, fitness = next_generation(population, fitness, objective, crossover)
        evaluations += N
        eval_history.append(evaluations)
        best_history.append(fitness.min())

    return population[0], fitness[0], evaluations, np.array(eval_history), np.array(best_history)



if __name__ == "__main__":

    if len(sys.argv) != 2 or sys.argv[1] not in ("b", "c"):
        print("Uso: python genetica.py [b|c]")
        sys.exit(1)

    item = sys.argv[1]

    D = 13
    BUDGET = 60000

    if item == "b":
        N_VALUES = [100, 200, 400]
        SEEDS = [0, 1, 2]

        results = {}

        for N_pop in N_VALUES:
            results[N_pop] = []
            for seed in SEEDS:
                t_best, J_best, n_evals, evals, best = run_ga(N_pop, D, seed, BUDGET)
                results[N_pop].append((t_best, J_best, evals, best))
                print(f"N = {N_pop:3d}  semilla = {seed}  evaluaciones = {n_evals}  J = {J_best:.6f}")
                print(f"    t = {np.round(t_best, 2)}")

        print()
        print("Resumen por N:")
        for N_pop in N_VALUES:
            Js = np.array([r[1] for r in results[N_pop]])
            print(f"N = {N_pop:3d}  mejor J = {Js.min():.6f}  promedio = {Js.mean():.6f}  desv. est. = {Js.std():.6f}")

        # Convergence curves: best J vs number of evaluations
        colors = {100: "tab:blue", 200: "tab:orange", 400: "tab:green"}
        for N_pop in N_VALUES:
            for k, (t_best, J_best, evals, best) in enumerate(results[N_pop]):
                plt.plot(evals, best, color=colors[N_pop], alpha=0.8,
                         label=f"N = {N_pop}" if k == 0 else None)
        plt.xlabel("Evaluaciones de J")
        plt.ylabel("Mejor J")
        plt.yscale("log")
        plt.title("Convergencia del algoritmo genético")
        plt.legend()
        plt.show()

    elif item == "c":
        N_POP = 200
        SEEDS = [0, 1, 2, 3, 4]
        OPERATORS = {
            "Dos puntos": two_point_crossover,
            "Blend (α = 0.3)": lambda p1, p2: blended_crossover(p1, p2, alpha=0.3),
        }

        results = {}

        for name, crossover in OPERATORS.items():
            results[name] = []
            for seed in SEEDS:
                t_best, J_best, n_evals, evals, best = run_ga(N_POP, D, seed, BUDGET, crossover)
                results[name].append((t_best, J_best, evals, best))
                print(f"{name:16s}  semilla = {seed}  evaluaciones = {n_evals}  J = {J_best:.6f}")
                print(f"    t = {np.round(t_best, 2)}")

        print()
        print("Resumen por operador de cruce:")
        for name in OPERATORS:
            Js = np.array([r[1] for r in results[name]])
            print(f"{name:16s}  promedio = {Js.mean():.6f}  desv. est. = {Js.std(ddof=1):.6f}  mejor J = {Js.min():.6f}")

        # Mean best-so-far J (± one sample std) vs number of evaluations
        colors = {"Dos puntos": "tab:blue", "Blend (α = 0.3)": "tab:red"}
        for name in OPERATORS:
            evals = results[name][0][2]
            bests = np.array([r[3] for r in results[name]])
            mean = bests.mean(axis=0)
            std = bests.std(axis=0, ddof=1)
            plt.plot(evals, mean, color=colors[name], label=name)
            plt.fill_between(evals, mean - std, mean + std, color=colors[name], alpha=0.25)
        plt.xlabel("Evaluaciones de J")
        plt.ylabel("Mejor J histórico (promedio)")
        plt.yscale("log")
        plt.title(f"Comparación de operadores de cruce (N = {N_POP})")
        plt.legend()
        plt.show()





