import os
# Un hilo de BLAS por proceso: las matrices son chicas y el paralelismo viene de multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt
import sys

from bridge_ga_model import SensorPlacement
from bridge_akf import evaluate_design_cases
from parallel_eval import make_pool, evaluate_designs

LOW = 0

HIGH = 60


EMPTY = -1.0  # Valor del gen cuando el sensor no esta presente

def decode(individual, sensor_types=1):
    # Convierte un individuo en la lista de SensorPlacement, saltandose los EMPTY.
    # sensor_types: un int (mismo tipo para todos) o un arreglo de largo num_genes con el tipo de cada gen.
    types =np.broadcast_to(sensor_types, individual.shape)
    return [SensorPlacement(position_m=float(pos), sensor_type=int(t))
            for pos, t in zip(individual, types) if pos != EMPTY]


def first_generation(population_size, num_genes, low, high):
    # Cada individuo es un vector de largo fijo num_genes.
    # Gen = posicion del sensor (float en [low, high]) o EMPTY si no hay sensor.
    population = np.full((population_size, num_genes), EMPTY)
    for i in range(population_size):
        # Numero de sensores uniforme entre 0 y num_genes, para cubrir todos los tamaños de red
        n_sensors = np.random.randint(0, num_genes + 1)
        active = np.random.choice(num_genes, n_sensors, replace=False)
        population[i, active] = np.random.uniform(low, high, n_sensors)
    return population




def blended_crossover(p1, p2, alpha=0.3):
    s1 = np.full(p1.size, EMPTY)
    s2 = np.full(p2.size, EMPTY)

    for i in range(p1.size):
        on1 = p1[i] != EMPTY
        on2 = p2[i] != EMPTY

        if on1 and on2:
            # Ambos sensores presentes: blend normal
            s1[i] = np.clip(p1[i] - alpha * (p1[i] - p2[i]), LOW, HIGH)
            s2[i] = np.clip(p2[i] + alpha * (p1[i] - p2[i]), LOW, HIGH)
        elif on1 or on2:
            # Solo uno presente: un hijo hereda el sensor y el otro queda vacio
            # (se conserva la cantidad total de sensores entre los dos hijos)
            pos = p1[i] if on1 else p2[i]
            if np.random.rand() < 0.5:
                s1[i] = pos
            else:
                s2[i] = pos
        # Ambos vacios: los dos hijos quedan en EMPTY

    return s1, s2



def two_point_crossover(p1, p2):
    # Solo intercambia genes completos, por lo que los EMPTY se mantienen validos
    size = p1.size
    point1, point2 = np.sort(np.random.choice(size, 2, replace=False))

    s1 = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
    s2 = np.concatenate([p2[:point1], p1[point1:point2], p2[point2:]])

    return s1, s2



def mutation(individual, mutation_rate=0.2, sigma=10.0, p_off=0.4):
    # p_off: probabilidad de que un sensor presente que muta se elimine (pase a EMPTY)
    mutated_individual = []
    for gene in individual:
        if np.random.rand() < mutation_rate:
            if gene == EMPTY:
                # Sensor ausente: se activa en una posicion aleatoria
                mutated_individual.append(np.random.uniform(LOW, HIGH))
            elif np.random.rand() < p_off:
                mutated_individual.append(EMPTY)
            else:
                delta = np.random.normal(0.0, sigma)
                mutated_individual.append(np.clip(gene + delta, LOW, HIGH))
        else:
            mutated_individual.append(gene)
    return np.array(mutated_individual)




def tournament(population, fitness, k=3):
    idx = np.random.choice(len(population), k, replace=False)   
    winner = idx[np.argmin(fitness[idx])]                         
    return population[winner]





def next_generation(population, fitness_scores, objective, crossover=two_point_crossover,
                    p_crossover=0.7, mutation_kwargs=None):
    # objective(individuo) -> float a minimizar. El individuo llega codificado (con EMPTY),
    # por lo que objective debe llamar a decode() antes de evaluar.
    N = len(population)
    mutation_kwargs = mutation_kwargs or {}

    children = []

    while len(children) < N:
        operador = np.random.choice(['crossover', 'mutation'], p=[p_crossover, 1 - p_crossover])
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
            mutated_child = mutation(parent1, **mutation_kwargs)
            children.append(mutated_child)

    children = np.array(children)
    objective.evaluate_many(children)  # Evalua en paralelo los hijos que no estan en cache
    children_fitness = np.array([objective(t) for t in children])

    
    all_population = np.vstack([population, children])
    all_fitness = np.concatenate([fitness_scores, children_fitness])
    best = np.argsort(all_fitness)[:N]

    return all_population[best], all_fitness[best]





D_MIN = 3.0  # Distancia minima entre sensores [m]


def violations(individual, d_min=D_MIN):
    # v_j(x) = max{0, g_j(x)}, con g_j(x) = d_min - (x_{j+1} - x_j) para cada par de sensores
    # consecutivos (ordenados por posicion). Si los consecutivos cumplen, todos los pares cumplen.
    pos = np.sort(individual[individual != EMPTY])
    if pos.size < 2:
        return np.zeros(0)
    return np.maximum(0.0, d_min - np.diff(pos))


def penalty(individual, gen, c=1.0, alpha=1.0, beta=2.0):
    # Termino dependiente de la generacion: (c * gen)^alpha * sum_j v_j(x)^beta
    v = violations(individual)
    return (c * gen) ** alpha * np.sum(v ** beta)


class Objective:
    # f'(x) = f(x) + penalty(x, gen), con f(x) = error medio de estimacion [%].
    # f(x) es caro (simula el filtro de Kalman), asi que se guarda en cache: la evaluacion es
    # deterministica (semillas fijas), y al cambiar de generacion solo se recalcula la penalizacion.

    def __init__(self, bridge, network, cases, sensor_type, penalty_kwargs=None, pool=None):
        self.bridge = bridge
        self.network = network
        self.cases = cases
        self.sensor_type = sensor_type
        self.penalty_kwargs = penalty_kwargs or {}
        self.pool = pool  # multiprocessing.Pool (de make_pool) o None para evaluar en serie
        self.gen = 1
        self.cache = {}

    @staticmethod
    def key(individual):
        return tuple(np.sort(individual[individual != EMPTY]))

    def evaluate_many(self, population):
        # Llena el cache evaluando en paralelo los individuos que aun no estan (sin repetidos)
        if self.pool is None:
            return
        pending = {}
        for ind in population:
            key = self.key(ind)
            if key not in self.cache and key not in pending:
                pending[key] = decode(ind, self.sensor_type)
        if pending:
            results = evaluate_designs(self.pool, list(pending.values()))
            for key, (err, _) in zip(pending, results):
                self.cache[key] = err

    def error(self, individual):
        key = self.key(individual)
        if key not in self.cache:
            out = evaluate_design_cases(self.bridge, self.network, decode(individual, self.sensor_type),
                                        self.cases, evaluate_failures=False)
            self.cache[key] = out['mean_error_pct']
        return self.cache[key]

    def __call__(self, individual):
        return self.error(individual) + penalty(individual, self.gen, **self.penalty_kwargs)


def run_ga(objective, population_size=20, num_genes=15, generations=15, crossover=two_point_crossover,
           p_crossover=0.7, mutation_kwargs=None, verbose=True):
    population = first_generation(population_size, num_genes, LOW, HIGH)
    history = {'best_fitness': [], 'best_error': [], 'n_sensors': []}

    for gen in range(1, generations + 1):
        # La penalizacion cambia con la generacion: se re-evalua a los padres (solo la penalizacion
        # es nueva, el error sale del cache) para que compitan en igualdad con los hijos.
        objective.gen = gen
        objective.evaluate_many(population)
        fitness = np.array([objective(ind) for ind in population])

        population, fitness = next_generation(population, fitness, objective, crossover=crossover,
                                              p_crossover=p_crossover, mutation_kwargs=mutation_kwargs)

        best = population[0]
        history['best_fitness'].append(fitness[0])
        history['best_error'].append(objective.error(best))
        history['n_sensors'].append(int(np.sum(best != EMPTY)))
        if verbose:
            print(f"  gen {gen:3d} | f' = {fitness[0]:8.3f} | error = {history['best_error'][-1]:8.3f} % "
                  f"| sensores = {history['n_sensors'][-1]:2d} | violacion = {np.sum(violations(best)):.3f} m "
                  f"| evaluaciones = {len(objective.cache)}")

    return population[0], history




if __name__ == "__main__":
    import time
    from bridge_ga_model import ThreeSpanBridgeFE, SensorNetwork
    from bridge_akf import PositionUncertainty, sample_truck_scenarios, prepare_cases
    from bridge_visualizations import plot_bridge_and_sensor_network

    np.random.seed(0)

    nominal_bridge = ThreeSpanBridgeFE()
    network = SensorNetwork(nominal_bridge)

    scenarios = sample_truck_scenarios(n=5, seed=123)
    pos_unc = PositionUncertainty(common_bias_std_m=0.015, spacing_error_std_m=0.005,
                                  jitter_std_m=0.03, correlation_time_s=0.02)  # Do not change
    cases = prepare_cases(nominal_bridge, scenarios, position_uncertainty=pos_unc)

    SENSOR_TYPES = {1: 'Aceleracion', 2: 'Deformacion', 3: 'Desplazamiento'}
    GA_PARAMS = dict(population_size=20, num_genes=15, generations=15, crossover=two_point_crossover,
                     p_crossover=0.7, mutation_kwargs=dict(mutation_rate=0.2, sigma=10.0, p_off=0.4))
    PENALTY = dict(c=1.0, alpha=1.0, beta=2.0)

    results = {}
    with make_pool(nominal_bridge, network, cases) as pool:
        for sensor_type, name in SENSOR_TYPES.items():
            print(f"\n=== Optimizacion sensor tipo {sensor_type} ({name}) ===")
            t0 = time.time()
            objective = Objective(nominal_bridge, network, cases, sensor_type, penalty_kwargs=PENALTY, pool=pool)
            best, history = run_ga(objective, **GA_PARAMS)
            results[sensor_type] = (best, history)

            design = decode(best, sensor_type)
            print(f"  Tiempo: {time.time() - t0:.1f} s")
            print(f"  Mejor error: {objective.error(best):.3f} % con {len(design)} sensores")
            print(f"  Posiciones [m]: {np.round(sorted(p.position_m for p in design), 2)}")
            print(f"  Cumple distancia minima: {np.sum(violations(best)) == 0}")

            plot_bridge_and_sensor_network(nominal_bridge, network, design)
            plt.suptitle(f"Mejor diseño - {name}")

    # Convergencia
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sensor_type, (best, history) in results.items():
        gens = np.arange(1, len(history['best_error']) + 1)
        ax[0].plot(gens, history['best_error'], marker='o', label=SENSOR_TYPES[sensor_type])
        ax[1].plot(gens, history['n_sensors'], marker='o', label=SENSOR_TYPES[sensor_type])
    ax[0].set_xlabel('Generacion'); ax[0].set_ylabel('Error medio del mejor individuo [%]'); ax[0].legend()
    ax[1].set_xlabel('Generacion'); ax[1].set_ylabel('Numero de sensores'); ax[1].legend()
    fig.tight_layout()

    plt.show()










