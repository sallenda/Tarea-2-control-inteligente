import os
# Un hilo de BLAS por proceso: las matrices son chicas y el paralelismo viene de multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import matplotlib.pyplot as plt

from bridge_ga_model import SensorPlacement
from bridge_akf import evaluate_design_cases
from parallel_eval import make_pool, evaluate_designs

LOW = 0

HIGH = 60


EMPTY = -1.0  # Valor del gen cuando el sensor no esta presente (en posicion y tipo)

D_MIN = 3.0  # Distancia minima entre sensores [m]

SENSOR_TYPES = (1, 2, 3)  # 1: aceleracion, 2: deformacion, 3: desplazamiento

POS, TYPE = 0, 1  # Columnas de cada gen


class NSGA_II(object):
    # Implementacion de NSGA-II para problemas de optimizacion multi-objetivo.
    # Cada individuo es una matriz (num_genes, 2) (=> maximo num_genes sensores).
    # Gen = [posicion, tipo]: posicion float en [low, high] y tipo en sensor_types,
    #       o [EMPTY, EMPTY] si no hay sensor.
    # Objetivos (ambos a minimizar): f1 = error medio de estimacion [%], f2 = costo de la red [$].
    # Restriccion de distancia minima: se maneja con dominancia restringida (Deb, 2002).

    def __init__(self, bridge, network, cases, sensor_types=SENSOR_TYPES, population_size=20, num_genes=15,
                 low=LOW, high=HIGH, d_min=D_MIN, crossover='two_point', p_crossover=0.7,
                 mutation_kwargs=None, tournament_k=2, pool=None):
        self.bridge = bridge
        self.network = network
        self.cases = cases
        self.sensor_types = np.array(sensor_types)
        self.population_size = population_size
        self.num_genes = num_genes
        self.low = low
        self.high = high
        self.d_min = d_min
        self.crossover = {'two_point': self.two_point_crossover,
                          'blended': self.blended_crossover}[crossover]
        self.p_crossover = p_crossover
        self.mutation_kwargs = mutation_kwargs or {}
        self.tournament_k = tournament_k
        self.pool = pool  # multiprocessing.Pool (de make_pool) o None para evaluar en serie
        self.cache = {}  # (pares (posicion, tipo) ordenados por posicion) -> (error, costo)

    # -----------------------------------------------------------------------
    # Codificacion
    # -----------------------------------------------------------------------

    @staticmethod
    def active(individual):
        # Mascara de los genes con sensor presente
        return individual[:, POS] != EMPTY

    def random_gene(self):
        return np.array([np.random.uniform(self.low, self.high), np.random.choice(self.sensor_types)])

    def decode(self, individual):
        # Convierte un individuo en la lista de SensorPlacement, saltandose los EMPTY.
        return [SensorPlacement(position_m=float(pos), sensor_type=int(t))
                for pos, t in individual[self.active(individual)]]

    def key(self, individual):
        # El orden de los genes no importa: dos individuos con los mismos (posicion, tipo) son la misma red
        genes = individual[self.active(individual)]
        genes = genes[np.argsort(genes[:, POS])]
        return tuple(map(tuple, genes))

    def ensure_sensor(self, individual):
        # Una red vacia tiene error infinito: si el individuo quedo sin sensores, se activa uno al azar
        if not np.any(self.active(individual)):
            individual = individual.copy()
            individual[np.random.randint(self.num_genes)] = self.random_gene()
        return individual

    def first_generation(self):
        population = np.full((self.population_size, self.num_genes, 2), EMPTY)
        for i in range(self.population_size):
            # Numero de sensores uniforme entre 1 y num_genes, para cubrir todos los tamaños de red
            n_sensors = np.random.randint(1, self.num_genes + 1)
            active = np.random.choice(self.num_genes, n_sensors, replace=False)
            population[i, active, POS] = np.random.uniform(self.low, self.high, n_sensors)
            population[i, active, TYPE] = np.random.choice(self.sensor_types, n_sensors)
        return population

    # -----------------------------------------------------------------------
    # Operadores geneticos
    # -----------------------------------------------------------------------

    def blended_crossover(self, p1, p2, alpha=0.3):
        s1 = np.full(p1.shape, EMPTY)
        s2 = np.full(p2.shape, EMPTY)

        for i in range(len(p1)):
            on1 = p1[i, POS] != EMPTY
            on2 = p2[i, POS] != EMPTY

            if on1 and on2:
                # Ambos sensores presentes: blend normal en la posicion
                x1, x2 = p1[i, POS], p2[i, POS]
                s1[i, POS] = np.clip(x1 - alpha * (x1 - x2), self.low, self.high)
                s2[i, POS] = np.clip(x2 + alpha * (x1 - x2), self.low, self.high)
                # El tipo es discreto: cruzamiento uniforme (cada hijo hereda el tipo de uno de los padres)
                if np.random.rand() < 0.5:
                    s1[i, TYPE], s2[i, TYPE] = p1[i, TYPE], p2[i, TYPE]
                else:
                    s1[i, TYPE], s2[i, TYPE] = p2[i, TYPE], p1[i, TYPE]
            elif on1 or on2:
                # Solo uno presente: un hijo hereda el sensor (posicion y tipo) y el otro queda vacio
                # (se conserva la cantidad total de sensores entre los dos hijos)
                gene = p1[i] if on1 else p2[i]
                if np.random.rand() < 0.5:
                    s1[i] = gene
                else:
                    s2[i] = gene
            # Ambos vacios: los dos hijos quedan en EMPTY

        return s1, s2

    def two_point_crossover(self, p1, p2):
        # Intercambia genes completos (posicion y tipo juntos), por lo que los EMPTY se mantienen validos
        size = len(p1)
        point1, point2 = np.sort(np.random.choice(size, 2, replace=False))

        s1 = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
        s2 = np.concatenate([p2[:point1], p1[point1:point2], p2[point2:]])

        return s1, s2

    def mutation(self, individual, mutation_rate=0.2, sigma=10.0, p_off=0.4, p_type=0.3):
        # p_off: probabilidad de que un sensor presente que muta se elimine (pase a EMPTY)
        # p_type: probabilidad de que un sensor presente que muta (y no se elimina) cambie de tipo
        mutated_individual = individual.copy()
        for i, gene in enumerate(individual):
            if np.random.rand() < mutation_rate:
                if gene[POS] == EMPTY:
                    # Sensor ausente: se activa en una posicion aleatoria con un tipo aleatorio
                    mutated_individual[i] = self.random_gene()
                elif np.random.rand() < p_off:
                    mutated_individual[i] = EMPTY
                else:
                    delta = np.random.normal(0.0, sigma)
                    mutated_individual[i, POS] = np.clip(gene[POS] + delta, self.low, self.high)
                    if np.random.rand() < p_type:
                        # Cambia a uno de los otros tipos disponibles
                        mutated_individual[i, TYPE] = np.random.choice(self.sensor_types[self.sensor_types != gene[TYPE]])
        return mutated_individual

    # -----------------------------------------------------------------------
    # Restriccion de distancia minima
    # -----------------------------------------------------------------------

    def violations(self, individual):
        # v_j(x) = max{0, g_j(x)}, con g_j(x) = d_min - (x_{j+1} - x_j) para cada par de sensores
        # consecutivos (ordenados por posicion). Si los consecutivos cumplen, todos los pares cumplen.
        # La distancia minima aplica entre todos los sensores, sin importar su tipo.
        pos = np.sort(individual[self.active(individual), POS])
        if pos.size < 2:
            return np.zeros(0)
        return np.maximum(0.0, self.d_min - np.diff(pos))

    def total_violation(self, individual):
        return float(np.sum(self.violations(individual)))

    # -----------------------------------------------------------------------
    # Evaluacion
    # -----------------------------------------------------------------------

    def evaluate(self, individual):
        # (error, costo) del individuo. Es caro (simula el filtro de Kalman), asi que se guarda en
        # cache: la evaluacion es deterministica (semillas fijas).
        key = self.key(individual)
        if key not in self.cache:
            out = evaluate_design_cases(self.bridge, self.network, self.decode(individual),
                                        self.cases, evaluate_failures=False)
            self.cache[key] = (out['mean_error_pct'], out['cost'])
        return self.cache[key]

    def evaluate_many(self, population):
        # Llena el cache evaluando en paralelo los individuos que aun no estan (sin repetidos)
        if self.pool is None:
            return
        pending = {}
        for ind in population:
            key = self.key(ind)
            if key not in self.cache and key not in pending:
                pending[key] = self.decode(ind)
        if pending:
            results = evaluate_designs(self.pool, list(pending.values()))
            self.cache.update(zip(pending, results))

    def evaluate_population(self, population):
        # F: (N, 2) con los objetivos [error, costo]; CV: (N,) con la violacion total de la restriccion
        self.evaluate_many(population)
        F = np.array([self.evaluate(ind) for ind in population])
        CV = np.array([self.total_violation(ind) for ind in population])
        return F, CV

    # -----------------------------------------------------------------------
    # a) Ranking de Pareto (calidad)
    # -----------------------------------------------------------------------

    @staticmethod
    def dominates(Fi, Fj, cvi, cvj):
        # Dominancia restringida:
        #   - una solucion factible domina a una infactible
        #   - entre dos infactibles, domina la de menor violacion
        #   - entre dos factibles, dominancia de Pareto usual (no peor en todo y mejor en al menos uno)
        if cvi == 0 and cvj > 0:
            return True
        if cvi > 0 and cvj == 0:
            return False
        if cvi > 0 and cvj > 0:
            return cvi < cvj
        return bool(np.all(Fi <= Fj) and np.any(Fi < Fj))

    def pareto_ranking(self, F, CV):
        # Para cada iteracion i = 1, 2, ...:
        #   1. Encontrar las soluciones no dominadas entre las que quedan
        #   2. Asignarles ranking i
        #   3. Removerlas
        # Retorna ranks (N,) y la lista de frentes (indices de cada ranking).
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        remaining = list(range(N))
        fronts = []
        i = 1
        while remaining:
            front = [p for p in remaining
                     if not any(self.dominates(F[q], F[p], CV[q], CV[p]) for q in remaining if q != p)]
            ranks[front] = i
            fronts.append(front)
            remaining = [p for p in remaining if p not in front]
            i += 1
        return ranks, fronts

    # -----------------------------------------------------------------------
    # b) Crowding distance (diversidad)
    # -----------------------------------------------------------------------

    @staticmethod
    def crowding_distance(F):
        # F: (n, m) objetivos de un conjunto de soluciones (generalmente un mismo frente).
        # c^j = sum_k (f_k^{j+1} - f_k^{j-1}) / (f_k^max - f_k^min), con los f_k ordenados ascendente.
        # Las soluciones borde reciben crowding infinito para conservar la extension del frente.
        n, m = F.shape
        distance = np.zeros(n)
        if n <= 2:
            distance[:] = np.inf
            return distance

        for k in range(m):
            order = np.argsort(F[:, k])
            f = F[order, k]
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf
            f_range = f[-1] - f[0]
            if f_range == 0:
                continue  # todos iguales en este objetivo: no aporta informacion
            distance[order[1:-1]] += (f[2:] - f[:-2]) / f_range
        return distance

    def rank_and_crowding(self, F, CV):
        ranks, fronts = self.pareto_ranking(F, CV)
        crowding = np.zeros(len(F))
        for front in fronts:
            crowding[front] = self.crowding_distance(F[front])
        return ranks, crowding, fronts

    # -----------------------------------------------------------------------
    # c) Archive (elitismo)
    # -----------------------------------------------------------------------

    def create_archive(self, population, F, CV):
        # Llena el Archive de tamaño population_size en orden de mejor a peor solucion:
        #   1. menor ranking de Pareto
        #   2. a igual ranking, mayor crowding distance
        # Antes se eliminan los individuos repetidos (misma red), para no perder diversidad.
        _, unique = np.unique([str(self.key(ind)) for ind in population], return_index=True)
        unique = np.sort(unique)
        population, F, CV = population[unique], F[unique], CV[unique]

        ranks, crowding, _ = self.rank_and_crowding(F, CV)
        order = np.lexsort((-crowding, ranks))[:self.population_size]  # lexsort: ultima llave = principal
        return population[order], F[order], CV[order], ranks[order], crowding[order]

    # -----------------------------------------------------------------------
    # Seleccion y reproduccion
    # -----------------------------------------------------------------------

    def tournament(self, population, ranks, crowding):
        # Torneo de comparacion por ranking y crowding: gana menor ranking, y a igual ranking mayor crowding
        idx = np.random.choice(len(population), self.tournament_k, replace=False)
        winner = idx[np.lexsort((-crowding[idx], ranks[idx]))[0]]
        return population[winner]

    def make_children(self, archive, ranks, crowding):
        # P(t+1): hijos generados desde el Archive con torneo, cruzamiento y mutacion
        N = self.population_size
        children = []

        while len(children) < N:
            operador = np.random.choice(['crossover', 'mutation'], p=[self.p_crossover, 1 - self.p_crossover])
            if operador == 'crossover':
                parent1 = self.tournament(archive, ranks, crowding)
                parent2 = self.tournament(archive, ranks, crowding)
                s1, s2 = self.crossover(parent1, parent2)
                children.append(self.ensure_sensor(s1))
                if len(children) < N:
                    children.append(self.ensure_sensor(s2))
            elif operador == 'mutation':
                parent1 = self.tournament(archive, ranks, crowding)
                children.append(self.ensure_sensor(self.mutation(parent1, **self.mutation_kwargs)))

        return np.array(children)

    # -----------------------------------------------------------------------
    # Loop principal
    # -----------------------------------------------------------------------

    def run(self, generations=15, verbose=True):
        population = self.first_generation()
        archive = np.empty((0, self.num_genes, 2))
        history = {'n_front': [], 'min_error': [], 'min_cost': [], 'evaluations': []}

        for gen in range(1, generations + 1):
            # A(t) U P(t): se evaluan todas las soluciones (las del Archive salen del cache)
            combined = np.vstack([archive, population])
            F, CV = self.evaluate_population(combined)

            # A(t+1) se llena por ranking de Pareto y crowding distance
            archive, F_a, CV_a, ranks_a, crowding_a = self.create_archive(combined, F, CV)

            # P(t+1) se genera desde A(t+1)
            population = self.make_children(archive, ranks_a, crowding_a)

            front = (ranks_a == 1) & (CV_a == 0)
            history['n_front'].append(int(np.sum(front)))
            history['min_error'].append(float(np.min(F_a[front, 0])) if front.any() else np.nan)
            history['min_cost'].append(float(np.min(F_a[front, 1])) if front.any() else np.nan)
            history['evaluations'].append(len(self.cache))
            if verbose:
                print(f"  gen {gen:3d} | frente = {history['n_front'][-1]:2d} sol. "
                      f"| error min = {history['min_error'][-1]:7.3f} % "
                      f"| costo min = {history['min_cost'][-1]:8.0f} "
                      f"| evaluaciones = {history['evaluations'][-1]}")

        # La ultima poblacion de hijos tambien compite por entrar al Archive final
        combined = np.vstack([archive, population])
        F, CV = self.evaluate_population(combined)
        self.archive, self.F, self.CV, self.ranks, self.crowding = self.create_archive(combined, F, CV)
        return self.pareto_front(), history

    def pareto_front(self):
        # Soluciones factibles con ranking 1 del Archive final, ordenadas por costo
        front = np.where((self.ranks == 1) & (self.CV == 0))[0]
        front = front[np.argsort(self.F[front, 1])]
        return self.archive[front], self.F[front]


def select_representatives(F):
    # F: frente de Pareto ordenado por costo. Retorna los indices de:
    #   - bajo costo: la de menor costo
    #   - bajo error: la de menor error
    #   - compromiso: la mas cercana al punto utopico en objetivos normalizados a [0, 1]
    Fn = (F - F.min(axis=0)) / np.where(np.ptp(F, axis=0) > 0, np.ptp(F, axis=0), 1.0)
    i_cost = int(np.argmin(F[:, 1]))
    i_error = int(np.argmin(F[:, 0]))
    i_comp = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    return {'Bajo costo': i_cost, 'Compromiso': i_comp, 'Bajo error': i_error}


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

    NSGA_PARAMS = dict(sensor_types=SENSOR_TYPES, population_size=20, num_genes=15, d_min=D_MIN,
                       crossover='two_point', p_crossover=0.7,
                       mutation_kwargs=dict(mutation_rate=0.2, sigma=10.0, p_off=0.4, p_type=0.3))
    TYPE_NAMES = {1: 'Acel', 2: 'Def', 3: 'Desp'}

    print("\n=== NSGA-II red heterogenea (tipos 1, 2 y 3) ===")
    t0 = time.time()
    with make_pool(nominal_bridge, network, cases) as pool:
        nsga = NSGA_II(nominal_bridge, network, cases, pool=pool, **NSGA_PARAMS)
        (front_pop, front_F), history = nsga.run(generations=15)
    print(f"  Tiempo: {time.time() - t0:.1f} s")

    def describe(individual):
        # Cantidad de sensores de cada tipo, ej: "2 Acel, 0 Def, 3 Desp"
        types = individual[nsga.active(individual), TYPE]
        return ", ".join(f"{int(np.sum(types == t))} {TYPE_NAMES[t]}" for t in SENSOR_TYPES)

    print("\n  Frente de Pareto:")
    for ind, (err, cost) in zip(front_pop, front_F):
        print(f"    costo = {cost:8.0f} | error = {err:7.3f} % | sensores = {int(np.sum(nsga.active(ind))):2d} "
              f"({describe(ind)})")

    reps = select_representatives(front_F)

    # Frente de Pareto, comparado con el de la red homogenea (p2_b.py) si ya fue calculado
    homog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pareto_homogeneo.npz')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nsga.F[:, 1], nsga.F[:, 0], c='lightgray', label='Archive final')
    ax.plot(front_F[:, 1], front_F[:, 0], 'o-', label='Frente heterogeneo (tipos 1, 2, 3)')
    if os.path.exists(homog_path):
        front_F_homog = np.load(homog_path)['F']
        ax.plot(front_F_homog[:, 1], front_F_homog[:, 0], 's--', label='Frente homogeneo (tipo 3)')
    else:
        print(f"\n  No se encontro {homog_path}: corre p2_b.py para compararlo con la red homogenea")
    for (name, i), marker in zip(reps.items(), ['s', 'D', '^']):
        ax.scatter(front_F[i, 1], front_F[i, 0], s=120, marker=marker, zorder=3, label=name)
    ax.set_xlabel('Costo total de la red [$]')
    ax.set_ylabel('Error medio de estimacion [%]')
    ax.set_title('Frente de Pareto - NSGA-II (red heterogenea vs homogenea)')
    ax.legend()
    fig.tight_layout()

    # Soluciones representativas
    for name, i in reps.items():
        design = sorted(nsga.decode(front_pop[i]), key=lambda p: p.position_m)
        print(f"\n  {name}: costo = {front_F[i, 1]:.0f} | error = {front_F[i, 0]:.3f} % "
              f"| {len(design)} sensores ({describe(front_pop[i])})")
        print(f"    Posiciones [m]: {np.round([p.position_m for p in design], 2)}")
        print(f"    Tipos:          {[p.sensor_type for p in design]}")
        print(f"    Cumple distancia minima: {nsga.total_violation(front_pop[i]) == 0}")
        plot_bridge_and_sensor_network(nominal_bridge, network, design)
        plt.suptitle(f"{name}: costo = {front_F[i, 1]:.0f}, error = {front_F[i, 0]:.3f} %")

    # Convergencia
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    gens = np.arange(1, len(history['min_error']) + 1)
    ax[0].plot(gens, history['min_error'], marker='o')
    ax[0].set_xlabel('Generacion'); ax[0].set_ylabel('Error minimo del frente [%]')
    ax[1].plot(gens, history['n_front'], marker='o')
    ax[1].set_xlabel('Generacion'); ax[1].set_ylabel('Soluciones en el frente')
    fig.tight_layout()

    plt.show()
