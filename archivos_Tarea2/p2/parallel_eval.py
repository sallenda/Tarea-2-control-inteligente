from multiprocessing import Pool

from bridge_akf import evaluate_design_cases

# Evaluacion de disenos en paralelo con multiprocessing.
# El puente, la red y los casos (~7 MB) se envian una sola vez a cada proceso (initializer),
# y luego a cada tarea solo se le pasa el diseno (lista de SensorPlacement).

_bridge = None
_network = None
_cases = None


def _init_worker(bridge, network, cases):
    global _bridge, _network, _cases
    _bridge, _network, _cases = bridge, network, cases


def _evaluate(design):
    out = evaluate_design_cases(_bridge, _network, design, _cases, evaluate_failures=False)
    return out['mean_error_pct'], out['cost']


def make_pool(bridge, network, cases, processes=None):
    # processes=None => usa todos los nucleos (os.cpu_count())
    return Pool(processes, initializer=_init_worker, initargs=(bridge, network, cases))


def evaluate_designs(pool, designs):
    # Lista de (error, costo), en el mismo orden que designs
    return pool.map(_evaluate, designs, chunksize=1)
