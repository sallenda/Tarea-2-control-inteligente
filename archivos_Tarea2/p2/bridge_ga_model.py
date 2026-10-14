from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Iterable, Union
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve


@dataclass(frozen=True)
class TruckScenario:
    """Two-axle truck moving from x=0 to x=L at constant speed."""
    axle_loads_N: Tuple[float, float] = (80e3, 120e3)
    axle_spacing_m: float = 4.0
    speed_m_s: float = 20.0


@dataclass(frozen=True)
class SensorSpec:
    name: str
    cost: float
    relative_noise: float


SENSOR_SPECS: Dict[int, SensorSpec] = {
    1: SensorSpec("accelerometer", 650.0, 0.04),
    2: SensorSpec("strain", 1100.0, 0.02),
    3: SensorSpec("displacement", 900.0, 0.01),
}


@dataclass(frozen=True)
class SensorPlacement:
    position_m: float
    sensor_type: int

SensorGene = Union[SensorPlacement, Tuple[float, int], List[float], Dict[str, float]]


@dataclass(frozen=True)
class PlantPerturbation:
    E_scale: float = 1.0
    damping_ratio_scale: float = 1.0


def sample_plant_perturbation(seed: Optional[int] = None, E_std_fraction: float = 0.03, damping_std_fraction: float = 0.15,
    min_scale: float = 0.70, max_scale: float = 1.30,) -> PlantPerturbation:

    rng = np.random.default_rng(seed)
    E_scale = np.clip(1.0 + rng.normal(0.0, E_std_fraction), min_scale, max_scale)
    damping_scale = np.clip(1.0 + rng.normal(0.0, damping_std_fraction), min_scale, max_scale)
    return PlantPerturbation(float(E_scale), float(damping_scale))



class ThreeSpanBridgeFE:

    def __init__(self, spans_m: Sequence[float] = (20.0, 20.0, 20.0), elements_per_span: int = 4,
        E_Pa: float = 3.15e10, rho_kg_m3: float = 2500.0, area_m2: float = 2.0, inertia_m4: float = 1.0,
        damping_ratio: float = 0.02, damping_fit_modes: Tuple[int, int] = (1, 3), strain_z_m: float = 0.5,):

        self.spans = np.asarray(spans_m, dtype=float)
        self.n_spans = len(self.spans)
        self.elements_per_span = int(elements_per_span)
        self.E = float(E_Pa)
        self.rho = float(rho_kg_m3)
        self.area = float(area_m2)
        self.inertia = float(inertia_m4)
        self.damping_ratio = float(damping_ratio)
        self.damping_fit_modes = tuple(damping_fit_modes)
        self.strain_z = float(strain_z_m)
        self.span_length = float(self.spans[0])
        self.length = float(np.sum(self.spans))
        self.n_elements = self.n_spans * self.elements_per_span
        self.element_length = self.length / self.n_elements
        self.n_nodes = self.n_elements + 1
        self.node_x = np.linspace(0.0, self.length, self.n_nodes)
        self.ndof_full = 2 * self.n_nodes

        support_nodes = [j * self.elements_per_span for j in range(self.n_spans + 1)]
        self.support_nodes = np.array(support_nodes, dtype=int)
        self.fixed_dofs = np.array([2 * n for n in support_nodes], dtype=int)
        fixed_set = set(self.fixed_dofs.tolist())
        self.free_dofs = np.array([i for i in range(self.ndof_full) if i not in fixed_set], dtype=int)
        self.ndof = len(self.free_dofs)

        self.K_full, self.M_full = self._assemble_global_matrices()
        self.K = self.K_full[np.ix_(self.free_dofs, self.free_dofs)]
        self.M = self.M_full[np.ix_(self.free_dofs, self.free_dofs)]

        self.alpha_rayleigh, self.beta_rayleigh = self._fit_rayleigh_damping()
        self.C = self.alpha_rayleigh * self.M + self.beta_rayleigh * self.K


    def copy_with_perturbation(self, perturbation: PlantPerturbation) -> "ThreeSpanBridgeFE":
        return ThreeSpanBridgeFE(
            spans_m=tuple(self.spans.tolist()),
            elements_per_span=self.elements_per_span,
            E_Pa=self.E * float(perturbation.E_scale),
            rho_kg_m3=self.rho,
            area_m2=self.area,
            inertia_m4=self.inertia,
            damping_ratio=self.damping_ratio * float(perturbation.damping_ratio_scale),
            damping_fit_modes=self.damping_fit_modes,
            strain_z_m=self.strain_z)



    def _element_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        L = self.element_length
        EI = self.E * self.inertia
        rhoA = self.rho * self.area

        Ke = (EI / L**3) * np.array(
            [
                [12.0, 6.0 * L, -12.0, 6.0 * L],
                [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
                [-12.0, -6.0 * L, 12.0, -6.0 * L],
                [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
            ]
        )

        Me = (rhoA * L / 420.0) * np.array(
            [
                [156.0, 22.0 * L, 54.0, -13.0 * L],
                [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
                [54.0, 13.0 * L, 156.0, -22.0 * L],
                [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
            ]
        )
        return Ke, Me


    def _assemble_global_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        Ke, Me = self._element_matrices()
        K = np.zeros((self.ndof_full, self.ndof_full))
        M = np.zeros_like(K)

        for e in range(self.n_elements):
            dofs = np.array(
                [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1], dtype=int
            )
            K[np.ix_(dofs, dofs)] += Ke
            M[np.ix_(dofs, dofs)] += Me
        return K, M

    def modal_analysis(self, n_modes: Optional[int] = None):
        lam, Phi = eigh(self.K, self.M)
        positive = lam > 1e-10
        lam = lam[positive]
        Phi = Phi[:, positive]
        omega = np.sqrt(lam)
        freq_hz = omega / (2.0 * np.pi)

        if n_modes is not None:
            freq_hz = freq_hz[:n_modes]
            Phi = Phi[:, :n_modes]
        return freq_hz, Phi

    def full_mode_shape(self, mode_free: np.ndarray) -> np.ndarray:
        phi = np.zeros(self.ndof_full)
        phi[self.free_dofs] = mode_free
        return phi

    def vertical_mode_shapes(self, n_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        freq, Phi = self.modal_analysis(n_modes)
        W = np.zeros((self.n_nodes, n_modes))
        for j in range(n_modes):
            full = self.full_mode_shape(Phi[:, j])
            w = full[0::2]
            scale = np.max(np.abs(w))
            if scale > 0:
                w = w / scale
            W[:, j] = w
        return freq, W

    def _fit_rayleigh_damping(self) -> Tuple[float, float]:
        freq, _ = self.modal_analysis(max(self.damping_fit_modes))
        i, j = self.damping_fit_modes
        w1 = 2 * np.pi * freq[i - 1]
        w2 = 2 * np.pi * freq[j - 1]
        zeta = self.damping_ratio
        A = np.array([[1.0 / (2.0 * w1), w1 / 2.0], [1.0 / (2.0 * w2), w2 / 2.0]])
        alpha, beta = np.linalg.solve(A, np.array([zeta, zeta]))
        return float(alpha), float(beta)

    def reduced_modal_matrices(self, n_modes: int = 6):
        freq, Phi = self.modal_analysis(n_modes)
        Mr = Phi.T @ self.M @ Phi
        Cr = Phi.T @ self.C @ Phi
        Kr = Phi.T @ self.K @ Phi
        return freq, Phi, Mr, Cr, Kr


    @staticmethod
    def hermite_shape(xi: float, L: float) -> np.ndarray:
        return np.array(
            [
                1.0 - 3.0 * xi**2 + 2.0 * xi**3,
                L * (xi - 2.0 * xi**2 + xi**3),
                3.0 * xi**2 - 2.0 * xi**3,
                L * (-xi**2 + xi**3),
            ]
        )

    @staticmethod
    def hermite_second_derivative(xi: float, L: float) -> np.ndarray:
        return np.array(
            [
                (-6.0 + 12.0 * xi) / L**2,
                (-4.0 + 6.0 * xi) / L,
                (6.0 - 12.0 * xi) / L**2,
                (-2.0 + 6.0 * xi) / L,
            ]
        )

    def _element_and_xi(self, x: float) -> Tuple[int, float]:
        if x < 0.0 or x > self.length:
            raise ValueError("x is outside the bridge.")
        if np.isclose(x, self.length):
            return self.n_elements - 1, 1.0
        e = int(np.floor(x / self.element_length))
        e = min(max(e, 0), self.n_elements - 1)
        x_left = self.node_x[e]
        xi = (x - x_left) / self.element_length
        return e, float(xi)

    def moving_point_load_vector(self, x: float, load_N: float) -> np.ndarray:
        if x < 0.0 or x > self.length:
            return np.zeros(self.ndof)
        e, xi = self._element_and_xi(x)
        N = self.hermite_shape(xi, self.element_length)
        f_local = -float(load_N) * N
        f_full = np.zeros(self.ndof_full)
        dofs = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1], dtype=int)
        f_full[dofs] += f_local
        return f_full[self.free_dofs]

    def truck_force_vector(self, t: float, scenario: TruckScenario) -> np.ndarray:
        x1 = scenario.speed_m_s * t
        x2 = x1 - scenario.axle_spacing_m
        F1, F2 = scenario.axle_loads_N
        f = np.zeros(self.ndof)
        if 0.0 <= x1 <= self.length:
            f += self.moving_point_load_vector(x1, F1)
        if 0.0 <= x2 <= self.length:
            f += self.moving_point_load_vector(x2, F2)
        return f


    def simulate_truck(self, scenario: TruckScenario = TruckScenario(), dt: float = 0.002, ring_down_s: float = 0.5,
        beta_newmark: float = 0.25, gamma_newmark: float = 0.5) -> Dict[str, np.ndarray]:
        t_end = (self.length + scenario.axle_spacing_m) / scenario.speed_m_s + ring_down_s
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        nt = len(t)

        u = np.zeros((nt, self.ndof))
        v = np.zeros_like(u)
        a = np.zeros_like(u)
        f = np.zeros_like(u)

        for k, tk in enumerate(t):
            f[k] = self.truck_force_vector(tk, scenario)

        a[0] = np.linalg.solve(self.M, f[0] - self.C @ v[0] - self.K @ u[0])

        beta = beta_newmark
        gamma = gamma_newmark
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        K_eff = self.K + a0 * self.M + a1 * self.C
        K_eff_factor = cho_factor(K_eff)

        for k in range(nt - 1):
            rhs = ( f[k + 1] + self.M @ (a0 * u[k] + a2 * v[k] + a3 * a[k])
                    + self.C @ (a1 * u[k] + a4 * v[k] + a5 * a[k]))
            u[k + 1] = cho_solve(K_eff_factor, rhs)
            a[k + 1] = a0 * (u[k + 1] - u[k]) - a2 * v[k] - a3 * a[k]
            v[k + 1] = v[k] + dt * ((1.0 - gamma) * a[k] + gamma * a[k + 1])

        return {"t": t, "u_free": u, "v_free": v,"a_free": a,"force_free": f,"scenario": scenario,
            "bridge": self,"truth_parameters": {"E_Pa": self.E,"damping_ratio": self.damping_ratio,}}



    def _expand_free_history(self, X_free: np.ndarray) -> np.ndarray:
        X_full = np.zeros((X_free.shape[0], self.ndof_full))
        X_full[:, self.free_dofs] = X_free
        return X_full

    def sensor_true_signal(self, simulation: Dict[str, np.ndarray], x_m: float, sensor_type: int) -> np.ndarray:
        """
        sensor_type:
            1 = acceleration [m/s²]
            2 = strain [-]
            3 = displacement [m]
        """
        if sensor_type not in (1, 2, 3):
            raise ValueError("sensor_type must be 1, 2, or 3.")
        e, xi = self._element_and_xi(float(x_m))
        dofs = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1], dtype=int )

        if sensor_type == 1:
            X_full = self._expand_free_history(simulation["a_free"])
            qe = X_full[:, dofs]
            N = self.hermite_shape(xi, self.element_length)
            return qe @ N

        if sensor_type == 3:
            X_full = self._expand_free_history(simulation["u_free"])
            qe = X_full[:, dofs]
            N = self.hermite_shape(xi, self.element_length)
            return qe @ N

        X_full = self._expand_free_history(simulation["u_free"])
        qe = X_full[:, dofs]
        B2 = self.hermite_second_derivative(xi, self.element_length)
        curvature = qe @ B2
        return -self.strain_z * curvature



class SensorNetwork:
    def __init__(self,bridge: ThreeSpanBridgeFE, specs: Optional[Dict[int, SensorSpec]] = None):

        self.bridge = bridge
        self.specs = SENSOR_SPECS if specs is None else specs

    def _coerce_gene(self, gene: SensorGene) -> SensorPlacement:
        if isinstance(gene, SensorPlacement):
            return gene
        if isinstance(gene, dict):
            return SensorPlacement(float(gene["position_m"]), int(gene["sensor_type"]))
        if isinstance(gene, (tuple, list, np.ndarray)) and len(gene) == 2:
            return SensorPlacement(float(gene[0]), int(gene[1]))
        raise TypeError(
            "Each sensor must be SensorPlacement, (position_m, sensor_type), "
            "or a dict with position_m/sensor_type.")

    def validate(self, design: Sequence[SensorGene], allow_empty: bool = True) -> List[SensorPlacement]:
        placements = [self._coerce_gene(g) for g in design]
        if not placements and not allow_empty:
            raise ValueError("The sensor design cannot be empty.")
        for p in placements:
            if not (0.0 <= p.position_m <= self.bridge.length):
                raise ValueError(
                    f"Sensor position {p.position_m:g} m is outside [0,{self.bridge.length:g}]."
                )
            if p.sensor_type not in self.specs:
                raise ValueError(f"Unknown sensor type {p.sensor_type}.")
        return placements

    def as_array(self, design: Sequence[SensorGene]) -> np.ndarray:
        p = self.validate(design)
        if not p:
            return np.empty((0, 2))
        return np.array([[x.position_m, x.sensor_type] for x in p], dtype=float)

    def cost(self, design: Sequence[SensorGene]) -> float:
        placements = self.validate(design)
        return float(sum(self.specs[p.sensor_type].cost for p in placements))

    def n_sensors(self, design: Sequence[SensorGene]) -> int:
        return len(self.validate(design))

    def measure( self, simulation: Dict[str, np.ndarray], design: Sequence[SensorGene],
        seed: Optional[int] = None, add_noise: bool = True, failed_design_indices: Optional[Iterable[int]] = None) -> Tuple[np.ndarray, List[dict]]:

        placements = self.validate(design)
        failed = set() if failed_design_indices is None else set(map(int, failed_design_indices))
        rng = np.random.default_rng(seed)

        truth_bridge = simulation.get("bridge", self.bridge)
        channels: List[np.ndarray] = []
        metadata: List[dict] = []

        for i, p in enumerate(placements):
            if i in failed:
                continue

            spec = self.specs[p.sensor_type]
            y_true = truth_bridge.sensor_true_signal(simulation, p.position_m, p.sensor_type)
            rms = np.sqrt(np.mean(y_true**2))
            sigma = spec.relative_noise * max(rms, 1e-14)
            y = (y_true + rng.normal(0.0, sigma, size=y_true.shape) if add_noise else y_true.copy())

            channels.append(y)
            metadata.append(
                {
                    "design_index": i,
                    "position_m": float(p.position_m),
                    "sensor_type": int(p.sensor_type),
                    "sensor_name": spec.name,
                    "cost": float(spec.cost),
                    "noise_sigma": float(sigma),
                }
            )

        if not channels:
            return np.empty((len(simulation["t"]), 0)), metadata
        return np.column_stack(channels), metadata



def example():
    nominal_bridge = ThreeSpanBridgeFE()
    perturbation = sample_plant_perturbation(seed=7)
    truth_bridge = nominal_bridge.copy_with_perturbation(perturbation)

    scenario = TruckScenario((80e3, 120e3), axle_spacing_m=4.0, speed_m_s=20.0)
    simulation = truth_bridge.simulate_truck(scenario, dt=0.002)

    network = SensorNetwork(nominal_bridge)
    design = [
        SensorPlacement(6.4, 1),
        SensorPlacement(14.8, 2),
        SensorPlacement(28.5, 3),
        SensorPlacement(37.2, 1),
        SensorPlacement(51.1, 2),
    ]

    Y, metadata = network.measure(simulation, design, seed=123)

    print("Nominal E [Pa]:", nominal_bridge.E)
    print("Truth E [Pa]:  ", truth_bridge.E)
    print("Nominal damping ratio:", nominal_bridge.damping_ratio)
    print("Truth damping ratio:  ", truth_bridge.damping_ratio)
    print("Design array:\n", network.as_array(design))
    print("Network cost:", network.cost(design))
    print("Measurements:", Y.shape)
    print("Metadata:")
    for m in metadata:
        print(m)


if __name__ == "__main__":
    example()
