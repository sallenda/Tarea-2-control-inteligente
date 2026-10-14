from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.linalg import expm, solve
from bridge_ga_model import (
    ThreeSpanBridgeFE,
    TruckScenario,
    SensorNetwork,
    SensorPlacement,
    SensorGene,
    PlantPerturbation,
    sample_plant_perturbation,
)



@dataclass
class AKFResult:
    t: np.ndarray
    x_hat: np.ndarray
    P_diag: np.ndarray
    innovations: np.ndarray
    force_hat_N: np.ndarray
    q_hat: np.ndarray
    qdot_hat: np.ndarray
    axle_positions_used_m: np.ndarray


@dataclass(frozen=True)
class PositionUncertainty:
    common_bias_std_m: float = 0.10
    spacing_error_std_m: float = 0.05
    jitter_std_m: float = 0.20
    correlation_time_s: float = 0.05


@dataclass
class PreparedCase:
    scenario: TruckScenario
    truth_bridge: ThreeSpanBridgeFE
    simulation: Dict[str, np.ndarray]
    plant_perturbation: PlantPerturbation
    axle_positions_true_m: np.ndarray
    axle_positions_est_m: np.ndarray
    seed: int



def true_axle_positions(t: np.ndarray, scenario: TruckScenario) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    front = scenario.speed_m_s * t
    rear = front - scenario.axle_spacing_m
    return np.column_stack([front, rear])


def _smooth_unit_noise( n: int,dt: float,correlation_time_s: float,rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    if correlation_time_s <= 0:
        return rng.normal(size=n)

    rho = float(np.exp(-dt / correlation_time_s))
    scale = np.sqrt(max(1.0 - rho**2, 0.0))
    x = np.zeros(n)
    x[0] = rng.normal()
    for k in range(1, n):
        x[k] = rho * x[k - 1] + scale * rng.normal()
    return x


def generate_noisy_axle_positions(t: np.ndarray,scenario: TruckScenario,
                                  uncertainty: PositionUncertainty = PositionUncertainty(), seed: Optional[int] = None,) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    true_pos = true_axle_positions(t, scenario)
    rng = np.random.default_rng(seed)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0

    common_bias = rng.normal(0.0, uncertainty.common_bias_std_m)
    spacing_error = rng.normal(0.0, uncertainty.spacing_error_std_m)

    jitter_front = uncertainty.jitter_std_m * _smooth_unit_noise(
        len(t), dt, uncertainty.correlation_time_s, rng)
    jitter_rear = uncertainty.jitter_std_m * _smooth_unit_noise(
        len(t), dt, uncertainty.correlation_time_s, rng)

    estimated = true_pos.copy()
    estimated[:, 0] += common_bias + jitter_front
    # If the estimated spacing is d + spacing_error, the rear axle estimate is
    # shifted by -spacing_error relative to the front reference.
    estimated[:, 1] += common_bias - spacing_error + jitter_rear
    return estimated



class AugmentedBridgeAKF:

    def __init__(self, bridge: ThreeSpanBridgeFE, n_modes: int = 6, dt: float = 0.002, q_modal: float = 1e-12,
        q_velocity: float = 1e-8, q_force: float = 1e3, initial_modal_std: float = 1e-4,
        initial_velocity_std: float = 1e-2, initial_force_std_N: float = 200e3, min_measurement_variance: float = 1e-24,):
        self.bridge = bridge
        self.n_modes = int(n_modes)
        self.dt = float(dt)

        self.freq_hz, self.Phi, self.Mr, self.Cr, self.Kr = (bridge.reduced_modal_matrices(self.n_modes))
        self.Mr_inv = np.linalg.inv(self.Mr)

        r = self.n_modes
        self.n_struct = 2 * r
        self.n_force = 2
        self.n_state = self.n_struct + self.n_force

        Z = np.zeros((r, r))
        I = np.eye(r)
        self.Ac_struct = np.block([[Z, I], [-self.Mr_inv @ self.Kr, -self.Mr_inv @ self.Cr]])
        aug = np.zeros((2 * self.n_struct, 2 * self.n_struct))
        aug[: self.n_struct, : self.n_struct] = self.Ac_struct
        aug[: self.n_struct, self.n_struct :] = np.eye(self.n_struct)
        exp_aug = expm(aug * self.dt)
        self.Ad_struct = exp_aug[: self.n_struct, : self.n_struct]
        self.Gint_struct = exp_aug[: self.n_struct, self.n_struct :]

        self.Q = np.diag(np.r_[
                np.full(r, float(q_modal)),
                np.full(r, float(q_velocity)),
                np.full(2, float(q_force))])
        self.P0 = np.diag(np.r_[
                np.full(r, float(initial_modal_std) ** 2),
                np.full(r, float(initial_velocity_std) ** 2),
                np.full(2, float(initial_force_std_N) ** 2)])
        self.min_measurement_variance = float(min_measurement_variance)
        self.Ix = np.eye(self.n_state)



    def modal_force_location_matrix_from_positions(self, axle_positions_m: Sequence[float]) -> np.ndarray:
        positions = np.asarray(axle_positions_m, dtype=float)
        if positions.shape != (2,):
            raise ValueError("axle_positions_m must have shape (2,).")

        Br = np.zeros((self.n_modes, 2))
        for j, x in enumerate(positions):
            if 0.0 <= x <= self.bridge.length:
                s_free = self.bridge.moving_point_load_vector(float(x), 1.0)
                Br[:, j] = self.Phi.T @ s_free
        return Br

    def transition_matrix_from_positions(self, axle_positions_mid_m: Sequence[float]) -> np.ndarray:
        Br = self.modal_force_location_matrix_from_positions(axle_positions_mid_m)
        r = self.n_modes
        Bc_struct = np.vstack([np.zeros((r, 2)), self.Mr_inv @ Br])
        Bd_struct = self.Gint_struct @ Bc_struct

        Aaug = np.zeros((self.n_state, self.n_state))
        Aaug[: self.n_struct, : self.n_struct] = self.Ad_struct
        Aaug[: self.n_struct, self.n_struct :] = Bd_struct
        Aaug[self.n_struct :, self.n_struct :] = np.eye(2)
        return Aaug


    def _physical_measurement_row(self, x_m: float, sensor_type: int) -> np.ndarray:
        bridge = self.bridge
        e, xi = bridge._element_and_xi(float(x_m))
        dofs = np.array(
            [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1], dtype=int
        )
        row_full = np.zeros(bridge.ndof_full)

        if sensor_type in (1, 3):
            local = bridge.hermite_shape(xi, bridge.element_length)
        elif sensor_type == 2:
            local = -bridge.strain_z * bridge.hermite_second_derivative(
                xi, bridge.element_length
            )
        else:
            raise ValueError("sensor_type must be 1, 2, or 3.")

        row_full[dofs] = local
        return row_full[bridge.free_dofs]

    def measurement_matrix_from_positions(
        self,
        axle_positions_m: Sequence[float],
        metadata: Sequence[dict],
    ) -> np.ndarray:
        if len(metadata) == 0:
            return np.empty((0, self.n_state))

        Br = self.modal_force_location_matrix_from_positions(axle_positions_m)
        Aq = -self.Mr_inv @ self.Kr
        Av = -self.Mr_inv @ self.Cr
        Af = self.Mr_inv @ Br
        rows = []

        for meta in metadata:
            x_m = float(meta["position_m"])
            sensor_type = int(meta["sensor_type"])
            h_free = self._physical_measurement_row(x_m, sensor_type)
            h_modal = h_free @ self.Phi

            if sensor_type in (2, 3):
                row = np.r_[h_modal, np.zeros(self.n_modes), np.zeros(2)]
            else:
                row = np.r_[h_modal @ Aq, h_modal @ Av, h_modal @ Af]
            rows.append(row)

        return np.vstack(rows)

    def measurement_covariance(self, metadata: Sequence[dict]) -> np.ndarray:
        if len(metadata) == 0:
            return np.empty((0, 0))
        variance = [
            max(float(m.get("noise_sigma", 0.0)) ** 2, self.min_measurement_variance)
            for m in metadata
        ]
        return np.diag(variance)


    def run(self,t: np.ndarray,Y: np.ndarray,metadata: Sequence[dict],scenario: TruckScenario,
        axle_positions_est_m: Optional[np.ndarray] = None,
            x0: Optional[np.ndarray] = None,P0: Optional[np.ndarray] = None) -> AKFResult:

        t = np.asarray(t, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if t.ndim != 1 or Y.ndim != 2:
            raise ValueError("t must be 1-D and Y must be 2-D.")
        if Y.shape[0] != len(t) or Y.shape[1] != len(metadata):
            raise ValueError("Y dimensions are inconsistent with t/metadata.")
        if len(metadata) == 0:
            raise ValueError("The AKF requires at least one active sensor.")

        if len(t) >= 2:
            dt_data = float(np.median(np.diff(t)))
            if not np.isclose(dt_data, self.dt, rtol=1e-4, atol=1e-10):
                raise ValueError(
                    f"Filter dt={self.dt:g} does not match data dt={dt_data:g}.")

        if axle_positions_est_m is None:
            positions = true_axle_positions(t, scenario)
        else:
            positions = np.asarray(axle_positions_est_m, dtype=float)
            if positions.shape != (len(t), 2):
                raise ValueError(
                    f"axle_positions_est_m must have shape ({len(t)},2).")

        nt, ns = Y.shape
        x = np.zeros(self.n_state) if x0 is None else np.asarray(x0, dtype=float).copy()
        P = self.P0.copy() if P0 is None else np.asarray(P0, dtype=float).copy()
        if x.shape != (self.n_state,):
            raise ValueError(f"x0 must have shape ({self.n_state},).")
        if P.shape != (self.n_state, self.n_state):
            raise ValueError(f"P0 must have shape ({self.n_state},{self.n_state}).")

        R = self.measurement_covariance(metadata)
        x_hist = np.zeros((nt, self.n_state))
        pdiag_hist = np.zeros((nt, self.n_state))
        innovation_hist = np.zeros((nt, ns))

        for k in range(nt):
            H = self.measurement_matrix_from_positions(positions[k], metadata)
            innovation = Y[k] - H @ x
            S = H @ P @ H.T + R
            S = 0.5 * (S + S.T)
            reg = max(1e-15, 1e-12 * np.max(np.abs(np.diag(S))))
            S = S + reg * np.eye(ns)

            PHt = P @ H.T
            K_gain = solve(S, PHt.T, assume_a="sym").T
            x = x + K_gain @ innovation

            IKH = self.Ix - K_gain @ H
            P = IKH @ P @ IKH.T + K_gain @ R @ K_gain.T
            P = 0.5 * (P + P.T)

            x_hist[k] = x
            pdiag_hist[k] = np.diag(P)
            innovation_hist[k] = innovation

            if k < nt - 1:
                positions_mid = 0.5 * (positions[k] + positions[k + 1])
                Aaug = self.transition_matrix_from_positions(positions_mid)
                x = Aaug @ x
                P = Aaug @ P @ Aaug.T + self.Q
                P = 0.5 * (P + P.T)

        r = self.n_modes
        return AKFResult(
            t=t.copy(),
            x_hat=x_hist,
            P_diag=pdiag_hist,
            innovations=innovation_hist,
            force_hat_N=x_hist[:, 2 * r : 2 * r + 2],
            q_hat=x_hist[:, :r],
            qdot_hat=x_hist[:, r : 2 * r],
            axle_positions_used_m=positions.copy(),
        )



def central_both_axles_mask(t: np.ndarray,scenario: TruckScenario,bridge: ThreeSpanBridgeFE,margin_m: float = 2.0) -> np.ndarray:
    pos = true_axle_positions(t, scenario)
    return ((pos[:, 0] >= margin_m)& (pos[:, 0] <= bridge.length - margin_m)& (pos[:, 1] >= margin_m)
        & (pos[:, 1] <= bridge.length - margin_m))


def summarize_force_estimation(result: AKFResult,scenario: TruckScenario,bridge: ThreeSpanBridgeFE,
    margin_m: float = 2.0) -> Dict[str, object]:
    mask = central_both_axles_mask(result.t, scenario, bridge, margin_m)
    if not np.any(mask):
        raise ValueError("No samples remain in the selected evaluation window.")

    F_hat_mean = np.mean(result.force_hat_N[mask], axis=0)
    F_true = np.asarray(scenario.axle_loads_N, dtype=float)
    error_pct = np.abs(F_hat_mean - F_true) / F_true * 100.0
    instantaneous = (np.abs(result.force_hat_N[mask] - F_true[None, :]) / F_true[None, :] * 100.0)

    return {
        "force_mean_N": F_hat_mean,
        "force_true_N": F_true,
        "error_pct_per_axle": error_pct,
        "mean_error_pct": float(np.mean(error_pct)),
        "instantaneous_mean_error_pct": float(np.mean(instantaneous)),
        "n_eval_samples": int(np.count_nonzero(mask)),
        "evaluation_mask": mask}



def prepare_case(nominal_bridge: ThreeSpanBridgeFE,scenario: TruckScenario,*,
    dt: float = 0.002,seed: int = 0, plant_perturbation: Optional[PlantPerturbation] = None,
    position_uncertainty: PositionUncertainty = PositionUncertainty(),E_std_fraction: float = 0.03,
    damping_std_fraction: float = 0.15) -> PreparedCase:

    if plant_perturbation is None:
        plant_perturbation = sample_plant_perturbation(
            seed=seed,
            E_std_fraction=E_std_fraction,
            damping_std_fraction=damping_std_fraction)

    truth_bridge = nominal_bridge.copy_with_perturbation(plant_perturbation)
    simulation = truth_bridge.simulate_truck(scenario, dt=dt)
    t = simulation["t"]
    positions_true = true_axle_positions(t, scenario)
    positions_est = generate_noisy_axle_positions(
        t, scenario, uncertainty=position_uncertainty, seed=seed + 100003)

    return PreparedCase(
        scenario=scenario,
        truth_bridge=truth_bridge,
        simulation=simulation,
        plant_perturbation=plant_perturbation,
        axle_positions_true_m=positions_true,
        axle_positions_est_m=positions_est,
        seed=int(seed))


def sample_truck_scenarios(n: int,seed: int = 0,
    front_load_kN_range: Tuple[float, float] = (60.0, 105.0),
    rear_load_kN_range: Tuple[float, float] = (90.0, 150.0),
    spacing_m_range: Tuple[float, float] = (3.5, 5.0),
    speed_m_s_range: Tuple[float, float] = (14.0, 28.0)) -> List[TruckScenario]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n)):
        F1 = rng.uniform(*front_load_kN_range) * 1e3
        F2 = rng.uniform(*rear_load_kN_range) * 1e3
        spacing = rng.uniform(*spacing_m_range)
        speed = rng.uniform(*speed_m_s_range)
        out.append(TruckScenario((float(F1), float(F2)), float(spacing), float(speed)))
    return out


def prepare_cases(nominal_bridge: ThreeSpanBridgeFE,scenarios: Sequence[TruckScenario],
    *,dt: float = 0.002,base_seed: int = 1000,position_uncertainty: PositionUncertainty = PositionUncertainty(),
    E_std_fraction: float = 0.03, damping_std_fraction: float = 0.15) -> List[PreparedCase]:
    return [
        prepare_case(
            nominal_bridge,
            s,
            dt=dt,
            seed=base_seed + i,
            position_uncertainty=position_uncertainty,
            E_std_fraction=E_std_fraction,
            damping_std_fraction=damping_std_fraction,
        )
        for i, s in enumerate(scenarios)
    ]



def _run_from_measurements(nominal_bridge: ThreeSpanBridgeFE,case: PreparedCase,
    Y: np.ndarray,metadata: Sequence[dict],*,n_modes: int,dt: float,
    margin_m: float,akf_kwargs: Optional[dict]) -> Tuple[Optional[AKFResult], Optional[Dict[str, object]]]:

    if Y.shape[1] == 0:
        return None, None
    kwargs = {} if akf_kwargs is None else dict(akf_kwargs)
    akf = AugmentedBridgeAKF(nominal_bridge, n_modes=n_modes, dt=dt, **kwargs)
    result = akf.run(case.simulation["t"],Y,
        metadata,case.scenario,axle_positions_est_m=case.axle_positions_est_m)
    summary = summarize_force_estimation(result, case.scenario, nominal_bridge, margin_m=margin_m)
    return result, summary


def evaluate_design_on_case(nominal_bridge: ThreeSpanBridgeFE,
    network: SensorNetwork,design: Sequence[SensorGene],case: PreparedCase,
    *,n_modes: int = 6,dt: float = 0.002,measurement_seed: Optional[int] = None,
    add_noise: bool = True,margin_m: float = 2.0,akf_kwargs: Optional[dict] = None,evaluate_failures: bool = True
                            ) -> Dict[str, object]:


    placements = network.validate(design)
    if measurement_seed is None:
        measurement_seed = case.seed + 200003

    Y, metadata = network.measure(
        case.simulation,
        placements,
        seed=measurement_seed,
        add_noise=add_noise,
    )

    result, summary = _run_from_measurements(
        nominal_bridge,
        case,
        Y,
        metadata,
        n_modes=n_modes,
        dt=dt,
        margin_m=margin_m,
        akf_kwargs=akf_kwargs)

    if summary is None:
        nominal_error = np.inf
    else:
        nominal_error = float(summary["mean_error_pct"])

    failure_errors: List[float] = []
    failure_details: List[dict] = []

    if evaluate_failures and len(metadata) > 0:
        for failed_channel in range(len(metadata)):
            keep = [j for j in range(len(metadata)) if j != failed_channel]
            failed_meta = metadata[failed_channel]

            if not keep:
                error = np.inf
                f_result = None
                f_summary = None
            else:
                Yf = Y[:, keep]
                Mf = [metadata[j] for j in keep]
                f_result, f_summary = _run_from_measurements(
                    nominal_bridge,
                    case,
                    Yf,
                    Mf,
                    n_modes=n_modes,
                    dt=dt,
                    margin_m=margin_m,
                    akf_kwargs=akf_kwargs,
                )
                error = float(f_summary["mean_error_pct"])

            failure_errors.append(error)
            failure_details.append(
                {
                    "failed_channel": failed_channel,
                    "failed_design_index": int(failed_meta["design_index"]),
                    "failed_sensor": failed_meta,
                    "mean_error_pct": float(error),
                    "result": f_result,
                    "summary": f_summary,
                }
            )

    if failure_errors:
        worst_failure = float(np.max(failure_errors))
        finite = np.asarray(failure_errors)[np.isfinite(failure_errors)]
        mean_failure = float(np.mean(finite)) if len(finite) else np.inf
    else:
        worst_failure = np.nan
        mean_failure = np.nan

    return {
        "nominal_error_pct": nominal_error,
        "mean_error_pct": nominal_error,  # convenient backward-compatible alias
        "cost": network.cost(placements),
        "n_sensors": len(placements),
        "design": placements,
        "summary": summary,
        "result": result,
        "metadata": metadata,
        "Y": Y,
        "case": case}


def evaluate_design_cases(nominal_bridge: ThreeSpanBridgeFE,network: SensorNetwork,
    design: Sequence[SensorGene],cases: Sequence[PreparedCase],
    *,n_modes: int = 6,dt: float = 0.002,add_noise: bool = True,margin_m: float = 2.0,
    akf_kwargs: Optional[dict] = None,evaluate_failures: bool = True, percentile: float = 90.0,
    mean_weight: float = 0.7,) -> Dict[str, object]:

    if len(cases) == 0:
        raise ValueError("cases cannot be empty.")
    if not (0.0 <= mean_weight <= 1.0):
        raise ValueError("mean_weight must lie in [0,1].")

    details = []
    nominal_errors = []
    worst_failures = []

    for i, case in enumerate(cases):
        out = evaluate_design_on_case(
            nominal_bridge,
            network,
            design,
            case,
            n_modes=n_modes,
            dt=dt,
            measurement_seed=case.seed + 300007,
            add_noise=add_noise,
            margin_m=margin_m,
            akf_kwargs=akf_kwargs,
            evaluate_failures=evaluate_failures)
        details.append(out)
        nominal_errors.append(float(out["nominal_error_pct"]))

    nominal_errors = np.asarray(nominal_errors, dtype=float)
    mean_error = float(np.mean(nominal_errors))
    p_error = float(np.percentile(nominal_errors, percentile))

    if evaluate_failures:
        worst_failures_arr = np.asarray(worst_failures, dtype=float)
        worst_failure_all = float(np.max(worst_failures_arr))
        finite = worst_failures_arr[np.isfinite(worst_failures_arr)]
        mean_worst_failure = float(np.mean(finite)) if len(finite) else np.inf
    else:
        worst_failures_arr = np.empty(0)
        worst_failure_all = np.nan
        mean_worst_failure = np.nan

    placements = network.validate(design)
    return {
        "scenario_errors_pct": nominal_errors,
        "mean_error_pct": mean_error,
        "percentile_error_pct": p_error,
        "cost": network.cost(placements),
        "n_sensors": len(placements),
        "design": placements,
        "details": details,
    }


evaluate_chromosome = evaluate_design_on_case
evaluate_scenarios = evaluate_design_cases



def example():
    nominal_bridge = ThreeSpanBridgeFE()
    network = SensorNetwork(nominal_bridge)

    scenario = TruckScenario((80e3, 120e3), 4.0, 20.0)
    pos_unc = PositionUncertainty(
        common_bias_std_m=0.10,
        spacing_error_std_m=0.05,
        jitter_std_m=0.20,
        correlation_time_s=0.05,
    )
    case = prepare_case(
        nominal_bridge,
        scenario,
        dt=0.002,
        seed=42,
        position_uncertainty=pos_unc,
    )

    design = [
        SensorPlacement(5.8, 1),
        SensorPlacement(13.9, 2),
        SensorPlacement(27.4, 3),
        SensorPlacement(39.1, 1),
        SensorPlacement(50.7, 2),
    ]

    out = evaluate_design_on_case(
        nominal_bridge,
        network,
        design,
        case,
        n_modes=6,
        dt=0.002,
        evaluate_failures=True,
    )

    print("Truth plant perturbation:", case.plant_perturbation)
    print("Sensors:", out["n_sensors"], " cost:", out["cost"])
    print("Nominal error [%]:", round(out["nominal_error_pct"], 3))
    print("Per-sensor outage errors [%]:", np.round(out["failure_errors_pct"], 3))
    print("Worst outage error [%]:", round(out["worst_failure_error_pct"], 3))



def multi_example():
    nominal_bridge = ThreeSpanBridgeFE()
    network = SensorNetwork(nominal_bridge)

    scenarios = sample_truck_scenarios(
        n=10,
        seed=123,
    )
    pos_unc = PositionUncertainty(
        common_bias_std_m=0.10,
        spacing_error_std_m=0.05,
        jitter_std_m=0.20,
        correlation_time_s=0.05,
    )

    cases = prepare_cases(
        nominal_bridge,
        scenarios,
        dt=0.002,
        base_seed=500,
        position_uncertainty=pos_unc,
    )


    design = [
        SensorPlacement(5.8, 1),
        SensorPlacement(13.9, 2),
        SensorPlacement(27.4, 3),
        SensorPlacement(39.1, 1),
        SensorPlacement(50.7, 2),
    ]

    out = evaluate_design_cases(
        nominal_bridge,
        network,
        design,
        cases,
        n_modes=6,
        dt=0.002,
        evaluate_failures=False,
    )
    print(out)




if __name__ == "__main__":
    #example()
    multi_example()
