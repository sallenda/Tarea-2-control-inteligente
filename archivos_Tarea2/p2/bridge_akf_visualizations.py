from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from bridge_ga_model import (
    ThreeSpanBridgeFE,
    TruckScenario,
    SensorNetwork,
    SensorPlacement)
from bridge_akf import (
    AKFResult,
    PositionUncertainty,
    PreparedCase,
    central_both_axles_mask,
    evaluate_design_on_case,
    evaluate_design_cases,
    prepare_case,
    prepare_cases,
    sample_truck_scenarios,
    true_axle_positions)



def _true_force_history(
    t: np.ndarray,
    scenario: TruckScenario,
    bridge: ThreeSpanBridgeFE,
    axle_index: int,
) -> np.ndarray:
    pos = true_axle_positions(t, scenario)[:, axle_index]
    y = np.full_like(t, np.nan, dtype=float)
    mask = (pos >= 0.0) & (pos <= bridge.length)
    y[mask] = scenario.axle_loads_N[axle_index]
    return y


def _evaluation_interval(
    t: np.ndarray,
    scenario: TruckScenario,
    bridge: ThreeSpanBridgeFE,
    margin_m: float = 2.0,
) -> Tuple[float, float] | None:
    mask = central_both_axles_mask(t, scenario, bridge, margin_m)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    return float(t[idx[0]]), float(t[idx[-1]])


def _decorate_eval_window(ax, result: AKFResult, scenario: TruckScenario, bridge: ThreeSpanBridgeFE):
    interval = _evaluation_interval(result.t, scenario, bridge)
    if interval is not None:
        ax.axvspan(interval[0], interval[1], alpha=0.08, label="Evaluation window")



def plot_force_estimates(
    result: AKFResult,
    scenario: TruckScenario,
    bridge: ThreeSpanBridgeFE,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(result.t, result.force_hat_N[:, j] / 1e3, label="AKF estimate")
        ax.plot(
            result.t,
            _true_force_history(result.t, scenario, bridge, j) / 1e3,
            linestyle="--",
            linewidth=2,
            label="True load",
        )
        _decorate_eval_window(ax, result, scenario, bridge)
        ax.set_ylabel("Load [kN]")
        ax.set_title("Front axle" if j == 0 else "Rear axle")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("AKF load estimation with model/position uncertainty", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig, axes



def plot_relative_error(
    result: AKFResult,
    scenario: TruckScenario,
    bridge: ThreeSpanBridgeFE,
    clip_pct: float = 100.0,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    pos = true_axle_positions(result.t, scenario)
    for j, ax in enumerate(axes):
        F = float(scenario.axle_loads_N[j])
        mask = (pos[:, j] >= 0) & (pos[:, j] <= bridge.length)
        err = np.full(len(result.t), np.nan)
        err[mask] = np.abs(result.force_hat_N[mask, j] - F) / F * 100.0
        err = np.minimum(err, clip_pct)
        ax.plot(result.t, err)
        _decorate_eval_window(ax, result, scenario, bridge)
        ax.set_ylabel("Relative error [%]")
        ax.set_title("Front axle" if j == 0 else "Rear axle")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Instantaneous load-estimation error", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig, axes


def build_example(n_modes: int = 6, dt: float = 0.002):
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
        dt=dt,
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
        n_modes=n_modes,
        dt=dt,
        evaluate_failures=True,
    )

    scenarios = sample_truck_scenarios(6, seed=91)
    cases = prepare_cases(
        nominal_bridge,
        scenarios,
        dt=dt,
        base_seed=500,
        position_uncertainty=pos_unc,
    )
    multi_out = evaluate_design_cases(
        nominal_bridge,
        network,
        design,
        cases,
        n_modes=n_modes,
        dt=dt,
        evaluate_failures=False,
    )
    return nominal_bridge, network, case, design, out, multi_out


def make_all_figures(
    save: bool = False,
    output_dir: str = "bridge_akf_figures",
    n_modes: int = 6,
):
    bridge, network, case, design, out, multi_out = build_example(n_modes=n_modes)
    result = out["result"]

    figures = []
    figures.append(("01_force_estimates.png", plot_force_estimates(result, case.scenario, bridge)[0]))
    figures.append(("02_relative_error.png", plot_relative_error(result, case.scenario, bridge)[0]))

    if save:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for filename, fig in figures:
            fig.savefig(outdir / filename, dpi=180, bbox_inches="tight")
        print(f"Saved {len(figures)} figures to: {outdir.resolve()}")

    print("\nSingle-case summary")
    print("-------------------")
    print(f"Sensors: {out['n_sensors']} | Cost: {out['cost']:.1f}")
    print(f"Nominal error: {out['nominal_error_pct']:.3f}%")
    print("\nMulti-case summary")
    print("------------------")
    print(f"Mean error: {multi_out['mean_error_pct']:.3f}%")
    print(f"90th percentile: {multi_out['percentile_error_pct']:.3f}%")

    return figures, out, multi_out


if __name__ == "__main__":
    SAVE_FIGURES = False
    figures, out, multi_out = make_all_figures(save=SAVE_FIGURES, n_modes=6)
    plt.show()
