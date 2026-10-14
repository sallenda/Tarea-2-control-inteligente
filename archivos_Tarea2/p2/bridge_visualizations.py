from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from bridge_ga_model import (
    ThreeSpanBridgeFE,
    TruckScenario,
    SensorNetwork,
    SensorPlacement,
    SensorGene,
    sample_plant_perturbation)
from bridge_akf import (
    PositionUncertainty,
    generate_noisy_axle_positions)


SENSOR_MARKERS = {1: "^", 2: "s", 3: "o"}
SENSOR_LABELS = {1: "Accelerometer", 2: "Strain gauge", 3: "Displacement"}
SENSOR_UNITS = {1: "m/s²", 2: "strain", 3: "m"}



def plot_bridge_and_sensor_network(
    bridge: ThreeSpanBridgeFE,
    network: SensorNetwork,
    design: Sequence[SensorGene],
    title: str = "Continuous sensor-network design"):
    placements = network.validate(design)
    fig, ax = plt.subplots(figsize=(12, 3.8))

    ax.plot([0, bridge.length], [0, 0], linewidth=3, label="Bridge deck")
    ax.scatter(
        bridge.node_x, np.zeros_like(bridge.node_x), s=22, zorder=3, label="FE nodes"
    )
    supports_x = bridge.node_x[bridge.support_nodes]
    ax.scatter(
        supports_x,
        np.full_like(supports_x, -0.06),
        marker="^",
        s=140,
        zorder=4,
        label="Supports",
    )

    for sensor_type in (1, 2, 3):
        xs = [p.position_m for p in placements if p.sensor_type == sensor_type]
        if xs:
            ax.scatter(
                xs,
                np.full(len(xs), 0.14 + 0.05 * sensor_type),
                marker=SENSOR_MARKERS[sensor_type],
                s=100,
                zorder=5,
                label=SENSOR_LABELS[sensor_type],
            )

    x0 = 0.0
    for i, span in enumerate(bridge.spans, start=1):
        ax.text(x0 + span / 2, -0.18, f"Span {i}", ha="center", va="top")
        x0 += span

    ax.set_ylim(-0.3, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Position along bridge [m]")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4)
    fig.tight_layout()
    return fig, ax



def plot_selected_sensor_histories(
    network: SensorNetwork,
    simulation: dict,
    design: Sequence[SensorGene],
    seed: int = 123,
    add_noise: bool = True,
):
    Y, metadata = network.measure(simulation, design, seed=seed, add_noise=add_noise)
    if Y.shape[1] == 0:
        raise ValueError("Design contains no sensors.")

    active_types = sorted(set(m["sensor_type"] for m in metadata))
    fig, axes = plt.subplots(
        len(active_types), 1, figsize=(11, 3.2 * len(active_types)), sharex=True
    )
    axes = np.atleast_1d(axes)
    t = simulation["t"]

    for ax, sensor_type in zip(axes, active_types):
        for j, meta in enumerate(metadata):
            if meta["sensor_type"] != sensor_type:
                continue
            y = Y[:, j]
            if sensor_type == 2:
                y = y * 1e6
                unit = "µε"
            elif sensor_type == 3:
                y = y * 1e3
                unit = "mm"
            else:
                unit = SENSOR_UNITS[sensor_type]
            ax.plot(t, y, label=f"x = {meta['position_m']:.2f} m")
        ax.set_ylabel(f"{SENSOR_LABELS[sensor_type]} [{unit}]")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        "Continuous-position sensor measurements" + (" (with noise)" if add_noise else ""),
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    return fig, axes


def build_example():
    nominal_bridge = ThreeSpanBridgeFE()
    perturbation = sample_plant_perturbation(seed=42)
    truth_bridge = nominal_bridge.copy_with_perturbation(perturbation)
    network = SensorNetwork(nominal_bridge)

    scenario = TruckScenario((80e3, 120e3), 4.0, 20.0)
    simulation = truth_bridge.simulate_truck(scenario, dt=0.002, ring_down_s=0.5)
    pos_unc = PositionUncertainty()
    pos_est = generate_noisy_axle_positions(
        simulation["t"], scenario, uncertainty=pos_unc, seed=77
    )

    design = [
        SensorPlacement(5.8, 1),
        SensorPlacement(13.9, 2),
        SensorPlacement(27.4, 3),
        SensorPlacement(39.1, 1),
        SensorPlacement(50.7, 2),
    ]
    return nominal_bridge, truth_bridge, network, scenario, simulation, pos_est, design


def make_all_figures(save: bool = False, output_dir: str = "bridge_figures"):
    nominal, truth, network, scenario, simulation, pos_est, design = build_example()
    figures = []
    figures.append(("01_continuous_sensor_design.png", plot_bridge_and_sensor_network(nominal, network, design)[0]))
    figures.append(("02_sensor_histories.png", plot_selected_sensor_histories(network, simulation, design, seed=123)[0]))

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for filename, fig in figures:
            fig.savefig(out / filename, dpi=180, bbox_inches="tight")
        print(f"Saved {len(figures)} figures to: {out.resolve()}")

    return figures


if __name__ == "__main__":
    SAVE_FIGURES = False
    figures  = make_all_figures(save=SAVE_FIGURES)
    plt.show()
