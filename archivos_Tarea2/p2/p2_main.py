from bridge_ga_model import ThreeSpanBridgeFE, SensorNetwork, SensorPlacement
from bridge_akf import PositionUncertainty, evaluate_design_cases, sample_truck_scenarios, prepare_cases
import matplotlib.pyplot as plt
from bridge_visualizations import plot_bridge_and_sensor_network, plot_selected_sensor_histories
from bridge_akf_visualizations import plot_force_estimates, plot_relative_error




nominal_bridge = ThreeSpanBridgeFE() # Create the bridge
network = SensorNetwork(nominal_bridge)

scenarios = sample_truck_scenarios(n=5, seed=123) # Simulate n trucks events, keep a constant seed to evaluate all individuals


pos_unc = PositionUncertainty(common_bias_std_m=0.015, spacing_error_std_m=0.005,
            jitter_std_m=0.03, correlation_time_s=0.02) # Do not change

cases = prepare_cases(nominal_bridge, scenarios, position_uncertainty=pos_unc)

design = [SensorPlacement(position_m=5.8, sensor_type=1),
    SensorPlacement(position_m=13.9, sensor_type=2),
    SensorPlacement(position_m=27.4, sensor_type=3),
    SensorPlacement(position_m=55.4, sensor_type=3),
    SensorPlacement(position_m=39.1, sensor_type=1),
    SensorPlacement(position_m=50.7, sensor_type=2)] # Sensors -> (position: continous number between 1-60, type: 1,2,3)

out = evaluate_design_cases(nominal_bridge, network, design, cases,
            evaluate_failures=False)

cost = out['cost']
individual_error = out['scenario_errors_pct']
mean_error = out['mean_error_pct']
print(cost, individual_error, mean_error)
print(out.keys())

# Plot your design
plot_bridge_and_sensor_network(nominal_bridge, network, design)

see_scenario = 0
# Plot the sensor history for one particular case
simulation = nominal_bridge.simulate_truck(scenarios[see_scenario], dt=0.002, ring_down_s=0.5)
plot_selected_sensor_histories(network, simulation, design, seed=123)

# Plot the Kalman estimation vs true for one case
result_scenario = out['details'][see_scenario]['result']
plot_force_estimates(result_scenario, scenarios[see_scenario], nominal_bridge)
plot_relative_error(result_scenario, scenarios[see_scenario], nominal_bridge)

plt.show()