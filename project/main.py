from solvers.networks.six_species_odes import (
    odes as six_species_odes,
    species_names as six_species_names,
    indexes as six_species_indexes,
    calculate_gamma as six_calculate_gamma,
    calculate_temp_from_energy as six_calculate_temp,
    calculate_energy_from_temp as six_calculate_energy,
    calculate_mu as six_calculate_mu,
    reaction_group_config as six_rg_config,
)
from solvers.networks.nine_species_odes import (
    odes as nine_species_odes,
    species_names as nine_species_names,
    indexes as nine_species_indexes,
    calculate_gamma as nine_calculate_gamma,
    calculate_temp_from_energy as nine_calculate_temp,
    calculate_energy_from_temp as nine_calculate_energy,
    calculate_mu as nine_calculate_mu,
    reaction_group_config as nine_rg_config,
)
from solvers.networks.twelve_species_odes import (
    odes as twelve_species_odes,
    species_names as twelve_species_names,
    indexes as twelve_species_indexes,
    calculate_gamma as twelve_calculate_gamma,
    calculate_temp_from_energy as twelve_calculate_temp,
    calculate_energy_from_temp as twelve_calculate_energy,
    calculate_mu as twelve_calculate_mu,
    reaction_group_config as twelve_rg_config,
)

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from pygrackle import chemistry_data, RateContainer
from pygrackle.utilities.physical_constants import (
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)
from scipy.integrate import solve_ivp
from solvers.pe import pe_solver
from solvers.asymptotic import asymptotic_methods_solver
from solvers.default import (
    odes as default_odes,
    euler_method_system as default_euler,
    species_names as default_species_names,
)
import similaritymeasures
from test import test_equilibrium
from plotting import (
    plot_solution,
    plot_prediction,
    plot_rate_values,
    plot_mu,
    plot_energy_and_temperature,
    plot_timestepper_data,
    plot_partial_equilibriums,
)
from collections import namedtuple
import itertools
import time
import copy
from experiment import experiment
from solvers.utils import network, get_initial_conditions

# todo reduce number density for pe to test that partial equilibrium works

# map a config name to the solver, list of odes and the species names
NetworkConfig = namedtuple(
    "NetworkConfig",
    [
        "odes",
        "species_names",
        "idxs",
        "calculate_gamma",
        "calculate_mu",
        "calculate_energy_from_temp",
        "calculate_temp_from_energy",
        "reaction_group_config",
    ],
)

network_configs = {
    "six": NetworkConfig(
        six_species_odes,
        six_species_names,
        six_species_indexes,
        six_calculate_gamma,
        six_calculate_mu,
        six_calculate_energy,
        six_calculate_temp,
        six_rg_config,
    ),
    "nine": NetworkConfig(
        nine_species_odes,
        nine_species_names,
        nine_species_indexes,
        nine_calculate_gamma,
        nine_calculate_mu,
        nine_calculate_energy,
        nine_calculate_temp,
        nine_rg_config,
    ),
    "twelve": NetworkConfig(
        twelve_species_odes,
        twelve_species_names,
        twelve_species_indexes,
        twelve_calculate_gamma,
        twelve_calculate_mu,
        twelve_calculate_energy,
        twelve_calculate_temp,
        twelve_rg_config,
    ),
}
solver_configs = {
    "default": (default_euler, default_odes, default_species_names),
    "pe": pe_solver,
    "asymptotic": asymptotic_methods_solver,
}


def get_rates():
    current_redshift = 0.0

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 1
    my_chemistry.metal_cooling = 0
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_dir = os.path.dirname(os.path.abspath(__file__))
    grackle_data_file = bytearray(
        os.path.join(my_dir, "..", "..", "..", "input", "CloudyData_UVB=HM2012.h5"),
        "utf-8",
    )
    my_chemistry.grackle_data_file = grackle_data_file

    # Set units
    my_chemistry.comoving_coordinates = 0  # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs  # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc  # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr  # 1 Gyr in s
    my_chemistry.set_velocity_units()

    rates = RateContainer(my_chemistry)
    return rates


def convert_to_ty(t, y):
    data = np.zeros((y.shape[0], 2))
    data[:, 0] = t
    data[:, 1] = y
    return data


# this function can solve any of our defined networks
def solve_network(
    rates,
    initial_conditions,
    t_span,
    T,
    error_threshold,
    timestepper_settings,
    solver_name="pe",
    network_name="default",
    plot_results=True,
    check_error=True,
    max_iters=10000,
):
    if solver_name not in solver_configs:
        raise ValueError(f"Solver {solver_name} not found")
    if network_name not in network_configs:
        raise ValueError(f"Network {network_name} not found")

    solver = solver_configs[solver_name]
    network_config = network_configs[network_name]

    for eq in network_config.odes:
        eq.rates = rates
    print("solution solver")
    solution = solve_ivp(
        network,
        t_span,
        initial_conditions,
        method="BDF",
        args=(network_config, T),
    )
    pred_t = solution.t
    pred_y = solution.y
    print(f"number of timesteps in solution: {len(pred_t)}")

    print("custom solver")
    exp_t, exp_y, rate_values, timestepper_data = solver(
        network_config,
        initial_conditions,
        t_span,
        T,
        rates,
        timestepper_settings,
        max_iters=max_iters,
    )

    print("final solution state: ", pred_y[:, -1])
    print("final solver state: ", exp_y[:, -1])

    err = abs(exp_y[:, -1] - pred_y[:, -1]) / pred_y[:, -1]
    print("error state: ", err)
    print(all(e < error_threshold for e in err))

    if check_error:
        # check the error of the two curves isn't too large
        # print("checking error")
        # zipped_results = zip(pred_y, exp_y)
        # for i, (predicted, experimental) in enumerate(zipped_results):
        #     pred_data = convert_to_ty(pred_t, predicted)
        #     exp_data = convert_to_ty(exp_t, experimental)

        #     err = similaritymeasures.frechet_dist(exp_data, pred_data)
        #     # print(err)
        #     if err > error_threshold:
        #         print(
        #             f"ERROR: species {i} has an error value of {err} > {error_threshold}"
        #         )

        print("checking equilibrium")
        test_equilibrium(
            solver_configs[network_name], initial_conditions, rates, T, t_span
        )

    if not plot_results:
        return

    plot_solution(
        pred_t,
        pred_y,
        network_config.species_names,
        network_name,
        solver_name,
    )
    plot_prediction(
        exp_t,
        exp_y,
        network_config.species_names,
        network_name,
        solver_name,
    )
    plot_energy_and_temperature(
        network_config, exp_t, exp_y, rates, network_name, solver_name
    )
    plot_mu(network_config, exp_t, exp_y, rates, network_name, solver_name)
    plot_timestepper_data(exp_t, timestepper_data, network_name, solver_name)
    plot_partial_equilibriums(network_config, exp_t, exp_y, network_name, solver_name)


def get_timestepper_settings(preset):
    if preset == "aggressive":
        return {
            "trial_timestep_tol": 0.15,
            "conservation_tol": 0.005,
            "conservation_satisfied_tol": 0.005,
            "decrease_dt_factor": 0.4,
            "increase_dt_factor": 1.2,
        }
    elif preset == "conservative":
        return {
            "trial_timestep_tol": 0.1,
            "conservation_tol": 0.001,
            "conservation_satisfied_tol": 0.001,
            "decrease_dt_factor": 0.1,
            "increase_dt_factor": 0.2,
        }
    else:
        raise ValueError(f"Unknown preset {preset}")


if __name__ == "__main__":
    rates = get_rates()
    error_threshold = 0.01
    tiny = 1e-20

    # parameters
    T = 1e6
    final_time = 100
    density = 0.1  # g /cm^3
    network_name = "twelve"
    solver_name = "pe"
    initial_gas_state = "neutral"
    timestepper_preset = "conservative"
    max_iters = 10000

    initial_conditions = get_initial_conditions(
        density, network_name, initial_gas_state
    )
    timestepper_settings = get_timestepper_settings(timestepper_preset)

    # timestepper_settings = {
    #     "trial_timestep_tol": 0.15,
    #     "conservation_tol": 0.005,
    #     "conservation_satisfied_tol": 0.005,
    #     "decrease_dt_factor": 0.4,
    #     "increase_dt_factor": 1.2,
    # }

    # timestepper_settings = (
    #     {
    #         "trial_timestep_tol": 0.1,
    #         "conservation_tol": 0.001,
    #         "conservation_satisfied_tol": 0.001,
    #         "decrease_dt_factor": 0.2,
    #         "increase_dt_factor": 0.4,
    #     },
    # )

    initial_conditions[-1] = (
        network_configs[network_name].calculate_energy_from_temp(
            initial_conditions, rates, T
        )
        / rates.chemistry_data.energy_units
    )

    solve_network(
        rates,
        initial_conditions,
        (0, final_time),
        T,
        error_threshold,
        timestepper_settings,
        check_error=False,
        solver_name=solver_name,
        network_name=network_name,
        max_iters=max_iters,
    )

    # start_time = time.time()
    # experiment(
    #     network_configs[network_name],
    #     solver_configs[solver_name],
    #     get_rates,
    #     density,
    #     (0, final_time),
    # )
    # end_time = time.time()

    # # Calculate elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")
