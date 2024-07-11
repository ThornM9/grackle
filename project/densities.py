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
from networks.six_species.default import (
    odes as default_odes,
    euler_method_system as default_euler,
    species_names as default_species_names,
)
from networks.six_species.flfd import (
    odes as flfd_odes,
    flux_limited_solver,
    species_names as flfd_species_names,
)
from networks.six_species.asymptotic import (
    odes as asymptotic_odes,
    asymptotic_methods_solver,
    species_names as asymptotic_species_names,
)
from networks.six_species.qss import (
    odes as qss_odes,
    qss_methods_solver,
    species_names as qss_species_names,
)
from networks.six_species.pe import (
    odes as pe_odes,
    pe_solver,
    species_names as pe_species_names,
)
from networks.six_species.odes import (
    calculate_temp_from_energy,
    calculate_energy_from_temp,
    calculate_mu,
)
import similaritymeasures
from test import test_equilibrium
from networks.six_species.odes import get_cloudy_rates
from plotting import (
    plot_solution,
    plot_prediction,
    plot_rate_values,
    plot_mu,
    plot_energy_and_temperature,
    plot_timestepper_data,
)

# todo reduce number density for pe to test that partial equilibrium works

# TODO species names can just be the one variable
# map a config name to the solver, list of odes and the species names
solver_configs = {
    "default": (default_euler, default_odes, default_species_names),
    "asymptotic": (
        asymptotic_methods_solver,
        asymptotic_odes,
        asymptotic_species_names,
    ),
    "qss": (qss_methods_solver, qss_odes, qss_species_names),
    "pe": (pe_solver, pe_odes, pe_species_names),
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


# this function is just a wrapper to pass to scipy to solve for an example solution
def network(t, y, odes, T, rates):
    results = []
    for ode in odes:
        results.append(ode(rates, *y, T))

    return results


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
    network_name="default",
    plot_results=True,
    check_error=True,
):
    if network_name not in solver_configs:
        raise ValueError(f"Network {network_name} not found")

    solver, odes, species_names = solver_configs[network_name]

    print("solution solver")
    solution = solve_ivp(
        network,
        t_span,
        initial_conditions[:-1],
        method="BDF",
        args=(default_odes, T, rates),
    )
    pred_t = solution.t
    pred_y = solution.y
    print(f"number of timesteps in solution: {len(pred_t)}")

    print("custom solver")
    exp_t, exp_y, rate_values, timestepper_data = solver(
        odes, initial_conditions, t_span, T, rates
    )

    print("final solution state: ", pred_y[:, -1])
    print("final solver state: ", exp_y[:, -1])

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

    plot_solution(pred_t, pred_y, species_names, network_name)
    plot_prediction(exp_t, exp_y, species_names, network_name)
    plot_energy_and_temperature(exp_t, exp_y, rates, network_name)
    plot_mu(exp_t, exp_y, rates, network_name)
    plot_timestepper_data(exp_t, timestepper_data, network_name)


if __name__ == "__main__":
    rates = get_rates()
    T = 1e6
    density = 0.1  # g /cm^3
    error_threshold = 0.01
    tiny = 1e-20
    initial_conditions = [
        0.76 * density,
        tiny * density,
        0.24 * density,
        tiny * density,
        tiny * density,
        tiny * density,
        None,  # energy
    ]

    # initial_conditions = [
    #     tiny * density,
    #     0.76 * density,
    #     tiny * density,
    #     0.02 * density,
    #     0.22 * density,
    #     0.87998 * density,
    #     None,  # energy
    # ]

    initial_conditions[-1] = (
        calculate_energy_from_temp(*initial_conditions, rates, T)
        / rates.chemistry_data.energy_units
    )

    solve_network(
        rates,
        initial_conditions,
        (0, 100),
        T,
        error_threshold,
        check_error=False,
        network_name="pe",
    )
