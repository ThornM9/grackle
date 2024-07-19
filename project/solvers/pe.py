# step 1: separate into reaction groups
# step 2: determine which reaction groups are in equilibrium
# step 3: omit equilibrated reaction groups from calculations
# step 4: calculate next step using qss
# step 5: restore the equilibrium on groups that have been affected


import numpy as np
from .networks.six_species_odes import (
    get_cloudy_rates,
)
import math
from collections import defaultdict
from .timesteppers import constant_timestepper, simple_timestepper
from .utils import calculate_population


def def_value():
    return False


rg_equilibrated = defaultdict(def_value)


# gets the relevant ya, yb and yc for the class B reaction groups
def get_rg_abundances(rg_cfg, rg_num, abundances):
    rg = rg_cfg[rg_num]
    if type(rg) is not list:
        raise Exception("Invalid reaction group config")

    rg_abundances = []
    for idx in rg:
        rg_abundances.append(abundances[idx])
    return rg_abundances


def calculate_equilibrium_values(rg_cfg, rg_num, abundances, rates, T):
    # NOTE this function will need to be modified if you want to add support for class C reaction groups in a future network
    kf = rg_cfg["get_kf"](rg_num, rates, T)
    kr = rg_cfg["get_kr"](rg_num, rates, T)
    rg_abundances = get_rg_abundances(rg_cfg, rg_num, abundances)

    if len(rg_abundances) == 3:
        # class B reaction
        ya, yb, yc = rg_abundances

        c1 = yb - ya
        c2 = yb + yc

        a = -kf
        b = -(c1 * kf + kr)
        c = kr * (c2 - c1)

        q = (4 * a * c) - (b**2)

        # equil values
        ya_eq = -(1 / 2 * a) * (b + math.sqrt(-q))
        yb_eq = c1 + ya_eq
        yc_eq = c2 - yb_eq

        return [ya_eq, yb_eq, yc_eq]

    elif len(rg_abundances) == 4:
        # could be class C or D, but we are assuming D since we don't have any C reactions in current networks
        ya, yb, yc, yd = rg_abundances
        c1 = ya - yb
        c2 = ya + yc
        c3 = ya + yd

        a = kr - kf
        b = -kr * (c2 + c3) + kf * c1
        c = kr * c2 * c3

        q = (4 * a * c) - (b**2)

        # equil values
        ya_eq = -(1 / 2 * a) * (b + math.sqrt(-q))
        yb_eq = c1 + ya_eq
        yc_eq = c2 - ya_eq
        yd_eq = c3 - ya_eq

        return [ya_eq, yb_eq, yc_eq, yd_eq]

    elif len(rg_abundances) == 5:
        # class E reaction
        ya, yb, yc, yd, ye = rg_abundances

        c1 = ya + ((yc + yd + ye) / 3)
        c2 = ya - yb
        c3 = yc - yd
        c4 = yc - ye

        beta = c1 - ((2 * c3) / 3) + (c4 / 3)
        alpha = c1 + ((c3 + c4) / 3)
        gamma = c1 + (c3 / 3) - (2 * c4 / 3)
        a = ((3 * c1 - ya) * kr) - kf
        b = c2 * kf - (alpha * beta + alpha * gamma + beta * gamma) * kr
        c = kr * alpha * beta * gamma

        q = (4 * a * c) - (b**2)

        # equil values
        ya_eq = -(1 / 2 * a) * (b + math.sqrt(-q))
        yb_eq = ya_eq - c2
        yc_eq = alpha - ya_eq
        yd_eq = beta - ya_eq
        ye_eq = gamma - ya_eq

        return [ya_eq, yb_eq, yc_eq, yd_eq, ye_eq]

    raise Exception("Invalid number of abundances")


def is_equilibrated(rg_cfg, rg_num, abundances, rates, T):
    return False
    if rg_num == -1 or rg_num == 4:  # TODO blocking rg 4 is temporary
        return False

    if rg_equilibrated[rg_num]:
        return True

    eps = 0.01  # error tolerance to check equilibrium
    near_zero_eps = 1e-25

    rg_abundances = get_rg_abundances(rg_cfg, rg_num, abundances)
    equilibrium_values = calculate_equilibrium_values(
        rg_cfg, rg_num, abundances, rates, T
    )
    near_equilibrium = True
    if len(rg_abundances) != len(equilibrium_values):
        raise Exception("Invalid number of equilibrium values")

    near_equil = [False for _ in range(len(rg_abundances))]
    # check that all abundances in the reaction are near their equilibrium values
    for i in range(len(rg_abundances)):
        if (
            abs(rg_abundances[i] - equilibrium_values[i]) / rg_abundances[i] < eps
            or abs(rg_abundances[i] - equilibrium_values[i]) <= near_zero_eps
        ):
            near_equil[i] = True

    near_equilibrium = all(near_equil)
    if near_equilibrium:
        print("\n\n")
        print(f"reaction group {rg_num} is near equilibrium")
        print("equilibrium values: ", equilibrium_values)
        print("actual values: ", rg_abundances)
        near_equil = [
            abs(rg_abundances[j] - equilibrium_values[j]) / rg_abundances[j] < eps
            or abs(rg_abundances[i] - equilibrium_values[i]) < near_zero_eps
            for j in range(len(rg_abundances))
        ]
        print("near equilibrium: ", near_equil)
        print("\n\n\n")
        rg_equilibrated[rg_num] = True

    return near_equilibrium


# equation 13 in the asymptotic paper
def prediction_2(F_p, k_n, dt, y_prev):
    yn = (1 / (1 + k_n * dt)) * (y_prev + (F_p * dt))
    return yn


# flux list is an array of calculated flux values (either creative or destructive)
# reaction_groups is an array of values for the reaction group it is part of (-1 if not part of any RG)
def filter_equilibrated_fluxes(
    rg_cfg, eq_idx, flux_list, reaction_groups, abundances, rates, T
):
    if len(flux_list) != len(reaction_groups):
        raise Exception("Invalid ODE reaction group setup at ODE index: ", eq_idx)

    filtered_fluxes = []
    zipped = zip(flux_list, reaction_groups)
    for flux_val, rg_num in zipped:
        if is_equilibrated(rg_cfg, rg_num, abundances, rates, T):
            continue

        filtered_fluxes.append(flux_val)

    return filtered_fluxes


def pe_solver(
    network_config,
    initial_conditions,
    t_span,
    T,
    rates,
    timestepper_settings,
    dt=None,
    max_iters=10000,
):
    equations = network_config.odes
    rg_cfg = network_config.reaction_group_config

    if dt is None:
        HI_rate = network_config.odes[0].get_rates(initial_conditions, T)
        dt = abs(
            initial_conditions[0]
            / (
                sum(HI_rate.positive_fluxes)
                - sum(HI_rate.destruction_rates) * initial_conditions[0]
            )
            * 0.0001
        )

    # print(f"timestep: {dt *  3.1536e13}s")
    t0, tf = t_span
    # n = int((tf - t0) / dt)
    # print(f"number of time steps: {n}")
    num_eqns = len(equations)
    # t = np.linspace(t0, tf, n + 1)
    y_values = np.zeros((num_eqns, 1))

    for i, initial_value in enumerate(initial_conditions):
        y_values[i, 0] = initial_value

    # rate_values = np.zeros((num_eqns, n + 1))
    rate_values = None

    def update(equation_rates, y_values, i, dt):
        if i > max_iters:
            raise Exception(f"Max iterations reached: {i} > {max_iters}")
        start_population = calculate_population(y_values, i, equations, False)
        abundances = y_values[:, i]
        for j in range(len(equation_rates)):
            er = equation_rates[j]
            eq = equations[j]
            rgs = eq.get_reaction_groups()

            # print(
            #     j,
            #     len(er.positive_fluxes),
            #     len(rgs.positive),
            #     len(er.destruction_rates),
            #     len(rgs.destruction),
            # )
            positive_fluxes = filter_equilibrated_fluxes(
                rg_cfg, j, er.positive_fluxes, rgs.positive, abundances, rates, T
            )
            # can slightly improve speed by filtering before calculating the rate
            destruction_rates = filter_equilibrated_fluxes(
                rg_cfg, j, er.destruction_rates, rgs.destruction, abundances, rates, T
            )

            y0 = y_values[j, i]
            k0 = sum(destruction_rates) * er.destruction_sign
            F_p = sum(positive_fluxes)
            F_m = k0 * y0

            rate = F_p - F_m

            y_prev = y_values[j, i]

            if i == 0 or abs(k0 * dt) < 1:
                y_values[j, i + 1] = y_values[j, i] + dt * rate
            else:
                y_values[j, i + 1] = prediction_2(F_p, k0, dt, y_prev)

            # tiny = 1e-40
            # y_values[j, i + 1] = max(y_values[j, i + 1], tiny)

        # adjust the equil rgs back to equil
        restore_equilibrium_values = [[] for _ in range(len(equations))]
        for rg_num in range(rg_cfg["rg_count"]):
            if rg_equilibrated[rg_num]:
                equilibrium_values = calculate_equilibrium_values(
                    rg_cfg, rg_num, y_values[:, i], rates, T
                )
                rg_idxs = rg_cfg[rg_num]
                for j, idx in enumerate(rg_idxs):
                    restore_equilibrium_values[idx].append(equilibrium_values[j])

        # if i % 50 == 0:
        #     print(restore_equilibrium_values)
        for j in range(len(restore_equilibrium_values)):
            if equations[j].is_energy:
                continue
            if len(restore_equilibrium_values[j]) > 0:
                # print(j, restore_equilibrium_values[j])
                y_values[j, i + 1] = sum(restore_equilibrium_values[j]) / len(
                    restore_equilibrium_values[j]
                )

        end_population = calculate_population(
            y_values, i, equations, False
        )  # exclude electron density from population scaling

        # print("ratio: ", start_population / end_population)
        # TODO uncomment? (but fix for different networks)
        # y_values[:5, i + 1] = y_values[:5, i + 1] * start_population / end_population

    t, y_values, timestepper_data = simple_timestepper(
        network_config,
        y_values,
        update,
        dt,
        t0,
        tf,
        timestepper_settings["trial_timestep_tol"],
        timestepper_settings["conservation_tol"],
        timestepper_settings["conservation_satisfied_tol"],
        timestepper_settings["decrease_dt_factor"],
        timestepper_settings["increase_dt_factor"],
        T,
    )

    # t, y_values = constant_timestepper(
    #     network_config,
    #     y_values,
    #     update,
    #     dt,
    #     t0,
    #     tf,
    #     T,
    # )
    # timestepper_data = None

    equilibrated_checks = [
        rg_equilibrated[rg_num] for rg_num in range(rg_cfg["rg_count"])
    ]
    print(equilibrated_checks)
    # print(rg_cfg)
    # print(get_kf(0, rates, T), get_kr(0))
    # print(get_kf(1, rates, T), get_kr(1))
    # print(get_kf(2, rates, T), get_kr(2))
    # print(calculate_equilibrium_values(rg_cfg, 0, y_values[:, 0], rates, T))
    # print(calculate_equilibrium_values(rg_cfg, 1, y_values[:, 0], rates, T))
    # print(calculate_equilibrium_values(rg_cfg, 2, y_values[:, 0], rates, T))
    # for i in range(len(rg_cfg.items()) - 2):
    #     if i == 4:
    #         continue
    #     equilibrium_values = calculate_equilibrium_values(
    #         rg_cfg, i, y_values[:, 0], rates, T
    #     )
    #     rg_abundances = get_rg_abundances(rg_cfg, i, y_values[:, 0])
    #     eps = 0.01  # error tolerance to check equilibrium

    #     print("reaction group: ", i)
    #     print("equilibrium values: ", equilibrium_values)
    #     print("actual values: ", rg_abundances)
    #     near_equil = [
    #         abs(rg_abundances[j] - equilibrium_values[j]) / rg_abundances[j] < eps
    #         for j in range(len(rg_abundances))
    #     ]
    #     print("near equilibrium: ", near_equil)
    #     print("\n\n\n")

    print(f"number of time steps: {len(t)}")

    return t, y_values, rate_values, timestepper_data
