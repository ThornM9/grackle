# step 1: separate into reaction groups
# step 2: determine which reaction groups are in equilibrium
# step 3: omit equilibrated reaction groups from calculations
# step 4: calculate next step using qss
# step 5: restore the equilibrium on groups that have been affected

### REACTION GROUPS
# Reaction: (HII + e <--> HI + photon)
# Forward rate: k2
# Reverse rate: TBD
#
# Reaction: (HeII + e <--> HeI + photon)
# Forward rate: k4
# Reverse rate: TBD
#
# Reaction: (HeIII + e <--> HeII + photon)
# Forward rate: k6
# Reverse rate: TBD


import numpy as np
from .odes import HI, HII, HeI, HeII, HeIII, e, Energy, get_cloudy_rates
import math
from collections import namedtuple
from .timesteppers import constant_timestepper, simple_timestepper

# data structure to store all config info about the reaction groups in this network
ReactionGroup = namedtuple(
    "ReactionGroup", ["ya_idx", "yb_idx", "yc_idx", "equilibrated"], defaults=[False]
)

# TODO move the get_kf and get_kr functions into this reaction group config and move it to odes.py since it is dependent on the specific network
# reaction group configs
rg_cfg = {
    0: ReactionGroup(1, 5, 0, False),
    1: ReactionGroup(3, 5, 2, False),
    2: ReactionGroup(4, 5, 3, False),
}


def get_kf(rg_num, rates, T):
    if rg_num == 0:
        return rates.k2(T)
    if rg_num == 1:
        return rates.k4(T)
    if rg_num == 2:
        return rates.k6(T)
    raise Exception("Invalid reaction group number")


def get_kr(rg_num):
    cloudy = get_cloudy_rates()
    if rg_num == 0:
        return cloudy["piHI"]
    if rg_num == 1:
        return cloudy["piHeI"]
    if rg_num == 2:
        return cloudy["piHeII"]
    raise Exception("Invalid reaction group number")


# gets the relevant ya, yb and yc for the class B reaction groups
def get_rg_abundances(rg_num, abundances):
    return (
        abundances[rg_cfg[rg_num].ya_idx],
        abundances[rg_cfg[rg_num].yb_idx],
        abundances[rg_cfg[rg_num].yc_idx],
    )


def calculate_equilibrium_values(rg_num, abundances, rates, T):
    # TODO this will need to be adjusted to add other reaction group types other than B
    kf = get_kf(rg_num, rates, T)
    kr = get_kr(rg_num)
    ya, yb, yc = get_rg_abundances(rg_num, abundances)

    c1 = yb - ya
    c2 = yb + yc

    a = -kf
    b = -(c1 * kf + kr)  # NOTE the paper says kb instead of kr here, possibly a typo
    c = kr * (c2 - c1)

    q = (4 * a * c) - (b**2)

    # equil values
    ya_eq = -(1 / 2 * a) * (b + math.sqrt(-q))
    yb_eq = c1 + ya_eq
    yc_eq = c2 - yb_eq

    return (ya_eq, yb_eq, yc_eq)


def is_equilibrated(rg_num, abundances, rates, T):
    # TODO needs adjustment for other reaction group types
    if rg_num == -1:
        return False

    rg = rg_cfg[rg_num]
    if rg.equilibrated:
        return True

    eps = 0.01  # error tolerance to check equilibrium

    ya, yb, yc = get_rg_abundances(rg_num, abundances)

    ya_eq, yb_eq, yc_eq = calculate_equilibrium_values(rg_num, abundances, rates, T)

    # differences between equilibrium values and current values
    ya_diff = abs(ya - ya_eq) / ya
    yb_diff = abs(yb - yb_eq) / yb
    yc_diff = abs(yc - yc_eq) / yc

    if ya_diff < eps and yb_diff < eps and yc_diff < eps:  # equilibrium check
        new_rg = ReactionGroup(rg.ya_idx, rg.yb_idx, rg.yc_idx, True)
        rg_cfg[rg_num] = new_rg
        return True

    return False


# equation 13 in the asymptotic paper
def prediction_2(F_p, k_n, dt, y_prev):
    yn = (1 / (1 + k_n * dt)) * (y_prev + (F_p * dt))
    return yn


# flux list is an array of calculated flux values (either creative or destructive)
# reaction_groups is an array of values for the reaction group it is part of (-1 if not part of any RG)
def filter_equilibrated_fluxes(flux_list, reaction_groups, abundances, rates, T):

    filtered_fluxes = []
    zipped = zip(flux_list, reaction_groups)
    for flux_val, rg_num in zipped:
        if is_equilibrated(rg_num, abundances, rates, T):
            continue

        filtered_fluxes.append(flux_val)

    return filtered_fluxes


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None), Energy(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII", "Electron"]


def pe_solver(equations, initial_conditions, t_span, T, rates):
    for eq in equations:
        eq.rates = rates

    HI_rate = odes[0].get_rates(*initial_conditions, T)
    dt = abs(
        initial_conditions[0]
        / (
            sum(HI_rate.positive_fluxes)
            - sum(HI_rate.destruction_rates) * initial_conditions[0]
        )
        * 0.1
    )

    print(f"timestep: {dt *  3.1536e13}s")
    t0, tf = t_span
    n = int((tf - t0) / dt)
    # print(f"number of time steps: {n}")
    num_eqns = len(equations)
    # t = np.linspace(t0, tf, n + 1)
    y_values = np.zeros((num_eqns, 1))

    for i, initial_value in enumerate(initial_conditions):
        y_values[i, 0] = initial_value

    # rate_values = np.zeros((num_eqns, n + 1))
    rate_values = None

    def update(equation_rates, y_values, i, dt):
        start_population = sum(y_values[:5, i])
        abundances = y_values[:, i]
        for j in range(len(equation_rates)):
            er = equation_rates[j]
            eq = equations[j]
            rgs = eq.get_reaction_groups()

            positive_fluxes = filter_equilibrated_fluxes(
                er.positive_fluxes, rgs.positive, abundances, rates, T
            )
            # can slightly improve speed by filtering before calculating the rate
            destruction_fluxes = filter_equilibrated_fluxes(
                er.destruction_rates, rgs.destruction, abundances, rates, T
            )

            y0 = y_values[j, i]
            k0 = sum(destruction_fluxes) * er.destruction_sign
            F_p = sum(positive_fluxes)
            F_m = k0 * y0

            rate = F_p - F_m

            # y_values[j, i + 1] = y_values[j, i] + dt * rate
            y_prev = y_values[j, i]

            if i == 0 or abs(k0 * dt) < 1:
                y_values[j, i + 1] = y_values[j, i] + dt * rate
            else:
                y_values[j, i + 1] = prediction_2(F_p, k0, dt, y_prev)

            # rate_values[j, i + 1] = rate

        # adjust the equil rgs back to equil
        restore_equilibrium_values = [[] for _ in range(len(equations))]
        for rg_num, rg in rg_cfg.items():
            if rg.equilibrated:
                ya_idx, yb_idx, yc_idx = rg.ya_idx, rg.yb_idx, rg.yc_idx

                ya_eq, yb_eq, yc_eq = calculate_equilibrium_values(
                    rg_num, y_values[:, i], rates, T
                )

                restore_equilibrium_values[ya_idx].append(ya_eq)
                restore_equilibrium_values[yb_idx].append(yb_eq)
                restore_equilibrium_values[yc_idx].append(yc_eq)

        for j in range(len(restore_equilibrium_values)):
            if j == 6:
                continue
            if len(restore_equilibrium_values[j]) > 0:
                y_values[j, i + 1] = sum(restore_equilibrium_values[j]) / len(
                    restore_equilibrium_values[j]
                )

        end_population = sum(
            y_values[:5, i + 1]
        )  # exclude electron density from population scaling
        # print("ratio: ", start_population / end_population)
        # y_values[:5, i + 1] = y_values[:5, i + 1] * start_population / end_population

    trial_timestep_tol = 0.1
    conservation_tol = 0.01
    conservation_satisfied_tol = 0.01
    decrease_dt_factor = 0.2
    increase_dt_factor = 0.4
    t, y_values, timestepper_data = simple_timestepper(
        y_values,
        equations,
        update,
        dt,
        t0,
        tf,
        trial_timestep_tol,
        conservation_tol,
        conservation_satisfied_tol,
        decrease_dt_factor,
        increase_dt_factor,
        T,
    )
    # print(rg_cfg)
    # print(get_kf(0, rates, T), get_kr(0))
    # print(get_kf(1, rates, T), get_kr(1))
    # print(get_kf(2, rates, T), get_kr(2))
    # print(calculate_equilibrium_values(0, y_values[:, 0], rates, T))
    # print(calculate_equilibrium_values(1, y_values[:, 0], rates, T))
    # print(calculate_equilibrium_values(2, y_values[:, 0], rates, T))

    # t, y_values = constant_timestepper(
    #     y_values,
    #     equations,
    #     update,
    #     dt,
    #     t0,
    #     tf,
    #     T,
    # )

    print(f"number of time steps: {len(t)}")

    return t, y_values, rate_values, timestepper_data
