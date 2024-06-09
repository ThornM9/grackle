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
from odes import HI, HII, HeI, HeII, HeIII, e
import math
from collections import namedtuple

# data structure to store all config info about the reaction groups in this network
ReactionGroup = namedtuple(
    "ReactionGroup", ["ya_idx", "yb_idx", "yc_idx", "equilibrated"], defaults=[False]
)

# reaction group configs
rg_cfg = {
    0: ReactionGroup(1, 5, 0, False),
    1: ReactionGroup(3, 5, 2, False),
    2: ReactionGroup(4, 5, 3, False),
}


def get_kf(rg_num, rates, T):
    # TODO this can be moved into the reaction group config to be cleaner
    if rg_num == 0:
        return rates.k2(T)
    if rg_num == 1:
        return rates.k4(T)
    if rg_num == 2:
        return rates.k6(T)
    raise Exception("Invalid reaction group number")


def get_kr(rg_num, rates, T):
    # TODO this isn't implemented yet
    return get_kf(rg_num, rates, T) * 0.9


# gets the relevant ya, yb and yc for the class B reaction groups
def get_rg_abundances(rg_num, abundances):
    return (
        abundances[rg_cfg[rg_num].ya_idx],
        abundances[rg_cfg[rg_num].yb_idx],
        abundances[rg_cfg[rg_num].yc_idx],
    )


def is_equilibrated(rg_num, abundances, rates, T):
    if rg_num == -1:
        return False

    rg = rg_cfg[rg_num]
    if rg.equilibrated:
        return True

    eps = 0.01  # error tolerance to check equilibrium

    kf = get_kf(rg_num, rates, T)
    kr = get_kr(rg_num, rates, T)
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

    # differences between equilibrium values and current values
    ya_diff = abs(ya - ya_eq) / ya
    yb_diff = abs(yb - yb_eq) / yb
    yc_diff = abs(yc - yc_eq) / yc

    if ya_diff < eps and yb_diff < eps and yc_diff < eps:  # equilibrium check
        new_rg = ReactionGroup(rg.ya_idx, rg.yb_idx, rg.yc_idx, True)
        rg_cfg[rg_num] = new_rg
        return True

    return False


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


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


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
        * 0.0001
    )

    print(f"timestep: {dt *  3.1536e13}s")
    t0, tf = t_span
    n = int((tf - t0) / dt)
    print(f"number of time steps: {n}")
    num_eqns = len(equations)
    t = np.linspace(t0, tf, n + 1)
    y_values = np.zeros((num_eqns, n + 1))

    for i, initial_value in enumerate(initial_conditions):
        y_values[i, 0] = initial_value

    rate_values = np.zeros((num_eqns, n + 1))

    for i in range(n):
        abundances = y_values[:, i]
        for j, eq in enumerate(equations):
            er = eq.get_rates(
                *abundances,
                T,
            )
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

            y_values[j, i + 1] = y_values[j, i] + dt * rate

            # TODO adjust the equil rgs back to equil
            rate_values[j, i + 1] = rate
    print("solver final state: ", y_values[:, n - 1])
    return t, y_values, rate_values
