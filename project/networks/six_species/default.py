import numpy as np


def dHI_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = rates.k2(T) * HII * e
    destruction = rates.k1(T) * e + rates.k57(T) * HI + rates.k58(T) * HeI / 4

    return creation - destruction * HI


def dHII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = (
        rates.k1(T) * HI * e + rates.k57(T) * HI * HI + rates.k58(T) * HI * HeI / 4
    )

    destruction = rates.k2(T) * e

    return creation - destruction * HII


def de_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = rates.k57(T) * HI * HI + rates.k58(T) * HI * HeI / 4

    destruction = -(
        rates.k1(T) * HI
        - rates.k2(T) * HII
        + rates.k3(T) * HeI / 4
        - rates.k6(T) * HeIII / 4
        + rates.k5(T) * HeII / 4
        - rates.k4(T) * HeII / 4
    )

    return creation - destruction * e


def dHeI_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = rates.k4(T) * HeII * e

    destruction = rates.k3(T) * e

    return creation - destruction * HeI


def dHeII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = rates.k3(T) * HeI * e + rates.k6(T) * HeIII * e

    destruction = rates.k4(T) * e + rates.k5(T) * e

    return creation - destruction * HeII


def dHeIII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = rates.k5(T) * HeII * e

    destruction = rates.k6(T) * e

    return creation - destruction * HeIII


odes = [dHI_dt, dHII_dt, dHeI_dt, dHeII_dt, dHeIII_dt, de_dt]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def euler_method_system(equations, initial_conditions, t_span, T, rates):
    dt = abs(initial_conditions[0] / dHI_dt(rates, *initial_conditions, T)) * 0.001
    print(f"timestep: {dt * 3.1536e13}s")

    # dt = abs(2 / dHI_dt(rates, *initial_conditions, T)) * 0.1
    # print(f"timestep: {timestep_to_seconds(dt)}")

    t0, tf = t_span
    n = int((tf - t0) / dt)
    print(f"number of time steps: {n}")
    num_eqns = len(equations)
    t = np.linspace(t0, tf, n + 1)
    values = np.zeros((num_eqns, n + 1))

    for i, initial_value in enumerate(initial_conditions):
        values[i, 0] = initial_value

    rate_values = np.zeros((num_eqns, n + 1))
    for i in range(n):
        for j, eq in enumerate(equations):
            values[j, i + 1] = values[j, i] + dt * eq(rates, *values[:, i], T)
            rate_values[j, i + 1] = eq(rates, *values[:, i], T)

    print("solver final state: ", values[:, n - 1])
    return t, values, rate_values
