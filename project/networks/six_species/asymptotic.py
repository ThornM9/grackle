import numpy as np


def nn(x):
    return max(x, 0)


def dHI_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = nn(rates.k2(T) * HII * e)
    destruction = (
        nn(rates.k1(T) * e * HI)
        + nn(rates.k57(T) * HI * HI)
        + nn((rates.k58(T) * HeI / 4) * HI)
    )

    return creation - destruction


def dHII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = (
        nn(rates.k1(T) * HI * e)
        + nn(rates.k57(T) * HI * HI)
        + nn(rates.k58(T) * HI * HeI / 4)
    )

    destruction = nn(rates.k2(T) * e * HII)

    return creation - destruction


def de_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = nn(rates.k57(T) * HI * HI) + nn(rates.k58(T) * HI * HeI / 4)

    destruction = -(
        nn(rates.k1(T) * HI * e)
        - nn(rates.k2(T) * HII * e)
        + nn((rates.k3(T) * HeI / 4) * e)
        - nn((rates.k6(T) * HeIII / 4) * e)
        + nn((rates.k5(T) * HeII / 4) * e)
        - nn((rates.k4(T) * HeII / 4) * e)
    )

    return creation - destruction


def dHeI_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = nn(rates.k4(T) * HeII * e)

    destruction = nn(rates.k3(T) * e * HeI)

    return creation - destruction


def dHeII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = nn(rates.k3(T) * HeI * e + rates.k6(T) * HeIII * e)

    destruction = nn(rates.k4(T) * e * HeII) + nn(rates.k5(T) * e * HeII)

    return creation - destruction


def dHeIII_dt(rates, HI, HII, HeI, HeII, HeIII, e, T):
    creation = nn(rates.k5(T) * HeII * e)

    destruction = nn(rates.k6(T) * e * HeIII)

    return creation - destruction


odes = [dHI_dt, dHII_dt, dHeI_dt, dHeII_dt, dHeIII_dt, de_dt]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def flux_limited_solver(equations, initial_conditions, t_span, T, rates):
    dt = abs(initial_conditions[0] / dHI_dt(rates, *initial_conditions, T)) * 0.001

    print(f"timestep: {dt *  3.1536e13}s")
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
