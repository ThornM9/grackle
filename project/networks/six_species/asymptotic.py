import numpy as np
from .odes import HI, HII, HeI, HeII, HeIII, e


USE_FLFD = False


def nn(x):
    if USE_FLFD:
        return max(x, 0)
    return x


# equation 12 in the asymptotic paper
def prediction_1(F_p, k_n, k_n_prev, F_p_prev, y_prev, dt):

    # if k_n == 0 or k_n_prev == 0 or dt == 0:
    #     print(k_n, k_n_prev, dt)
    #     print("FOUND")

    if k_n == 0:
        k_n = 1e-20
    if k_n_prev == 0:
        k_n_prev = 1e-20
    yn = F_p / k_n - ((1 / (k_n * dt)) * ((F_p / k_n) - (F_p_prev / k_n_prev)))

    return yn
    # return first_term - second_term


# equation 13 in the asymptotic paper
def prediction_2(F_p, k_n, dt, y_prev):
    yn = (1 / (1 + k_n * dt)) * (y_prev + (F_p * dt))
    return yn


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def asymptotic_methods_solver(equations, initial_conditions, t_span, T, rates):
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

    # these are arrays which store the value of their calculation from the previous timestep.
    # initial value is infinity so incorrect use of the array will hopefully be obvious in results
    k_ns = [float("inf") for _ in range(len(equations))]
    F_ps = [float("inf") for _ in range(len(equations))]

    using = 0
    for i in range(n):
        for j, eq in enumerate(equations):
            er = eq.get_rates(
                *y_values[:, i],
                T,
            )
            y_prev = y_values[j, i]
            k_n = sum(er.destruction_rates) * er.destruction_sign
            creation = sum(er.positive_fluxes)
            destruction = k_n * y_values[j, i]

            if i == 0 or abs(k_n * dt) >= 1:
                y_values[j, i + 1] = y_values[j, i] + dt * (creation - destruction)
            else:
                using += 1
                k_n_prev = k_ns[j]
                F_p_prev = F_ps[j]

                # y_values[j, i + 1] = prediction_1(
                #     creation, k_n, k_n_prev, F_p_prev, y_prev, dt
                # )
                y_values[j, i + 1] = prediction_2(creation, k_n, dt, y_prev)

            k_ns[j] = k_n
            F_ps[j] = creation

    print(f"used {using} / {n * len(equations)}")
    # print("solver final state: ", y_values[:, n - 1])
    # print(y_values)
    return t, y_values, rate_values
