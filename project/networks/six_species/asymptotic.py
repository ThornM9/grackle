import numpy as np
from .odes import HI, HII, HeI, HeII, HeIII, e, Energy
from .timesteppers import simple_timestepper, constant_timestepper


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


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None), Energy(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII", "Electron"]


def asymptotic_methods_solver(equations, initial_conditions, t_span, T, rates):
    for eq in equations:
        eq.rates = rates

    HI_rate = odes[0].get_rates(*initial_conditions, T)

    t0, tf = t_span
    num_eqns = len(equations)
    # TODO this solution just adds one row to the array at a time, computationally inefficient.
    # improve by pre allocating larger arrays as needed during runtime
    y_values = np.zeros((num_eqns, 1))

    for i in range(num_eqns):
        y_values[i, 0] = initial_conditions[i]

    dt = abs(
        initial_conditions[0]
        / (
            sum(HI_rate.positive_fluxes)
            - sum(HI_rate.destruction_rates) * initial_conditions[0]
        )
        * 0.001
    )

    # dt = (tf - t0) / 50000

    rate_values = np.zeros((num_eqns, 1))

    def update(equation_rates, y_values, i, dt):
        nonlocal rate_values
        rate_values = np.hstack((rate_values, np.zeros((num_eqns, 1))))

        for j in range(len(equation_rates)):
            er = equation_rates[j]

            y_prev = y_values[j, i]
            k_n = sum(er.destruction_rates) * er.destruction_sign
            creation = sum(er.positive_fluxes)
            if (
                j == 6
            ):  # todo this should just check if its an energy equation, new networks will break this
                y_values[j, i + 1] = y_values[j, i] + dt * (creation - k_n) / 0.1
                continue

            destruction = k_n * y_values[j, i]

            rate_values[j, i] = creation

            if i == 0 or abs(k_n * dt) < 1:
                y_values[j, i + 1] = y_values[j, i] + dt * (creation - destruction)
            else:
                y_values[j, i + 1] = prediction_2(creation, k_n, dt, y_prev)

    trial_timestep_tol = 0.1
    conservation_tol = 0.1
    conservation_satisfied_tol = 0.01
    decrease_dt_factor = 0.2
    increase_dt_factor = 0.2

    t, y_values = simple_timestepper(
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

    # t, y_values = constant_timestepper(
    #     y_values,
    #     equations,
    #     update,
    #     dt,
    #     t0,
    #     tf,
    #     T,
    # )

    # print(t.shape, y_values.shape)

    print(f"number of time steps: {len(t)}")

    return t, y_values, rate_values
