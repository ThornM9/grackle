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


def calculate_trial_timestep(equation_rates, y_values, i, trial_timestep_tolerance):
    max_dts = []
    for j in range(len(equation_rates)):
        er = equation_rates[j]
        k_n = sum(er.destruction_rates) * er.destruction_sign
        creation = sum(er.positive_fluxes)
        destruction = k_n * y_values[j, i]
        y = y_values[j, i]

        if creation > destruction:
            new_y = y + y * trial_timestep_tolerance
        elif creation < destruction:
            new_y = y - y * trial_timestep_tolerance

        if creation == destruction:
            continue
        max_dt = (new_y - y) / (creation - destruction)
        max_dts.append(max_dt)

    if len(max_dts) == 0:
        return float("inf")
    return max(max_dts)


def update(equation_rates, y_values, i, dt):
    for j in range(len(equation_rates)):
        er = equation_rates[j]

        y_prev = y_values[j, i]
        k_n = sum(er.destruction_rates) * er.destruction_sign
        creation = sum(er.positive_fluxes)
        destruction = k_n * y_values[j, i]

        if i == 0 or abs(k_n * dt) >= 1:
            y_values[j, i + 1] = y_values[j, i] + dt * (creation - destruction)
        else:
            y_values[j, i + 1] = prediction_2(creation, k_n, dt, y_prev)


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

    t0, tf = t_span
    num_eqns = len(equations)
    # TODO this solution just adds one row to the array at a time, computationally inefficient.
    # improve by pre allocating larger arrays as needed during runtime
    y_values = np.zeros((num_eqns, 1))

    for i, initial_value in enumerate(initial_conditions):
        y_values[i, 0] = initial_value

    dt = abs(
        initial_conditions[0]
        / (
            sum(HI_rate.positive_fluxes)
            - sum(HI_rate.destruction_rates) * initial_conditions[0]
        )
        * 0.0001
    )

    t = np.array([t0])
    curr_t = t0
    trial_timestep_tolerance = 0.1
    pop_conservation_tolerance = 0.01
    pop_conservation_strongly_satisfied_tolerance = 0.001
    decrease_dt_factor = 0.2
    increase_dt_factor = 0.2
    i = 0
    while curr_t < tf:
        if i < 100:
            print(f"timestep: {dt *  3.1536e13}s")
        # add a row
        y_values = np.hstack((y_values, np.zeros((num_eqns, 1))))

        # calculate rates and fluxes
        ers = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(*y_values[:, i], T)
            ers.append(er)

        # calculate trial timestep
        trial_dt = calculate_trial_timestep(ers, y_values, i, trial_timestep_tolerance)

        dt = min(dt, trial_dt)

        # update pops with asym algo
        update(ers, y_values, i, dt)

        # check populations are conserved
        old_pop = np.sum(y_values[:, i])
        new_pop = np.sum(y_values[:, i + 1])

        # increase/decrease dt if needed
        population_difference = abs(new_pop - old_pop) / old_pop
        if i < 100:
            print(population_difference)
        if population_difference > pop_conservation_tolerance:
            dt = dt * (1 - decrease_dt_factor)

        if population_difference < pop_conservation_strongly_satisfied_tolerance:
            dt = dt * (1 + increase_dt_factor)

        # recalculate populations with asym algo
        # TODO check if the dt has changed before doing this
        y_values[:, i + 1] = 0
        update(ers, y_values, i, dt)

        t = np.append(t, t[-1] + dt)
        curr_t += dt
        i += 1

    print(f"number of time steps: {i}")

    return t, y_values, None
