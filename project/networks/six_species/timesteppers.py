import numpy as np
from .odes import calculate_temp_from_energy

### HELPERS ###


def calculate_trial_timestep(equation_rates, y_values, i, trial_timestep_tol):
    max_dts = []
    for j in range(len(equation_rates)):
        er = equation_rates[j]
        k_n = sum(er.destruction_rates) * er.destruction_sign
        creation = sum(er.positive_fluxes)
        destruction = k_n * y_values[j, i]
        y = y_values[j, i]

        if creation > destruction:
            new_y = y + y * trial_timestep_tol
        elif creation < destruction:
            new_y = y - y * trial_timestep_tol
        else:
            continue
        max_dt = (new_y - y) / (creation - destruction)
        max_dts.append(max_dt)

    if len(max_dts) == 0:
        return float("inf")
    return max(max_dts)


### TIMESTEPPERS ###


def constant_timestepper(y_values, equations, update_func, initial_dt, t0, tf, T):
    dt = initial_dt
    n = int((tf - t0) / dt)
    print(f"number of time steps: {n}")
    t = np.linspace(t0, tf, n + 1)
    new_y_values = np.zeros((len(equations), n + 1))
    new_y_values[:, 0] = y_values[:, 0]
    y_values = new_y_values

    dt = initial_dt
    prev_T = T
    for i in range(n):
        T = calculate_temp_from_energy(*y_values[:, i], equations[0].rates, prev_T)
        # calculate rates and fluxes
        ers = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(*y_values[:, i], T)
            ers.append(er)
            # if j == 6:
            #     print(sum(er.destruction_rates), dt)

        try:
            # update populations
            update_func(ers, y_values, i, dt)
        except Exception as e:
            print(e)
            raise e
            return t, y_values

        prev_T = T

    return t, y_values


def simple_timestepper(
    y_values,
    equations,
    update_func,
    initial_dt,
    t0,
    tf,
    trial_timestep_tol,
    conservation_tol,  # if the population difference is above this tolerance, decrease dt
    conservation_satisfied_tol,  # if the population difference is below this tolerance, increase dt
    decrease_dt_factor,
    increase_dt_factor,
    T,
):
    dt = initial_dt
    t = np.array([t0])
    curr_t = t0
    prev_T = T
    i = 0
    while curr_t < tf:
        if i % 250 == 0:
            print(f"timestep: {i}, time: {curr_t}")
        # add a row
        y_values = np.hstack((y_values, np.zeros((len(equations), 1))))

        # T = calculate_temp_from_energy(*y_values[:, i], prev_T)
        # print(prev_T)
        # calculate rates and fluxes
        ers = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(*y_values[:, i], prev_T)
            ers.append(er)

        # prev_T = T
        # calculate trial timestep
        trial_dt = calculate_trial_timestep(ers, y_values, i, trial_timestep_tol)

        # if trial_dt < dt:
        #     print("using trial")
        dt = min(dt, trial_dt)
        t = np.append(t, t[-1] + dt)

        # update pops with asym algo
        try:
            # update populations
            update_func(ers, y_values, i, dt)
        except Exception as e:
            y_values[:, -1] = 0
            raise e
            return t, y_values

        # check populations are conserved
        old_pop = np.sum(y_values[:, i])
        new_pop = np.sum(y_values[:, i + 1])

        # increase/decrease dt if needed
        population_difference = abs(new_pop - old_pop) / old_pop
        # print("pop diff: ", population_difference)

        changed_dt = False

        if curr_t < 0.05:
            if population_difference > conservation_tol:
                changed_dt = True
                dt = dt * (1 - decrease_dt_factor)

            if population_difference < conservation_satisfied_tol:
                changed_dt = True
                dt = dt * (1 + increase_dt_factor)

        # recalculate populations with asym algo
        if changed_dt:
            y_values[:, i + 1] = 0
            try:
                # update populations
                update_func(ers, y_values, i, dt)
            except Exception as e:
                print(e)
                y_values[:, -1] = 0
                return t, y_values

        curr_t += dt
        i += 1

    return t, y_values
