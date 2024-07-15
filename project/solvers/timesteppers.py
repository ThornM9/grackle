import numpy as np
from .utils import calculate_population

### HELPERS ###


def calculate_trial_timestep(
    equation_rates, y_values, i, trial_timestep_tol, curr_t, t0, tf, equations
):
    # TODO this whole function is very brittle and will probably break for other networks
    # max_dts = []
    # for j in range(len(equation_rates)):
    #     er = equation_rates[j]
    #     k_n = sum(er.destruction_rates) * er.destruction_sign
    #     creation = sum(er.positive_fluxes)
    #     destruction = k_n * y_values[j, i]
    #     y = y_values[j, i]

    #     if creation > destruction:
    #         new_y = y + y * trial_timestep_tol
    #     elif creation < destruction:
    #         new_y = y - y * trial_timestep_tol
    #     else:
    #         continue
    #     max_dt = (new_y - y) / (creation - destruction)
    #     max_dts.append(max_dt)

    # if len(max_dts) == 0:
    #     return float("inf")
    # return max(max_dts)

    # dtit(i) = min(abs(0.1_DKIND*de(i,j,k)/dedot(i)),
    #  &              abs(0.1_DKIND*HI(i,j,k)/HIdot(i)),
    #  &              dt-ttot(i), 0.5_DKIND*dt)

    tiny = 1e-20
    HI_er = equation_rates[0]
    HI = y_values[0, i]
    HI_k_n = sum(HI_er.destruction_rates) * HI_er.destruction_sign
    HI_creation = sum(HI_er.positive_fluxes)
    HI_destruction = HI_k_n * HI
    HI_rate = HI_creation - HI_destruction
    # if abs(HI_rate) < tiny:
    #     HI_rate = tiny

    e_er = equation_rates[5]
    e = y_values[5, i]
    e_k_n = sum(e_er.destruction_rates) * e_er.destruction_sign
    e_creation = sum(e_er.positive_fluxes)
    e_destruction = e_k_n * e
    e_rate = e_creation - e_destruction
    # if abs(e_rate) < tiny:
    #     e_rate = tiny

    if HI_rate == 0:
        HI_dt = float("inf")
    else:
        HI_dt = trial_timestep_tol * HI / HI_rate
    if e_rate == 0:
        e_dt = float("inf")
    else:
        e_dt = trial_timestep_tol * e / e_rate

    old_pop = calculate_population(y_values, i, equations)
    max_diff = old_pop * trial_timestep_tol

    if HI / old_pop < 0.01:
        new_HI = HI + np.sign(HI_rate) * max_diff
        HI_dt = max(HI_dt, abs((new_HI - HI) / HI_rate))

    if e / old_pop < 0.01:
        new_e = e + np.sign(e_rate) * max_diff
        e_dt = max(e_dt, abs((new_e - e) / e_rate))

    # TODO can we reverse the asymptotic approximation to calculate theses? might improve timestepping
    # TODO how do we solve the problem of when the abundances are near zero?
    max_dt = min(
        abs(e_dt),
        abs(HI_dt),
        (tf - t0) - curr_t,
        0.5 * (tf - t0),
    )
    return max_dt

    # TODO try calculating the tolerance relative to overall density rather than individual species
    old_pop = calculate_population(y_values, i, equations)
    max_diff = old_pop * trial_timestep_tol

    new_HI = HI + np.sign(HI_rate) * max_diff
    new_e = e + np.sign(e_rate) * max_diff

    max_HI_dt = abs((new_HI - HI) / HI_rate)
    max_e_dt = abs((new_e - e) / e_rate)

    # print(max_HI_dt, max_e_dt, max_dt)
    print("new solution: ", min(max_HI_dt, max_e_dt), "old solution: ", max_dt)
    return min(min(max_HI_dt, max_e_dt), max_dt)

    #     if creation > destruction:
    #         new_y = y + y * trial_timestep_tol
    #     elif creation < destruction:
    #         new_y = y - y * trial_timestep_tol
    #     else:
    #         continue
    #     max_dt = (new_y - y) / (creation - destruction)
    #     max_dts.append(max_dt)


### TIMESTEPPERS ###


def constant_timestepper(network_cfg, y_values, update_func, initial_dt, t0, tf, T):
    equations = network_cfg.odes

    dt = initial_dt
    n = int((tf - t0) / dt)
    print(f"number of time steps: {n}")
    t = np.linspace(t0, tf, n + 1)
    new_y_values = np.zeros((len(equations), n + 1))
    new_y_values[:, 0] = y_values[:, 0]
    y_values = new_y_values

    dt = initial_dt
    for i in range(n):
        T = network_cfg.calculate_temp_from_energy(y_values[:, i], equations[0].rates)
        # calculate rates and fluxes
        ers = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(y_values[:, i], T)
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

    return t, y_values


def simple_timestepper(
    network_cfg,
    y_values,
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
    equations = network_cfg.odes
    timestepper_data = {
        "i": [],
        "dt": [],
        "old_dt": [],
        "trial_step_used": [],
        "trial_step": [],
        "conservation_unsatisfied": [],
        "conservation_satisfied": [],
    }

    dt = initial_dt
    t = np.array([t0])
    curr_t = t0
    prev_T = T
    i = 0
    trial_used = 0
    conservation_unsatisfied = 0
    conservation_satisfied = 0
    while curr_t < tf:
        timestepper_data["old_dt"].append(dt)

        if i % 250 == 0:
            print(f"timestep: {i}, time: {curr_t}, dt: {dt}, trial used: {trial_used}")
        # add a row
        y_values = np.hstack((y_values, np.zeros((len(equations), 1))))

        gamma = network_cfg.calculate_gamma(y_values[:, i], equations[0].rates)
        T = network_cfg.calculate_temp_from_energy(
            y_values[:, i], equations[0].rates, gamma
        )

        # print(prev_T)
        # calculate rates and fluxes
        ers = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(y_values[:, i], prev_T)
            ers.append(er)

        prev_T = T
        # calculate trial timestep
        trial_dt = calculate_trial_timestep(
            ers, y_values, i, trial_timestep_tol, curr_t, t0, tf, equations
        )
        if trial_dt < dt:
            trial_used += 1

        # if trial_dt < dt:
        #     print("using trial")
        dt = min(dt, trial_dt)

        # update pops with asym algo
        try:
            # update populations
            update_func(ers, y_values, i, dt)
        except Exception as e:
            y_values[:, -1] = 0
            raise e
            return t, y_values

        # check populations are conserved
        old_pop = calculate_population(y_values, i, equations)
        new_pop = calculate_population(y_values, i + 1, equations)

        # increase/decrease dt if needed
        population_difference = abs(new_pop - old_pop) / old_pop
        # print("pop diff: ", population_difference)

        changed_dt = False

        if population_difference > conservation_tol:
            conservation_unsatisfied += 1
            changed_dt = True
            dt = dt * (1 - decrease_dt_factor)

        if population_difference < conservation_satisfied_tol:
            conservation_satisfied += 1
            changed_dt = True
            dt = dt * (1 + increase_dt_factor)

        dt = min(dt, (tf - t0) - curr_t)
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

        # add timestepper data
        timestepper_data["i"].append(i)
        timestepper_data["dt"].append(dt)
        timestepper_data["trial_step_used"].append(trial_used)
        timestepper_data["trial_step"].append(trial_dt)
        timestepper_data["conservation_satisfied"].append(conservation_satisfied)
        timestepper_data["conservation_unsatisfied"].append(conservation_unsatisfied)

        t = np.append(t, t[-1] + dt)
        curr_t += dt
        i += 1

    timestepper_data["i"].append(i)
    timestepper_data["dt"].append(dt)
    timestepper_data["old_dt"].append(timestepper_data["old_dt"][-1])
    timestepper_data["trial_step_used"].append(trial_used)
    timestepper_data["trial_step"].append(trial_dt)
    timestepper_data["conservation_satisfied"].append(conservation_satisfied)
    timestepper_data["conservation_unsatisfied"].append(conservation_unsatisfied)

    print(f"timestep: {i}, time: {curr_t}, dt: {dt}, trial used: {trial_used}")
    return t, y_values, timestepper_data
