import itertools
import copy
from solvers.utils import network
from scipy.integrate import solve_ivp
import concurrent.futures
import multiprocessing
import logging
from plotting import plot_solution


# Configure logging
logging.basicConfig(level=logging.DEBUG, filename="debug.log", filemode="w")
logger = logging.getLogger(__name__)


def test_solver(
    network_config,
    get_rates,
    solver,
    t_span,
    initial_conditions_tuple,
    T,
    trial_timestep_tol,
    conservation_tol,
    conservation_satisfied_tol,
    decrease_dt_factor,
    increase_dt_factor,
    initial_dt_factor,
    minimum_found,
    minimum_found_lock,
    file_lock,
):
    solutions = {}
    rates = get_rates()
    for eq in network_config.odes:
        eq.rates = rates
    logger.debug("Starting test_solver")
    if (
        conservation_satisfied_tol > conservation_tol
        or decrease_dt_factor > increase_dt_factor
    ):
        return
    initial_conditions, gas_ionisation_state = initial_conditions_tuple
    initial_conditions = copy.deepcopy(initial_conditions)
    key = f"{gas_ionisation_state}_{T}"
    log_file = "timestepper_log.txt"
    error_threshold = 0.01
    initial_conditions[-1] = (
        network_config.calculate_energy_from_temp(initial_conditions, rates, T)
        / rates.chemistry_data.energy_units
    )
    if key not in solutions:
        solution = solve_ivp(
            network,
            t_span,
            initial_conditions,
            method="BDF",
            args=(network_config, T),
        )
        pred_y = solution.y
        logger.debug(f"Solution found for {key} of length: {len(pred_y)}")
        final_solver_y = pred_y[:, -1]
        solutions[key] = pred_y
    else:
        pred_y = solutions[key]
        final_solver_y = pred_y[:, -1]

    max_iters = 10000

    HI_rate = network_config.odes[0].get_rates(initial_conditions, T)
    initial_dt = abs(
        initial_conditions[0]
        / (
            sum(HI_rate.positive_fluxes)
            - sum(HI_rate.destruction_rates) * initial_conditions[0]
        )
        * initial_dt_factor
    )
    timestepper_settings = {
        "trial_timestep_tol": trial_timestep_tol,
        "conservation_tol": conservation_tol,
        "conservation_satisfied_tol": conservation_satisfied_tol,
        "decrease_dt_factor": decrease_dt_factor,
        "increase_dt_factor": increase_dt_factor,
    }
    try:
        exp_t, exp_y, rate_values, timestepper_data = solver(
            network_config,
            initial_conditions,
            t_span,
            T,
            rates,
            initial_dt,
            timestepper_settings,
            max_iters,
        )
        final_exp_y = exp_y[:, -1]
        err = abs(final_exp_y - final_solver_y) / final_solver_y
        if all(e < 0.1 for e in err):
            logger.debug(
                f"found good solution with error {err} for settings: {timestepper_settings}"
            )
            with minimum_found_lock:
                minimum_found[key] = (
                    min(minimum_found[key], len(exp_t))
                    if key in minimum_found
                    else len(exp_t)
                )
            # with open(log_file, "a") as file:
            #     data_to_append = [
            #         f"Gas ionisation state: {gas_ionisation_state}\n",
            #         f"Temperature: {T}\n",
            #         f"Timestepper settings: {timestepper_settings}\n",
            #         f"Number of timesteps: {len(exp_t)}\n",
            #         f"Number of solution timesteps: {len(pred_y)}\n",
            #         "\n\n",
            #     ]
            #     file.writelines(data_to_append)
            with file_lock:
                with open(log_file, "a") as file:
                    data_to_append = [
                        f"Gas ionisation state: {gas_ionisation_state}\n",
                        f"Temperature: {T}\n",
                        f"Timestepper settings: {timestepper_settings}\n",
                        f"Number of timesteps: {len(exp_t)}\n",
                        f"Number of solution timesteps: {len(pred_y)}\n",
                        "\n\n",
                    ]
                    file.writelines(data_to_append)

    except Exception as e:
        logger.error(e)
        pass


import pickle


def pickle_test(obj, name):
    try:
        pickle.dumps(obj)
        logger.debug(f"Pickling {name} succeeded.")
    except pickle.PicklingError as e:
        logger.error(f"Pickling {name} failed: {e}")


def experiment(
    network_config,
    solver,
    get_rates,
    density,
    t_span,
):
    tiny = 1e-20
    log_file = "timestepper_log.txt"

    with open(log_file, "w") as file:  # clear the file
        pass

    initial_conditions_list = [
        (
            [
                0.76 * density,
                tiny * density,
                0.24 * density,
                tiny * density,
                tiny * density,
                tiny * density,
                None,
            ],
            "neutral",
        ),  # fully neutral 6 species gas
        (
            [
                tiny * density,
                0.76 * density,
                tiny * density,
                0.02 * density,
                0.22 * density,
                0.87998 * density,
                None,
            ],
            "ionised",
        ),  # fully ionised 6 species gas
    ]

    temps = [1e3, 5e4, 1e6]
    # settings = [
    #     (
    #         {
    #             "trial_timestep_tol": 0.15,
    #             "conservation_tol": 0.005,
    #             "conservation_satisfied_tol": 0.005,
    #             "decrease_dt_factor": 0.4,
    #             "increase_dt_factor": 1.2,
    #         },
    #         "aggressive",
    #     ),
    #     (
    #         {
    #             "trial_timestep_tol": 0.1,
    #             "conservation_tol": 0.001,
    #             "conservation_satisfied_tol": 0.001,
    #             "decrease_dt_factor": 0.2,
    #             "increase_dt_factor": 0.4,
    #         },
    #         "conservative",
    #     ),
    # ]
    trial_timestep_tols = [0.1, 0.15]
    conservation_tols = [0.001, 0.01]
    conservation_satisfied_tols = [0.001, 0.01]
    decrease_dt_factors = [0.2, 0.4, 0.6]
    increase_dt_factors = [0.4, 0.8, 1.2, 1.6, 2.0]
    initial_dt_factors = [0.0001]

    # temps = [1e3]
    # trial_timestep_tols = [0.1]
    # conservation_tols = [0.001]
    # conservation_satisfied_tols = [0.001]
    # decrease_dt_factors = [0.2]
    # increase_dt_factors = [0.4]
    # initial_dt_factors = [0.0001]

    all_combinations = list(
        itertools.product(
            initial_conditions_list,
            temps,
            trial_timestep_tols,
            conservation_tols,
            conservation_satisfied_tols,
            decrease_dt_factors,
            increase_dt_factors,
            # settings,
            initial_dt_factors,
        )
    )
    logger.debug(f"number of combinations: {len(all_combinations)}")

    with multiprocessing.Manager() as manager:
        minimum_found = manager.dict()
        minimum_found_lock = manager.Lock()
        file_lock = manager.Lock()

        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for combination in all_combinations:
                ic = combination[0]
                T = combination[1]
                # trial_timestep_tol = combination[2][0]["trial_timestep_tol"]
                # conservation_tol = combination[2][0]["conservation_tol"]
                # conservation_satisfied_tol = combination[2][0][
                #     "conservation_satisfied_tol"
                # ]
                # decrease_dt_factor = combination[2][0]["decrease_dt_factor"]
                # increase_dt_factor = combination[2][0]["increase_dt_factor"]

                initial_dt_factor = combination[3]

                futures.append(
                    executor.submit(
                        test_solver,
                        network_config,
                        get_rates,
                        solver,
                        t_span,
                        # ic,
                        # T,
                        # trial_timestep_tol,
                        # conservation_tol,
                        # conservation_satisfied_tol,
                        # decrease_dt_factor,
                        # increase_dt_factor,
                        # initial_dt_factor,
                        *combination,
                        minimum_found,
                        minimum_found_lock,
                        file_lock,
                    )
                )

            for future in futures:
                future.result()
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")

        rates = get_rates()
        for eq in network_config.odes:
            eq.rates = rates
        solutions = {}
        for ic in initial_conditions_list:
            for T in temps:
                key = f"{ic[1]}_{T}"
                initial_conditions = copy.deepcopy(ic[0])

                initial_conditions[-1] = (
                    network_config.calculate_energy_from_temp(
                        initial_conditions, rates, T
                    )
                    / rates.chemistry_data.energy_units
                )
                print(initial_conditions)
                solution = solve_ivp(
                    network,
                    t_span,
                    initial_conditions,
                    method="BDF",
                    args=(network_config, T),
                )
                print(len(solution.y))
                solutions[key] = solution.y

        logger.debug(f"Minimum stable timesteps found: {dict(minimum_found)}")
        print(f"Minimum stable timesteps found: {dict(minimum_found)}")
        solution_timesteps = {k: len(v[0]) for k, v in solutions.items()}
        logger.debug(f"Solutions timesteps found: {solution_timesteps}")
        print(f"Solutions timesteps found: {solution_timesteps}")
        print(f"Total combination number: {len(all_combinations)}")


# Example usage (assuming network_config, solver, rates, density, t_span are defined):
# experiment(network_config, solver, rates, density, t_span)
