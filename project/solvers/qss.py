import numpy as np
from collections import namedtuple
from .networks.six_species_odes import HI, HII, HeI, HeII, HeIII, e
from .timesteppers import simple_timestepper, constant_timestepper

Rates = namedtuple(
    "Rates", ["positive_fluxes", "destruction_rates", "destruction_sign"], defaults=[1]
)


# equation 4 in qss paper
def alpha(r):
    return ((160 * r**3) + (60 * r**2) + (11 * r) + 1) / (
        (360 * r**3) + (60 * r**2) + (12 * r) + 1
    )


# predictor from equation 5 in qss paper
def predictor(y0, k0, F_p0, F_m0, dt):
    a = alpha(1 / (k0 * dt))

    yp = y0 + ((dt * (F_p0 - F_m0)) / (1 + a * k0 * dt))
    return yp


# corrector from equation 5 in qss paper
def corrector(y0, k0, kp, F_p0, F_pp, dt):
    k_avg = 0.5 * (k0 + kp)
    a_avg = alpha(1 / (k_avg * dt))
    F_p_avg = a_avg * F_pp + (1 - a_avg) * F_p0
    yc = y0 + ((F_p_avg - k_avg * y0) / (1 + a_avg * k_avg * dt))

    return yc


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def qss_methods_solver(equations, initial_conditions, t_span, T, rates):
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
    num_eqns = len(equations)
    y_values = np.zeros((num_eqns, 1))

    for i, initial_value in enumerate(initial_conditions):
        y_values[i, 0] = initial_value

    def update(equation_rates, y_values, i, dt):
        predictors = []
        initial_destruction_rates = []
        initial_positive_fluxes = []
        for j in range(len(equation_rates)):
            er = equation_rates[j]

            y0 = y_values[j, i]
            k0 = sum(er.destruction_rates) * er.destruction_sign
            F_p0 = sum(er.positive_fluxes)
            F_m0 = k0 * y0

            yp = predictor(y0, k0, F_p0, F_m0, dt)

            predictors.append(yp)
            initial_destruction_rates.append(k0)
            initial_positive_fluxes.append(F_p0)

        for j, eq in enumerate(equations):
            er = eq.get_rates(*predictors, T)
            y0 = y_values[j, i]
            k0 = initial_destruction_rates[j]
            kp = sum(er.destruction_rates) * er.destruction_sign
            F_p0 = initial_positive_fluxes[j]
            F_pp = sum(er.positive_fluxes)

            rate = corrector(y0, k0, kp, F_p0, F_pp, dt)

            y_values[j, i + 1] = y_values[j, i] + dt * rate

    trial_timestep_tol = 0.1
    conservation_tol = 0.001
    conservation_satisfied_tol = 0.001
    decrease_dt_factor = 0.1
    increase_dt_factor = 0.1

    # t, y_values = simple_timestepper(
    #     y_values,
    #     equations,
    #     update,
    #     dt,
    #     t0,
    #     tf,
    #     trial_timestep_tol,
    #     conservation_tol,
    #     conservation_satisfied_tol,
    #     decrease_dt_factor,
    #     increase_dt_factor,
    #     T,
    # )

    t, y_values = constant_timestepper(
        y_values,
        equations,
        update,
        dt,
        t0,
        tf,
        T,
    )
    print(f"number of time steps: {len(t)}")

    return t, y_values, None
