import numpy as np
import math
from collections import namedtuple

Rates = namedtuple(
    "Rates", ["positive_fluxes", "destruction_rates", "destruction_sign"], defaults=[1]
)

USE_FLFD = False


def nn(x):
    if USE_FLFD:
        return max(x, 0)
    return x


# predictor from equation 16 in asymptotic approximations paper
def predictor(y0, k0, F_p0, dt):
    tao0 = 1 / k0

    yp = (y0 * (2 * tao0 - dt) + 2 * F_p0 * tao0 * dt) / (2 * tao0 * dt)
    return yp


# corrector from equation 16 in asymptotic approximations paper
def corrector(y0, k0, kp, F_p0, F_pp, dt):
    tao0 = 1 / k0
    taop = 1 / kp

    yc = (y0 * (taop + tao0 - dt) + 0.5 * dt * (F_pp + F_p0) * (taop + tao0)) / (
        taop + tao0 + dt
    )
    return yc


class HI:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [self.rates.k2(T) * HII * e]
        destruction_rates = [
            self.rates.k1(T) * e,
            self.rates.k57(T) * HI,
            self.rates.k58(T) * HeI / 4,
        ]

        return Rates(positive_fluxes, destruction_rates)


class HII:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [
            self.rates.k1(T) * HI * e,
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
        ]
        destruction_rates = [self.rates.k2(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class e:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
        ]
        destruction_rates = [
            self.rates.k1(T) * HI,
            -self.rates.k2(T) * HII,
            +self.rates.k3(T) * HeI / 4,
            -self.rates.k6(T) * HeIII / 4,
            +self.rates.k5(T) * HeII / 4,
            -self.rates.k4(T) * HeII / 4,
        ]

        destruction_sign = -1

        return Rates(positive_fluxes, destruction_rates, destruction_sign)


class HeI:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [self.rates.k4(T) * HeII * e]
        destruction_rates = [self.rates.k3(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class HeII:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [self.rates.k3(T) * HeI * e, self.rates.k6(T) * HeIII * e]
        destruction_rates = [self.rates.k4(T) * e, self.rates.k5(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class HeIII:
    def __init__(self, rates):
        self.rates = rates

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [self.rates.k5(T) * HeII * e]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)


odes = [HI(None), HII(None), HeI(None), HeII(None), HeIII(None), e(None)]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def asymptotic_predictor_corrector_solver(
    equations, initial_conditions, t_span, T, rates
):
    for eq in equations:
        eq.rates = rates
    # dt = (
    #     abs(initial_conditions[0] / odes[0].next(*initial_conditions, T, 0, False))
    #     * 0.0001
    # )
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
        predictors = []
        initial_destruction_rates = []
        initial_positive_fluxes = []
        for j, eq in enumerate(equations):
            er = eq.get_rates(
                *y_values[:, i],
                T,
            )
            y0 = y_values[j, i]
            k0 = sum(er.destruction_rates) * er.destruction_sign
            F_p0 = sum(er.positive_fluxes)

            yp = predictor(y0, k0, F_p0, dt)

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

            if k0 * dt >= 1:
                print("tm1")
                rate = F_p0 - k0 * y0

            y_values[j, i + 1] = y_values[j, i] + dt * rate
            rate_values[j, i + 1] = rate
            # y_values[j, i + 1] = yc
    print("solver final state: ", y_values[:, n - 1])
    return t, y_values, rate_values
