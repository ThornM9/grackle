import numpy as np
import math

USE_FLFD = False


def nn(x):
    if USE_FLFD:
        return max(x, 0)
    return x


# equation 12 in the asymptotic paper
def prediction_1(F_p, k_n, k_n_prev, F_p_prev, y_prev, dt):
    first_term = F_p / k_n

    # if k_n == 0 or k_n_prev == 0 or dt == 0:
    #     print(k_n, k_n_prev, dt)
    #     print("FOUND")

    if k_n == 0:
        k_n = 1e-20
    if k_n_prev == 0:
        k_n_prev = 1e-20
    second_term = (1 / (k_n * dt)) * ((F_p / k_n) - (F_p_prev / k_n_prev))

    if not first_term or not second_term:
        print(first_term, second_term)

    y_n = first_term - second_term
    return y_n
    # return first_term - second_term


# equation 13 in the asymptotic paper
def prediction_2(F_p, k_n, dt, y_prev):
    yn = (1 / (1 + k_n * dt)) * (y_prev + (F_p * dt))
    return yn


# equation 16 in the asymptotic paper (predictor corrector scheme)
def prediction_3(F_p, k_n, dt, values, i, j):
    pass


# equation 4 in qss paper
def alpha(r):
    return ((160 * r**3) + (60 * r**2) + (11 * r) + 1) / (
        (360 * r**3) + (60 * r**2) + (12 * r) + 1
    )


class HI:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):
        positive_fluxes = [self.rates.k2(T) * HII * e]
        destruction_rates = [
            self.rates.k1(T) * e,
            self.rates.k57(T) * HI,
            self.rates.k58(T) * HeI / 4,
        ]

        # print("tm1", destruction_rates, sum(destruction_rates))

        destruction = sum(list(map(lambda x: nn(x * HI), destruction_rates)))
        creation = sum(list(map(lambda x: nn(x), positive_fluxes)))

        F_p = sum(positive_fluxes)
        k_n = sum(destruction_rates)

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = HI + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)
        return diff


class HII:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):
        positive_fluxes = [
            self.rates.k1(T) * HI * e,
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
        ]
        destruction_rates = [self.rates.k2(T) * e]
        destruction = sum(list(map(lambda x: nn(x * HII), destruction_rates)))
        creation = sum(list(map(lambda x: nn(x), positive_fluxes)))

        F_p = sum(positive_fluxes)
        k_n = sum(destruction_rates)

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = HII + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)

        return diff


class e:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):

        creation = nn(self.rates.k57(T) * HI * HI) + nn(
            self.rates.k58(T) * HI * HeI / 4
        )

        destruction = -(
            nn(self.rates.k1(T) * HI * e)
            - nn(self.rates.k2(T) * HII * e)
            + nn((self.rates.k3(T) * HeI / 4) * e)
            - nn((self.rates.k6(T) * HeIII / 4) * e)
            + nn((self.rates.k5(T) * HeII / 4) * e)
            - nn((self.rates.k4(T) * HeII / 4) * e)
        )

        F_p = self.rates.k57(T) * HI * HI + self.rates.k58(T) * HI * HeI / 4
        k_n = -(
            self.rates.k1(T) * HI
            - self.rates.k2(T) * HII
            + self.rates.k3(T) * HeI / 4
            - self.rates.k6(T) * HeIII / 4
            + self.rates.k5(T) * HeII / 4
            - self.rates.k4(T) * HeII / 4
        )

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = e + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)
        return diff


class HeI:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):
        creation = nn(self.rates.k4(T) * HeII * e)
        destruction = nn(self.rates.k3(T) * e * HeI)

        F_p = self.rates.k4(T) * HeII * e
        k_n = self.rates.k3(T) * e

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = HeI + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)

        return diff


class HeII:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):

        positive_fluxes = [self.rates.k3(T) * HeI * e, self.rates.k6(T) * HeIII * e]
        destruction_rates = [self.rates.k4(T) * e, self.rates.k5(T) * e]

        destruction = sum(list(map(lambda x: nn(x * HeII), destruction_rates)))
        creation = sum(list(map(lambda x: nn(x), positive_fluxes)))

        F_p = sum(positive_fluxes)
        k_n = sum(destruction_rates)

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = HeII + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)

        return diff


class HeIII:
    F_p = []
    k_n = []
    y_n = []
    rates = None

    def next(self, HI, HII, HeI, HeII, HeIII, e, T, dt, save_results=True):

        positive_fluxes = [self.rates.k5(T) * HeII * e]
        destruction_rates = [self.rates.k6(T) * e]

        destruction = sum(list(map(lambda x: nn(x * HeIII), destruction_rates)))
        creation = sum(list(map(lambda x: nn(x), positive_fluxes)))

        F_p = sum(positive_fluxes)
        k_n = sum(destruction_rates)

        if len(self.F_p) <= 1 or abs(k_n * dt) >= 1:
            diff = HeIII + (creation - destruction) * dt
        else:
            k_n_prev = self.k_n[len(self.k_n) - 2]
            F_p_prev = self.F_p[len(self.F_p) - 2]
            y_prev = self.y_n[len(self.y_n) - 2]
            # diff = prediction_1(
            #     F_p,
            #     k_n,
            #     k_n_prev,
            #     F_p_prev,
            #     y_prev,
            #     dt,
            # )

            diff = prediction_2(F_p, k_n, dt, y_prev)

        if save_results:
            self.F_p.append(F_p)
            self.k_n.append(k_n)
            self.y_n.append(diff)

        return diff


odes = [HI(), HII(), HeI(), HeII(), HeIII(), e()]
species_names = ["HI", "HII", "HeI", "HeII", "HeIII"]


def asymptotic_methods_solver(equations, initial_conditions, t_span, T, rates):
    for eq in equations:
        eq.rates = rates
    dt = (
        abs(initial_conditions[0] / odes[0].next(*initial_conditions, T, 0, False))
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
        print("timestep", i + 1)
        for j, eq in enumerate(equations):
            rate = eq.next(
                *y_values[:, i],
                T,
                dt,
            )
            # y_values[j, i + 1] = y_values[j, i] + dt * rate
            y_values[j, i + 1] = rate
            rate_values[j, i + 1] = rate
    print("solver final state: ", y_values[:, n - 1])
    print(y_values)
    return t, y_values, rate_values
