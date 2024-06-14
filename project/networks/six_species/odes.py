import numpy as np
from collections import namedtuple
import h5py
import os


Rates = namedtuple(
    "Rates", ["positive_fluxes", "destruction_rates", "destruction_sign"], defaults=[1]
)

ReactionGroups = namedtuple("ReactionGroups", ["positive", "destruction"])

# to add a positive flux, multiply the rate by all inputs
# to add a negative flux, multiply the rate by all inputs except itself


class HI:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups([0], [-1, -1, -1])

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k2(T) * HII * e]
        destruction_rates = [
            self.rates.k1(T) * e,
            self.rates.k57(T) * HI,
            self.rates.k58(T) * HeI / 4,
            cloudy["piHI"],
        ]

        return Rates(positive_fluxes, destruction_rates)


class HII:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups([-1, -1, -1], [0])

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [
            self.rates.k1(T) * HI * e,
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
            cloudy["piHI"] * HI,
        ]
        destruction_rates = [self.rates.k2(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class e:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, -1],
            [-1, 0, -1, 1, -1, 2],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()
        # TODO: add the photoionization rates
        positive_fluxes = [
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
        ]
        destruction_rates = [
            self.rates.k1(T) * HI,
            -self.rates.k2(T) * HII,
            +self.rates.k3(T) * HeI / 4,
            -self.rates.k4(T) * HeII / 4,
            +self.rates.k5(T) * HeII / 4,
            -self.rates.k6(T) * HeIII / 4,
        ]

        destruction_sign = -1

        return Rates(positive_fluxes, destruction_rates, destruction_sign)


# to add a positive flux, multiply the rate by all inputs
# to add a negative flux, multiply the rate by all inputs except itself
class HeI:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups(
            [1],
            [-1],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k4(T) * HeII * e]
        destruction_rates = [self.rates.k3(T) * e, cloudy["piHeI"]]

        return Rates(positive_fluxes, destruction_rates)


class HeII:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, 2],
            [1, -1],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [
            self.rates.k3(T) * HeI * e,
            self.rates.k6(T) * HeIII * e,
            cloudy["piHeI"] * HeI,
        ]
        destruction_rates = [
            self.rates.k4(T) * e,
            self.rates.k5(T) * e,
            cloudy["piHeII"],
        ]

        return Rates(positive_fluxes, destruction_rates)


class HeIII:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1],
            [2],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k5(T) * HeII * e, cloudy["piHeII"] * HeII]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)


def get_cloudy_rates():
    my_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(
        my_dir, "..", "..", "..", "input", "CloudyData_UVB=HM2012.h5"
    )

    with h5py.File(file_name, "r") as file:
        return {
            "piHI": file["UVBRates"]["Photoheating"]["piHI"][0],
            "piHeI": file["UVBRates"]["Photoheating"]["piHeI"][0],
            "piHeII": file["UVBRates"]["Photoheating"]["piHeII"][0],
        }
