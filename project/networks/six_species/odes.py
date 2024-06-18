import numpy as np
from collections import namedtuple
import h5py
import os
from pygrackle.utilities.physical_constants import (
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)


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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
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

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k5(T) * HeII * e, cloudy["piHeII"] * HeII]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class Energy:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups([], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, E, T):
        print("tm1", self.rates.comp())
        positive_fluxes = []
        destruction_rates = [
            # collisional excitations
            self.rates.ceHI(T) * HI * e,
            self.rates.ceHeI(T) * HeII * (e**2) / 4,
            self.rates.ceHeII(T) * HeII * e / 4,
            # collisional ionizations
            self.rates.ciHI(T) * HI * e,
            self.rates.ciHeI(T) * HeI * e / 4,
            self.rates.ciHeII(T) * HeII * e / 4,
            self.rates.ciHeIS(T) * HeII * (e**2) / 4,
            # recombinations
            self.rates.reHII(T) * HII * e,
            self.rates.reHeII1(T) * HeII * e / 4,
            self.rates.reHeII2(T) * HeII * e / 4,
            self.rates.reHeIII(T) * HeIII * e / 4,
            # brem
            self.rates.brem(T) * (HII * HeII / 4 + HeIII) * e,
        ]

        return Rates(positive_fluxes, destruction_rates)


def calculate_temp_from_energy(HI, HII, HeI, HeII, HeIII, e, E, prevT):
    mu = (HeI + HeII + HeIII) / 4 + HI + HII + e

    gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    mh = 1.67262171e-24  # mass of hydrogen atom (grams)
    k = 1.3806504e-16  # boltzmann constant
    # T = (gamma - 1) * mu * E * mh / k

    velocity_units = cm_per_mpc / sec_per_Myr
    temperature_units = mh * velocity_units**2 / k
    return temperature_units / max(mu, 1e-20)
    # return 0.5 * (T + prevT)


def calculate_energy_from_temp(HI, HII, HeI, HeII, HeIII, e, T):
    mu = (HeI + HeII + HeIII) / 4 + HI + HII + e

    gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    mh = 1.67262171e-24  # mass of hydrogen
    k = 1.3806504e-16  # boltzmann constant
    E = k * T / ((gamma - 1) * mu * mh)

    # velocity_units = cm_per_mpc / sec_per_Myr
    # temperature_units = mh * velocity_units**2 / k
    # return temperature_units / max(mu, 1e-20)
    # return E


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
