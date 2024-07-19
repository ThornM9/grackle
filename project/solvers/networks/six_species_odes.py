import numpy as np
from collections import namedtuple
from ..utils import get_cloudy_rates

Rates = namedtuple(
    "Rates", ["positive_fluxes", "destruction_rates", "destruction_sign"], defaults=[1]
)

ReactionGroups = namedtuple("ReactionGroups", ["positive", "destruction"])

# to add a positive flux, multiply the rate by all inputs
# to add a negative flux, multiply the rate by all inputs except itself


### REACTION GROUPS
# Reaction: (HII + e <--> HI + photon) (0)
# Forward rate: k2
# Reverse rate: piHI
#
# Reaction: (HeII + e <--> HeI + photon) (1)
# Forward rate: k4
# Reverse rate: piHeI
#
# Reaction: (HeIII + e <--> HeII + photon) (2)
# Forward rate: k6
# Reverse rate: piHeII


class HI_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups([0], [-1, -1, -1, 0])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances
        cloudy = get_cloudy_rates()
        positive_fluxes = [self.rates.k2(T) * HII * e]
        destruction_rates = [
            self.rates.k1(T) * e,
            self.rates.k57(T) * HI,
            self.rates.k58(T) * HeI / 4,
            cloudy["piHI"],
        ]

        return Rates(positive_fluxes, destruction_rates)


class HII_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups([-1, -1, -1, 0], [0])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances
        cloudy = get_cloudy_rates()
        positive_fluxes = [
            self.rates.k1(T) * HI * e,
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
            cloudy["piHI"] * HI,
        ]
        destruction_rates = [self.rates.k2(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class e_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = True

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, -1, 0, 1, 2],
            [-1, 0, -1, 1, -1, 2],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances

        cloudy = get_cloudy_rates()
        positive_fluxes = [
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
            cloudy["piHI"] * HI,
            cloudy["piHeI"] * HeI / 4,
            cloudy["piHeII"] * HeII / 4,
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


class HeI_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups(
            [1],
            [-1, 1],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances

        cloudy = get_cloudy_rates()
        positive_fluxes = [self.rates.k4(T) * HeII * e]
        destruction_rates = [self.rates.k3(T) * e, cloudy["piHeI"]]

        return Rates(positive_fluxes, destruction_rates)


class HeII_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, 2, 1],
            [1, -1, 2],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances

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


class HeIII_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, 2],
            [2],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k5(T) * HeII * e, cloudy["piHeII"] * HeII]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class Energy_ODE:

    def __init__(self, rates):
        self.rates = rates
        self.is_energy = True

    def get_reaction_groups(self):
        # return ReactionGroups([], [])
        return ReactionGroups(
            [-1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, E = abundances
        cloudy = get_cloudy_rates()

        #     edot(i) = edot(i) + real(ipiht, DKIND) * photogamma(i,j,k)
        #  &                          / coolunit * HI(i,j,k) / dom
        positive_fluxes = [
            cloudy["piHI"] * HI * self.rates.chemistry_data.cooling_units,
        ]
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
            # comp
            self.rates.comp() * e,
        ]

        return Rates(positive_fluxes, destruction_rates)


def calculate_gamma(abundances, rates):
    return 5 / 3


def calculate_energy_from_temp(abundances, rates, T):
    HI, HII, HeI, HeII, HeIII, e, E = abundances

    mu = calculate_mu(rates, abundances)

    gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    mh = 1.67262171e-24  # mass of hydrogen
    k = 1.3806504e-16  # boltzmann constant
    # E = k * T / ((gamma - 1) * mu * mh)

    E = T / (mh / k) / mu / (gamma - 1)
    return E


def calculate_temp_from_energy(abundances, rates, gamma):
    HI, HII, HeI, HeII, HeIII, e, E = abundances

    # gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    density = HI + HII + HeI + HeII + HeIII
    number_density = (
        (HI + HII) + (HeI + HeII + HeIII) / 4 + e
    )  # todo maybe this should include mh
    pressure = (gamma - 1) * density * E
    temperature = (
        pressure * rates.chemistry_data.temperature_units / max(number_density, 1e-20)
    )
    return temperature


def calculate_mu(rates, abundances):
    HI, HII, HeI, HeII, HeIII, e, E = abundances

    if E is None:
        nden = HI + HII + e + (HeI + HeII + HeIII) / 4
        density = HI + HII + HeI + HeII + HeIII
        return density / nden

    gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    temperature = calculate_temp_from_energy(abundances, rates, gamma)

    mu = temperature / (E * (gamma - 1) * rates.chemistry_data.temperature_units)

    return mu


# data structure to store all config info about the reaction groups in this network


def get_kf(rg_num, rates, T):
    if rg_num == 0:
        return rates.k2(T)
    if rg_num == 1:
        return rates.k4(T)
    if rg_num == 2:
        return rates.k6(T)
    raise Exception("Invalid reaction group number: ", rg_num)


def get_kr(rg_num, rates, T):
    cloudy = get_cloudy_rates()
    if rg_num == 0:
        return cloudy["piHI"]
    if rg_num == 1:
        return cloudy["piHeI"]
    if rg_num == 2:
        return cloudy["piHeII"]
    raise Exception("Invalid reaction group number: ", rg_num)


# maps a reaction group to a list of abundance indices for the species involved in the network
reaction_group_config = {
    0: [1, 5, 0],
    1: [3, 5, 2],
    2: [4, 5, 3],
    "rg_count": 3,
    "get_kf": get_kf,
    "get_kr": get_kr,
}

odes = [
    HI_ODE(None),
    HII_ODE(None),
    HeI_ODE(None),
    HeII_ODE(None),
    HeIII_ODE(None),
    e_ODE(None),
    Energy_ODE(None),
]

indexes = {
    "HI": 0,
    "HII": 1,
    "HeI": 2,
    "HeII": 3,
    "HeIII": 4,
    "Electron": 5,
    "Energy": 6,
}

species_names = [
    "HI",
    "HII",
    "HeI",
    "HeII",
    "HeIII",
    "Electron",
]
