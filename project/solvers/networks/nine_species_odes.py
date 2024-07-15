import numpy as np
from collections import namedtuple
from ..utils import get_cloudy_rates
import math

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
#
# Reaction: (HI + e <--> HM + photon) (3)
# Forward rate: k7
# Reverse rate: k27
#
# Reaction: (HI + HII <--> H2II + photon) (4)
# Forward rate: k9
# Reverse rate: TBD
#
# Reaction: (H2II + HI <--> H2I + HII) (5)
# Forward rate: k10
# Reverse rate: k11
#
# Reaction: (H2I + HI <--> HI + HI + HI) (6)
# Forward rate: k13
# Reverse rate: k22


class HI_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups(
            [0, 6, 5, -1, -1, -1, -1, -1, -1], [-1, 3, -1, 4, 5, 6, -1, -1, -1]
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()
        positive_fluxes = [
            self.rates.k2(T) * HII * e,
            2 * self.rates.k13(T) * HI * H2I / 2,
            self.rates.k11(T) * HII * H2I / 2,
            2 * self.rates.k12(T) * e * H2I / 2,
            self.rates.k14(T) * HM * e,
            self.rates.k15(T) * HM * HI,
            2 * self.rates.k16(T) * HM * HII,
            2 * self.rates.k18(T) * H2II * e / 2,
            self.rates.k19(T) * H2II * HM / 2,
        ]
        destruction_rates = [
            self.rates.k1(T) * e,
            self.rates.k7(T) * e,
            self.rates.k8(T) * HM,
            self.rates.k9(T) * HII,
            self.rates.k10(T) * H2II / 2,
            2 * self.rates.k22(T) * (HI**2),
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
        return ReactionGroups(
            [
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [0, 4, -1, -1, -1],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()
        positive_fluxes = [
            self.rates.k1(T) * HI * e,
            self.rates.k10(T) * H2II * HI / 2,
            self.rates.k57(T) * HI * HI,
            self.rates.k58(T) * HI * HeI / 4,
            cloudy["piHI"] * HI,
        ]
        destruction_rates = [
            self.rates.k2(T) * e,
            self.rates.k9(T) * HI,
            self.rates.k11(T) * H2I / 2,
            self.rates.k16(T) * HM,
            self.rates.k17(T) * HM,
        ]

        return Rates(positive_fluxes, destruction_rates)


class HeI_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups(
            [1],
            [-1, -1],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

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
            [-1, 2, -1],
            [1, -1, -1],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

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
            [-1, -1],
            [2],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()

        positive_fluxes = [self.rates.k5(T) * HeII * e, cloudy["piHeII"] * HeII]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)


class e_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = True

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, -1, 1, -1, 2, -1, 3, -1],
        )

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

        cloudy = get_cloudy_rates()
        positive_fluxes = [
            self.rates.k8(T) * HM * HI,
            self.rates.k15(T) * HM * HI,
            self.rates.k17(T) * HM * HII,
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
            +self.rates.k14(T) * HM,
            -self.rates.k7(T) * HI,
            self.rates.k18(T) * H2II / 2,
        ]

        destruction_sign = -1

        return Rates(positive_fluxes, destruction_rates, destruction_sign)


class H2I_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups([-1, 5, -1, 6], [6, 5, -1])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()
        positive_fluxes = [
            2 * self.rates.k8(T) * HM * HI,
            2 * self.rates.k10(T) * H2II * HI / 2,
            2 * self.rates.k19(T) * H2II * HM / 2,
            2 * self.rates.k22(T) * HI * (HI**2),
        ]
        destruction_rates = [
            self.rates.k13(T) * HI,
            self.rates.k11(T) * HII,
            self.rates.k12(T) * e,
        ]

        return Rates(positive_fluxes, destruction_rates)


class HM_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups([-1], [-1, -1, -1, -1, -1, -1, 3])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()

        positive_fluxes = [
            self.rates.k7(T) * HI * e,
        ]
        destruction_rates = [
            self.rates.k8(T) * HI,
            self.rates.k15(T) * HI,
            self.rates.k16(T) * HII,
            self.rates.k17(T) * HII,
            self.rates.k14(T) * e,
            self.rates.k19(T) * H2II / 2,
            cloudy["k27"],
        ]

        return Rates(positive_fluxes, destruction_rates)


class H2II_ODE:
    def __init__(self, rates):
        self.rates = rates
        self.is_energy = False
        self.is_electron = False

    def get_reaction_groups(self):
        return ReactionGroups([4, 5, -1], [5, -1, -1])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        positive_fluxes = [
            2 * self.rates.k9(T) * HI * HII,
            2 * self.rates.k11(T) * H2I / 2 * HII,
            2 * self.rates.k17(T) * HM * HII,
        ]
        destruction_rates = [
            self.rates.k10(T) * HI,
            self.rates.k18(T) * e,
            self.rates.k19(T) * HM,
        ]

        return Rates(positive_fluxes, destruction_rates)


class Energy_ODE:

    def __init__(self, rates):
        self.rates = rates
        self.is_energy = True

    def get_reaction_groups(self):
        return ReactionGroups([], [])

    def get_rates(self, abundances, T):
        HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances
        cloudy = get_cloudy_rates()

        #     edot(i) = edot(i) + real(ipiht, DKIND) * photogamma(i,j,k)
        #  &                          / coolunit * HI(i,j,k) / dom
        positive_fluxes = [
            # cloudy["piHI"] * HI * self.rates.chemistry_data.cooling_units,
        ]
        destruction_rates = [
            # # collisional excitations
            # self.rates.ceHI(T) * HI * e,
            # self.rates.ceHeI(T) * HeII * (e**2) / 4,
            # self.rates.ceHeII(T) * HeII * e / 4,
            # # collisional ionizations
            # self.rates.ciHI(T) * HI * e,
            # self.rates.ciHeI(T) * HeI * e / 4,
            # self.rates.ciHeII(T) * HeII * e / 4,
            # self.rates.ciHeIS(T) * HeII * (e**2) / 4,
            # # recombinations
            # self.rates.reHII(T) * HII * e,
            # self.rates.reHeII1(T) * HeII * e / 4,
            # self.rates.reHeII2(T) * HeII * e / 4,
            # self.rates.reHeIII(T) * HeIII * e / 4,
            # # brem
            # self.rates.brem(T) * (HII * HeII / 4 + HeIII) * e,
            # # comp
            # self.rates.comp() * e,
        ]

        return Rates(positive_fluxes, destruction_rates)


def calculate_gamma(abundances, rates):
    HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

    gamma = 5 / 3
    gamma_inverse = 1 / (gamma - 1)

    T = calculate_temp_from_energy(abundances, rates, gamma)

    number_density = (HI + HII) + (HeI + HeII + HeIII) / 4 + e

    nH2 = 0.5 * (H2I + H2II)

    # only do full computation if there is a reasonable amount of H2
    gamma_H2_inverse = 0.5 * 5
    if nH2 / number_density > 1e-3:
        x = 6100 / T
        if x < 10:
            gamma_H2_inverse = 0.5 * (
                5 + 2 * x * x * math.exp(x) / ((math.exp(x) - 1) ** 2)
            )

    gamma = 1 + (nH2 + number_density) / (
        nH2 * gamma_H2_inverse + number_density * gamma_inverse
    )
    return gamma


def calculate_energy_from_temp(abundances, rates, T):
    HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

    mu = calculate_mu(rates, abundances)

    gamma = 5 / 3  # adiabatic index of ideal gas for 6 species network
    mh = 1.67262171e-24  # mass of hydrogen
    k = 1.3806504e-16  # boltzmann constant
    # E = k * T / ((gamma - 1) * mu * mh)

    E = T / (mh / k) / mu / (gamma - 1)
    return E


def calculate_temp_from_energy(abundances, rates, gamma):
    HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

    density = HI + HII + HeI + HeII + HeIII
    number_density = (HI + HII) + (HeI + HeII + HeIII) / 4 + e

    number_density += HM + 0.5 * (H2I + H2II)
    pressure = (gamma - 1) * density * E
    temperature = (
        pressure * rates.chemistry_data.temperature_units / max(number_density, 1e-20)
    )
    return temperature


def calculate_mu(rates, abundances):
    HI, HII, HeI, HeII, HeIII, e, H2I, HM, H2II, E = abundances

    if E is None:
        nden = HI + HII + e + (HeI + HeII + HeIII) / 4
        density = HI + HII + HeI + HeII + HeIII
        nden += HM + 0.5 * (H2I + H2II)
        return density / nden

    gamma = calculate_gamma(
        abundances, rates
    )  # TODO maybe this should be calculated after temperature and temperature should use a previous value. check fluid_container.py for order
    temperature = calculate_temp_from_energy(abundances, rates, gamma)

    mu = temperature / (E * (gamma - 1) * rates.chemistry_data.temperature_units)

    return mu


odes = [
    HI_ODE(None),
    HII_ODE(None),
    HeI_ODE(None),
    HeII_ODE(None),
    HeIII_ODE(None),
    e_ODE(None),
    H2I_ODE(None),
    HM_ODE(None),
    H2II_ODE(None),
    Energy_ODE(None),
]

indexes = {
    "HI": 0,
    "HII": 1,
    "HeI": 2,
    "HeII": 3,
    "HeIII": 4,
    "e": 5,
    "H2I": 6,
    "HM": 7,
    "H2II": 8,
    "Energy": 9,
}

species_names = ["HI", "HII", "HeI", "HeII", "HeIII", "Electron", "H2I", "HM", "H2II"]


def get_kf(rg_num, rates, T):
    if rg_num == 0:
        return rates.k2(T)
    if rg_num == 1:
        return rates.k4(T)
    if rg_num == 2:
        return rates.k6(T)
    if rg_num == 3:
        return rates.k7(T)
    if rg_num == 4:
        return rates.k9(T)
    if rg_num == 5:
        return rates.k10(T)
    if rg_num == 6:
        return rates.k13(T)
    raise Exception("Invalid reaction group number")


def get_kr(rg_num, rates, T):
    cloudy = get_cloudy_rates()
    if rg_num == 0:
        return cloudy["piHI"]
    if rg_num == 1:
        return cloudy["piHeI"]
    if rg_num == 2:
        return cloudy["piHeII"]
    if rg_num == 3:
        return cloudy["k27"]
    if rg_num == 4:
        raise Exception("reaction group 4 kr is still TBD")
    if rg_num == 5:
        return rates.k11(T)
    if rg_num == 6:
        return rates.k22(T)

    raise Exception("Invalid reaction group number")


# maps a reaction group to a list of abundance indices for the species involved in the network
reaction_group_config = {
    0: [indexes["HII"], indexes["e"], indexes["HI"]],
    1: [indexes["HeII"], indexes["e"], indexes["HeI"]],
    2: [indexes["HeIII"], indexes["e"], indexes["HeII"]],
    3: [indexes["HI"], indexes["e"], indexes["HM"]],
    4: [indexes["HI"], indexes["HII"], indexes["H2II"]],
    5: [indexes["H2II"], indexes["HI"], indexes["H2I"], indexes["HII"]],
    6: [indexes["H2I"], indexes["HI"], indexes["HI"], indexes["HI"], indexes["HI"]],
    "get_kf": get_kf,
    "get_kr": get_kr,
}
