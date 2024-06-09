import numpy as np
from collections import namedtuple

Rates = namedtuple(
    "Rates", ["positive_fluxes", "destruction_rates", "destruction_sign"], defaults=[1]
)

ReactionGroups = namedtuple("ReactionGroups", ["positive", "destruction"])


class HI:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups([0], [-1, -1, -1])

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

    def get_reaction_groups(self):
        return ReactionGroups([-1, -1, -1], [0])

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

    def get_reaction_groups(self):
        return ReactionGroups(
            [-1, -1],
            [-1, 0, -1, 1, -1, 2],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
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


class HeI:
    def __init__(self, rates):
        self.rates = rates

    def get_reaction_groups(self):
        return ReactionGroups(
            [1],
            [-1],
        )

    def get_rates(self, HI, HII, HeI, HeII, HeIII, e, T):
        positive_fluxes = [self.rates.k4(T) * HeII * e]
        destruction_rates = [self.rates.k3(T) * e]

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
        positive_fluxes = [self.rates.k3(T) * HeI * e, self.rates.k6(T) * HeIII * e]
        destruction_rates = [self.rates.k4(T) * e, self.rates.k5(T) * e]

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
        positive_fluxes = [self.rates.k5(T) * HeII * e]
        destruction_rates = [self.rates.k6(T) * e]

        return Rates(positive_fluxes, destruction_rates)
