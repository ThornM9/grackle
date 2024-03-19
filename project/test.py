import numpy as np
from pygrackle.utilities.primordial_equilibrium import (
    nHI,
    nHII,
    nHeI,
    nHeII,
    nHeIII,
)

from pygrackle.utilities.testing import assert_rel_equal
import matplotlib.pyplot as plt


def test_equilibrium(solver_config, initial_conditions, rates, T, t_span):
    solver, odes, names = solver_config

    print("custom solver")
    # TODO this is solving for the second time if called from densities, add an option to pass in already completed solver results
    t, values, rate_values = solver(odes, initial_conditions, t_span, T, rates)
    print("finished custom solver")
    final_state = values[:, -1]

    final_HI = final_state[0]
    final_HII = final_state[1]
    final_HeI = final_state[2]
    final_HeII = final_state[3]
    final_HeIII = final_state[4]

    my_nH = initial_conditions[0] + initial_conditions[1]

    # calculate the balance figures
    nH_eq = nHI(np.array([T]), my_nH) + nHII(np.array([T]), my_nH)
    nH_g = final_HI + final_HII
    fHI_eq = nHI(np.array([T]), my_nH) / nH_eq
    fHI_g = final_HI / nH_g
    fHII_eq = nHII(np.array([T]), my_nH) / nH_eq
    fHII_g = final_HII / nH_g

    nHe_eq = (
        nHeI(np.array([T]), my_nH)
        + nHeII(np.array([T]), my_nH)
        + nHeIII(np.array([T]), my_nH)
    )
    nHe_g = final_HeI + final_HeII + final_HeIII
    fHeI_eq = nHeI(np.array([T]), my_nH) / nHe_eq
    fHeI_g = final_HeI / nHe_g
    fHeII_eq = nHeII(np.array([T]), my_nH) / nHe_eq
    fHeII_g = final_HeII / nHe_g
    fHeIII_eq = nHeIII(np.array([T]), my_nH) / nHe_eq
    fHeIII_g = final_HeIII / nHe_g

    test_precision = 2

    # plot the results
    # for i, val in enumerate(values):
    #     if i < 5:
    #         plt.plot(t, val, label=f"{names[i]}")
    plt.plot([t_span[0], t_span[1]], [fHI_g, fHI_g], label="HI")
    plt.plot([t_span[0], t_span[1]], [fHII_g, fHII_g], label="HII")
    plt.plot([t_span[0], t_span[1]], [fHeI_g, fHeI_g], label="HeI")
    plt.plot([t_span[0], t_span[1]], [fHeII_g, fHeII_g], label="HeII")
    plt.plot([t_span[0], t_span[1]], [fHeIII_g, fHeIII_g], label="HeIII")

    plt.xlabel("Time (Myr)")
    plt.ylabel("Density (g/cm^3)")
    plt.legend()
    plt.savefig("test_results.png")
    plt.clf()

    plt.plot([t_span[0], t_span[1]], [fHI_eq, fHI_eq], label="HI")
    plt.plot([t_span[0], t_span[1]], [fHII_eq, fHII_eq], label="HII")
    plt.plot([t_span[0], t_span[1]], [fHeI_eq, fHeI_eq], label="HeI")
    plt.plot([t_span[0], t_span[1]], [fHeII_eq, fHeII_eq], label="HeII")
    plt.plot([t_span[0], t_span[1]], [fHeIII_eq, fHeIII_eq], label="HeIII")
    plt.xlabel("Time (Myr)")
    plt.ylabel("Density (g/cm^3)")
    plt.legend()
    plt.savefig("test_expected.png")
    plt.clf()

    # test the ionization balance
    print("HI: ", fHI_eq, fHI_g)
    print("HII: ", fHII_eq, fHII_g)
    print("HeI: ", fHeI_eq, fHeI_g)
    print("HeII: ", fHeII_eq, fHeII_g)
    print("HeIII: ", fHeIII_eq, fHeIII_g)
    assert_rel_equal(
        fHI_eq,
        np.array([fHI_g]),
        test_precision,
        f"HI fractions disagree. \nExpected: {fHI_eq}. \nActual: {fHI_g}.",
    )

    assert_rel_equal(
        fHII_eq,
        np.array([fHII_g]),
        test_precision,
        f"HII fractions disagree. \nExpected: {fHII_eq}. \nActual: {fHII_g}.",
    )

    assert_rel_equal(
        fHeI_eq,
        np.array([fHeI_g]),
        test_precision,
        f"HeI fractions disagree. \nExpected: {fHeI_eq}. \nActual: {fHeI_g}.",
    )

    assert_rel_equal(
        fHeII_eq,
        np.array([fHeII_g]),
        test_precision,
        f"HeII fractions disagree. \nExpected: {fHeII_eq}. \nActual: {fHeII_g}.",
    )

    assert_rel_equal(
        fHeIII_eq,
        np.array([fHeIII_g]),
        test_precision,
        f"HeIII fractions disagree. \nExpected: {fHeIII_eq}. \nActual: {fHeIII_g}.",
    )

    print(f"Test succeeded for temperature: {T}K")


if __name__ == "__main__":
    from densities import solver_configs, get_rates

    # test_equilibrium(1.0e6)

    rates = get_rates()
    density = 0.1  # g /cm^3
    error_threshold = 0.01
    tiny_number = 1e-20
    initial_conditions = [
        0.76 * density,
        tiny_number * density,
        (1.0 - 0.76) * density,
        tiny_number * density,
        tiny_number * density,
        0,
    ]

    temperatures = np.linspace(5, 100, 3)
    temperatures *= 1.0e4
    for T in temperatures:
        test_equilibrium(
            solver_configs["default"], initial_conditions, rates, T, (0, 5)
        )
