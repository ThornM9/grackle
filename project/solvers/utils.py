import os
import h5py


# this function is just a wrapper to pass to scipy to solve for an example solution
def network(t, y, network_cfg, T):
    odes = network_cfg.odes
    gamma = network_cfg.calculate_gamma(y, odes[0].rates)
    T = network_cfg.calculate_temp_from_energy(y, odes[0].rates, gamma)

    results = []
    for i, ode in enumerate(odes):
        er = ode.get_rates(y, T)
        k_n = sum(er.destruction_rates) * er.destruction_sign
        creation = sum(er.positive_fluxes)

        destruction = k_n * y[i]
        results.append(creation - destruction)

    return results


def get_cloudy_rates():
    my_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(my_dir, "..", "..", "input", "CloudyData_UVB=HM2012.h5")

    with h5py.File(file_name, "r") as file:
        return {
            "piHI": file["UVBRates"]["Photoheating"]["piHI"][0],
            "piHeI": file["UVBRates"]["Photoheating"]["piHeI"][0],
            "piHeII": file["UVBRates"]["Photoheating"]["piHeII"][0],
            "k27": file["UVBRates"]["Chemistry"]["k27"][0],
        }


def calculate_population(y_values, i, equations, include_electrons=True):
    pop = 0
    for j, eq in enumerate(equations):
        if eq.is_energy or (eq.is_electron and not include_electrons):
            continue
        pop += y_values[j, i]
    return pop


def get_initial_conditions(density, network_name, initial_gas_state):
    initial_conditions = []
    tiny = 1e-20
    if initial_gas_state == "neutral":
        initial_conditions = [
            0.76 * density,
            tiny * density,
            0.24 * density,
            tiny * density,
            tiny * density,
            tiny * density,
        ]
    elif initial_gas_state == "ionised":
        initial_conditions = [
            tiny * density,
            0.76 * density,
            tiny * density,
            0.02 * density,
            0.22 * density,
            0.87998 * density,
        ]
    else:
        raise Exception("Invalid initial gas state")

    if network_name == "nine":
        initial_conditions.extend([tiny * density] * 3)

    if network_name == "twelve":
        initial_conditions.extend([tiny * density] * 3)
        initial_conditions.extend(
            [2.0 * 3.4e-5 * density, tiny * density, tiny * density]
        )

    initial_conditions.append(None)  # energy

    return initial_conditions
