import os
import h5py


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
