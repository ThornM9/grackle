# ########################################################################
# #
# # Cooling cell example script
# #
# #  This will initialize a single cell at a given temperature,
# #  iterate the cooling solver for a fixed time, and output the
# #  temperature vs. time.
# #
# #
# # Copyright (c) 2015-2016, Grackle Development Team.
# #
# # Distributed under the terms of the Enzo Public Licence.
# #
# # The full license is in the file LICENSE, distributed with this
# # software.
# ########################################################################

from matplotlib import pyplot
import os
import yt

from pygrackle import (
    FluidContainer,
    chemistry_data,
    evolve_constant_density,
    create_data_arrays,
)

from pygrackle.utilities.physical_constants import (
    mass_hydrogen_cgs,
    sec_per_Myr,
    cm_per_mpc,
)

tiny = 1e-20

if __name__ == "__main__":
    current_redshift = 0.0

    # Set initial values
    density = 0.1  # g /cm^3
    initial_temperature = 5e4  # K
    final_time = 0.2  # Myr

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 0
    my_chemistry.primordial_chemistry = 1
    my_chemistry.metal_cooling = 0
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = 0
    my_chemistry.H2_self_shielding = 0
    my_dir = os.path.dirname(os.path.abspath(__file__))
    grackle_data_file = bytearray(
        os.path.join(my_dir, "..", "..", "..", "input", "CloudyData_UVB=HM2012.h5"),
        "utf-8",
    )
    my_chemistry.grackle_data_file = grackle_data_file

    # my_chemistry.use_specific_heating_rate = 1
    # my_chemistry.use_volumetric_heating_rate = 1

    # Set units
    my_chemistry.comoving_coordinates = 0  # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs  # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc  # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr  # 1 Myr in s
    my_chemistry.set_velocity_units()

    rval = my_chemistry.initialize()

    fc = FluidContainer(my_chemistry, 1)
    fc["density"][:] = density
    # 0.76 * fc["density"]
    if my_chemistry.primordial_chemistry > 0:
        fc["HI"][:] = 0.76 * fc["density"]
        fc["HII"][:] = tiny * fc["density"]
        fc["HeI"][:] = 0.24 * fc["density"]
        fc["HeII"][:] = tiny * fc["density"]
        fc["HeIII"][:] = tiny * fc["density"]

        # fc["HI"][:] = tiny * fc["density"]
        # fc["HII"][:] = 0.76 * fc["density"]
        # fc["HeI"][:] = tiny * fc["density"]
        # fc["HeII"][:] = 0.02 * fc["density"]
        # fc["HeIII"][:] = 0.22 * fc["density"]
    if my_chemistry.primordial_chemistry > 1:
        fc["H2I"][:] = tiny * fc["density"]
        fc["H2II"][:] = tiny * fc["density"]
        fc["HM"][:] = tiny * fc["density"]
        fc["de"][:] = tiny * fc["density"]
    if my_chemistry.primordial_chemistry > 2:
        fc["DI"][:] = 2.0 * 3.4e-5 * fc["density"]
        fc["DII"][:] = tiny * fc["density"]
        fc["HDI"][:] = tiny * fc["density"]
    if my_chemistry.metal_cooling == 1:
        fc["metal"][:] = 0.1 * fc["density"] * my_chemistry.SolarMetalFractionByMass

    fc["x-velocity"][:] = 0.0
    fc["y-velocity"][:] = 0.0
    fc["z-velocity"][:] = 0.0

    fc["energy"][:] = initial_temperature / fc.chemistry_data.temperature_units
    fc.calculate_temperature()
    fc["energy"][:] *= initial_temperature / fc["temperature"]

    # timestepping safety factor
    safety_factor = 0.001

    # let gas cool at constant density
    data = evolve_constant_density(
        fc, final_time=final_time, safety_factor=safety_factor
    )

    if len(data["HI"]) == 1:
        data["time"] *= 2
        # data["time"].append(data["time"][-1] + 0.1)
        print(len(data["time"]))
        data["time"][-1] = data["time"][-2] + 0.1
        data["HI"] *= 2
        data["HII"] *= 2
        data["HeI"] *= 2
        data["HeII"] *= 2
        data["HeIII"] *= 2
        data["de"] *= 2

    data = create_data_arrays(fc, data)

    # plotting densities
    (p1,) = pyplot.plot(
        data["time"].to("Myr"), data["HI"] / my_chemistry.density_units, label="HI"
    )
    (p2,) = pyplot.plot(
        data["time"].to("Myr"), data["HII"] / my_chemistry.density_units, label="HII"
    )
    (p3,) = pyplot.plot(
        data["time"].to("Myr"), data["HeI"] / my_chemistry.density_units, label="HeI"
    )
    (p4,) = pyplot.plot(
        data["time"].to("Myr"), data["HeII"] / my_chemistry.density_units, label="HeII"
    )
    (p5,) = pyplot.plot(
        data["time"].to("Myr"),
        data["HeIII"] / my_chemistry.density_units,
        label="HeIII",
    )
    (p6,) = pyplot.plot(
        data["time"].to("Myr"),
        data["de"] / my_chemistry.density_units,
        label="Electron",
    )
    pyplot.legend(
        [p1, p2, p3, p4, p5],
        ["HI", "HII", "HeI", "HeII", "HeIII"],
        fancybox=True,
    )

    if my_chemistry.primordial_chemistry > 1:
        (p7,) = pyplot.plot(data["time"].to("Myr"), data["H2I"], label="H2I")
        (p8,) = pyplot.plot(data["time"].to("Myr"), data["HM"], label="HM")
        (p9,) = pyplot.plot(data["time"].to("Myr"), data["H2II"], label="H2II")
        pyplot.legend(
            [p1, p2, p3, p4, p5, p7, p8, p9],
            ["HI", "HII", "HeI", "HeII", "HeIII", "H2I", "HM", "H2II"],
            fancybox=True,
        )

    if my_chemistry.primordial_chemistry > 2:
        (p10,) = pyplot.plot(data["time"].to("Myr"), data["DI"], label="DI")
        (p11,) = pyplot.plot(data["time"].to("Myr"), data["DII"], label="DII")
        (p12,) = pyplot.plot(data["time"].to("Myr"), data["HDI"], label="HDI")
        pyplot.legend(
            [p1, p2, p3, p4, p5, p7, p8, p9, p10, p11, p12],
            [
                "HI",
                "HII",
                "HeI",
                "HeII",
                "HeIII",
                "H2I",
                "HM",
                "H2II",
                "DI",
                "DII",
                "HDI",
            ],
            fancybox=True,
        )

    pyplot.title("Grackle Densities")
    pyplot.savefig("cooling_cell.png")

    pyplot.clf()

    # plotting mu
    pyplot.xlabel("Time [Myr]")
    pyplot.ylabel("Mu")
    (p7,) = pyplot.plot(
        data["time"].to("Myr"),
        data["mu"],
        label="mu",
    )

    pyplot.legend()
    # pyplot.plot(p1)
    pyplot.title("Grackle Mu")

    pyplot.tight_layout()
    pyplot.savefig("cooling_cell_mu.png")
    pyplot.clf()

    # plotting temperature
    print("plotting energy and temperature")
    pyplot.title("Energy and Temperature")

    fig, ax1 = pyplot.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Time (Myr)")
    ax1.set_ylabel("Energy", color=color)
    ax1.plot(data["time"].to("Myr"), data["energy"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # Creating ax2, which shares the same x-axis with ax1
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Temperature", color=color)
    ax2.plot(data["time"].to("Myr"), data["temperature"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    # pyplot.xlabel("Time [Myr]")
    # pyplot.ylabel("Stuff")
    # (p8,) = pyplot.plot(
    #     data["time"].to("Myr"),
    #     data["temperature"],
    #     label="T",
    # )
    # pyplot.legend()
    # pyplot.tight_layout()
    pyplot.title("Grackle Energy and Temperature")
    pyplot.tight_layout()
    pyplot.savefig("cooling_cell_temp.png")

    pyplot.clf()

    # plot the timestep vs the time
    pyplot.semilogy(
        [i for i in range(len(data["time"]))], data["time"].to("Myr"), label="time"
    )
    pyplot.title(f"Timestep")
    pyplot.xlabel("Timestep")
    pyplot.ylabel("Time (Myr)")
    pyplot.legend()
    pyplot.savefig("cooling_cell_timestep.png")
    pyplot.clf()

    print("Number of timesteps: ", len(data["time"]))

    # # save data arrays as a yt dataset
    # yt.save_as_dataset({}, "cooling_cell.h5", data)


# from matplotlib import pyplot
# import os
# import yt

# from pygrackle import FluidContainer, chemistry_data, evolve_constant_density

# from pygrackle.utilities.physical_constants import (
#     mass_hydrogen_cgs,
#     sec_per_Myr,
#     cm_per_mpc,
# )

# tiny = 1e-20

# if __name__ == "__main__":
#     current_redshift = 0.0

#     # Set initial values
#     density = 0.1  # g /cm^3
#     initial_temperature = 1.0e6  # K
#     final_time = 100.0  # Myr

#     # Set solver parameters
#     my_chemistry = chemistry_data()
#     my_chemistry.use_grackle = 1
#     my_chemistry.with_radiative_cooling = 1
#     my_chemistry.primordial_chemistry = 0
#     my_chemistry.metal_cooling = 1
#     my_chemistry.UVbackground = 1
#     my_chemistry.self_shielding_method = 0
#     my_chemistry.H2_self_shielding = 0
#     my_dir = os.path.dirname(os.path.abspath(__file__))
#     grackle_data_file = bytearray(
#         os.path.join(my_dir, "..", "..", "..", "input", "CloudyData_UVB=HM2012.h5"),
#         "utf-8",
#     )
#     my_chemistry.grackle_data_file = grackle_data_file

#     # Set units
#     my_chemistry.comoving_coordinates = 0  # proper units
#     my_chemistry.a_units = 1.0
#     my_chemistry.a_value = 1.0 / (1.0 + current_redshift) / my_chemistry.a_units
#     my_chemistry.density_units = mass_hydrogen_cgs  # rho = 1.0 is 1.67e-24 g
#     my_chemistry.length_units = cm_per_mpc  # 1 Mpc in cm
#     my_chemistry.time_units = sec_per_Myr  # 1 Myr in s
#     my_chemistry.set_velocity_units()

#     rval = my_chemistry.initialize()

#     fc = FluidContainer(my_chemistry, 1)
#     fc["density"][:] = density
#     if my_chemistry.primordial_chemistry > 0:
#         fc["HI"][:] = 0.76 * fc["density"]
#         fc["HII"][:] = tiny * fc["density"]
#         fc["HeI"][:] = (1.0 - 0.76) * fc["density"]
#         fc["HeII"][:] = tiny * fc["density"]
#         fc["HeIII"][:] = tiny * fc["density"]
#     if my_chemistry.primordial_chemistry > 1:
#         fc["H2I"][:] = tiny * fc["density"]
#         fc["H2II"][:] = tiny * fc["density"]
#         fc["HM"][:] = tiny * fc["density"]
#         fc["de"][:] = tiny * fc["density"]
#     if my_chemistry.primordial_chemistry > 2:
#         fc["DI"][:] = 2.0 * 3.4e-5 * fc["density"]
#         fc["DII"][:] = tiny * fc["density"]
#         fc["HDI"][:] = tiny * fc["density"]
#     if my_chemistry.metal_cooling == 1:
#         fc["metal"][:] = 0.1 * fc["density"] * my_chemistry.SolarMetalFractionByMass

#     fc["x-velocity"][:] = 0.0
#     fc["y-velocity"][:] = 0.0
#     fc["z-velocity"][:] = 0.0

#     fc["energy"][:] = initial_temperature / fc.chemistry_data.temperature_units
#     fc.calculate_temperature()
#     fc["energy"][:] *= initial_temperature / fc["temperature"]

#     # timestepping safety factor
#     safety_factor = 0.01

#     # let gas cool at constant density
#     data = evolve_constant_density(
#         fc, final_time=final_time, safety_factor=safety_factor
#     )

#     (p1,) = pyplot.loglog(
#         data["time"].to("Myr"), data["temperature"], color="black", label="T"
#     )
#     pyplot.xlabel("Time [Myr]")
#     pyplot.ylabel("T [K]")

#     pyplot.twinx()
#     (p2,) = pyplot.semilogx(
#         data["time"].to("Myr"), data["mu"], color="red", label="$\\mu$"
#     )
#     pyplot.ylabel("$\\mu$")
#     pyplot.legend([p1, p2], ["T", "$\\mu$"], fancybox=True, loc="center left")
#     pyplot.tight_layout()
#     pyplot.savefig("cooling_cell.png")

#     # save data arrays as a yt dataset
#     yt.save_as_dataset({}, "cooling_cell.h5", data)
