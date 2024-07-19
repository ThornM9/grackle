import numpy as np
import matplotlib.pyplot as plt
from solvers.pe import calculate_equilibrium_values
import os


def plot_solution(pred_t, pred_y, species_names, network_name, solver_name):
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    print("plotting solution")
    # total = np.zeros(values[0].shape)
    for i in range(len(pred_y)):
        if i < len(species_names):
            plt.plot(pred_t, pred_y[i], label=f"{species_names[i]} Solution")

    # plt.plot(t, total, label="total")
    plt.title("Solution")
    plt.xlabel("Time (Myr)")
    plt.ylabel("Density (g/cm^3)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_densities_solution.png")
    plt.clf()


def plot_prediction(exp_t, exp_y, species_names, network_name, solver_name):
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    print("plotting prediction")
    for i in range(len(exp_y)):
        if i < len(species_names):
            plt.plot(
                exp_t,
                exp_y[i],
                label=f"{species_names[i]} Predicted",
            )

    # total = np.sum(exp_y, 0)
    # total -= exp_y[5]  # don't include the electron density
    # plt.plot(exp_t, total, label="total density")

    plt.title(f"{network_name} Prediction")
    plt.xlabel("Time (Myr)")
    plt.ylabel("Density (g/cm^3)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_densities_prediction.png")
    plt.clf()


def plot_rate_values(
    exp_t, exp_y, rate_values, species_names, network_name, solver_name
):
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    print("plotting rate values")
    for i in range(len(exp_y)):
        if i < len(species_names):
            plt.plot(exp_t, rate_values[i], label=f"{species_names[i]} Rate")

    plt.xlabel("Time (Myr)")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_rate_values.png")
    plt.clf()


def plot_energy_and_temperature(
    network_config, exp_t, exp_y, rates, network_name, solver_name
):
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    print("plotting energy and temperature")
    plt.title("Energy and Temperature")
    temperature = np.array([])
    for i in range(len(exp_y[0])):
        gamma = network_config.calculate_gamma(exp_y[:, i], rates)
        T = network_config.calculate_temp_from_energy(exp_y[:, i], rates, gamma)
        temperature = np.append(temperature, T)
    # plt.plot(exp_t, exp_y[6], label="Energy")
    # plt.plot(exp_t, temperature, label="Temperature")
    # plt.xlabel("Time (Myr)")
    # plt.ylabel("Energy")

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Time (Myr)")
    ax1.set_ylabel("Energy", color=color)
    ax1.plot(exp_t, exp_y[6] * rates.chemistry_data.energy_units, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # Creating ax2, which shares the same x-axis with ax1
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Temperature", color=color)
    ax2.plot(exp_t, temperature, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Energy and Temperature")
    fig.tight_layout()

    plt.savefig(f"outputs/{network_name}/{solver_name}_energy.png")
    plt.clf()


def plot_mu(network_config, exp_t, exp_y, rates, network_name, solver_name):
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    print("plotting mu")
    mus = np.array([])
    for i in range(len(exp_y[0])):
        mu = network_config.calculate_mu(rates, exp_y[:, i])
        mus = np.append(mus, mu)

    plt.plot(exp_t, mus, label="Mu")
    plt.title("Mu")
    plt.xlabel("Time (Myr)")
    plt.ylabel("Mu")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{network_name}/{solver_name}_mu.png")
    plt.clf()


def plot_timestepper_data(exp_t, timestepper_data, network_name, solver_name):
    if timestepper_data is None:
        return
    if not os.path.exists(f"outputs/{network_name}"):
        os.makedirs(f"outputs/{network_name}")
    xaxis = timestepper_data["i"]
    # xaxis = exp_t
    print("plotting timestepper data")

    # plot the timestep vs the time
    plt.semilogy(timestepper_data["i"], exp_t, label="time")
    plt.title(f"{network_name} Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Time (Myr)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_timestep.png")
    plt.clf()

    # plot the dt values
    plt.semilogy(xaxis, timestepper_data["dt"], label="dt")
    plt.semilogy(xaxis, timestepper_data["trial_step"], label="trial step")
    plt.title(f"{network_name} Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Time (Myr)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_dt.png")
    plt.clf()

    # plot the trial step used
    plt.plot(
        xaxis,
        timestepper_data["trial_step_used"],
        label="trial step used",
    )
    plt.title(f"{network_name} Trial Step Used")
    plt.xlabel("Timestep")
    plt.ylabel("Time (Myr)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_trial_step_used.png")
    plt.clf()

    # plot the conservation satisfied
    plt.plot(
        xaxis,
        timestepper_data["conservation_satisfied"],
        label="conservation satisfied",
    )
    plt.plot(
        xaxis,
        timestepper_data["conservation_unsatisfied"],
        label="conservation unsatisfied",
    )
    plt.title(f"{network_name} Conservation Satisfaction")
    plt.xlabel("Timestep")
    plt.ylabel("Time (Myr)")
    plt.legend()
    plt.savefig(f"outputs/{network_name}/{solver_name}_conservation_satisfied.png")
    plt.clf()


def plot_partial_equilibriums(network_cfg, exp_t, exp_y, network_name, solver_name):

    if not os.path.exists(f"outputs/{network_name}/equilibrium_plots"):
        os.makedirs(f"outputs/{network_name}/equilibrium_plots")
    rg_cfg = network_cfg.reaction_group_config

    for rg_num in range(rg_cfg["rg_count"]):
        if rg_num == 4:
            continue
        print(f"plotting partial equilibrium {rg_num}")
        num_items = len(rg_cfg[rg_num])
        num_rows = 2 + (num_items > 4)
        num_cols = 2

        all_equilibrium_values = np.zeros((num_items, len(exp_t)))

        for i in range(len(exp_t)):
            gamma = network_cfg.calculate_gamma(exp_y[:, i], network_cfg.odes[0].rates)
            T = network_cfg.calculate_temp_from_energy(
                exp_y[:, i], network_cfg.odes[0].rates, gamma
            )
            rates = network_cfg.odes[0].rates
            equilibrium_values = calculate_equilibrium_values(
                rg_cfg,
                rg_num,
                exp_y[:, i],
                rates,
                T,
            )

            # if num_items == 3:
            #     kf = rg_cfg["get_kf"](rg_num, rates, T)
            #     kr = rg_cfg["get_kr"](rg_num, rates, T)
            #     creation = kf * equilibrium_values[0] * equilibrium_values[1]
            #     destruction = kr
            #     print(f"Creation: {creation}, Destruction: {destruction}")
            # if num_items == 4:
            #     kf = rg_cfg["get_kf"](rg_num, rates, 1000)
            #     kr = rg_cfg["get_kr"](rg_num, rates, 1000)
            #     creation = kf * equilibrium_values[0] * equilibrium_values[1]
            #     destruction = kr * equilibrium_values[2]
            #     print(f"Creation: {creation}, Destruction: {destruction}")

            all_equilibrium_values[:, i] = equilibrium_values

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 4))

        # Iterate through the array and plot each item
        for i in range(len(rg_cfg[rg_num])):
            row = i // 2
            col = i % 2
            if i < 4:
                ax = axes[row, col]
            else:
                ax = fig.add_subplot(num_rows, 1, 3)

            # reached_equilibrium = np.any(
            #     np.abs(all_equilibrium_values[i] - exp_y[i]) / exp_y[i] > 0.01
            # )
            nonzero = np.where(
                all_equilibrium_values == 0, 1e-50, all_equilibrium_values
            )
            ax.semilogx(exp_t, exp_y[i], label="Value")
            ax.semilogx(
                exp_t,
                nonzero[i],
                label="Equilibrium Value",
            )
            idx = rg_cfg[rg_num][i]
            ax.set_title(f"{network_cfg.species_names[idx]}")
            ax.legend()
            ax.set_xlabel("Time (Myr)")
            ax.set_ylabel("Density (g/cm^3)")

        # Adjust layout
        plt.tight_layout()

        plt.savefig(
            f"outputs/{network_name}/equilibrium_plots/rg_{rg_num}_equilibrium.png"
        )
