import numpy as np
import matplotlib.pyplot as plt


def plot_solution(pred_t, pred_y, species_names, network_name, solver_name):
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
