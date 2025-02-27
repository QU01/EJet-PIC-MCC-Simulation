# Technical Report: PIC-MCC Simulation for Gas Discharge Plasma
### Nick Vázquez, Eng.
## Table of Contents
1.  **Introduction and Theoretical Framework**
2.  **Code Description**
3.  **Main Components**
4.  **Key Algorithms**
5.  **Validation and Optimization**
6.  **Conclusions**

---

## 1. Introduction and Theoretical Framework

This report documents an implementation of a plasma simulator using the Particle-In-Cell method with Monte Carlo Collisions (PIC-MCC) in Julia. The simulation models the behavior of electrons in a gas under the influence of magnetic fields, focusing on the energy transfer between electrons and gas molecules.

The simulation numerically solves the following physical equations:

1.  **Equation of motion** of charged particles under Lorentz force:
    $$\frac{d\vec{v}}{dt} = \frac{q}{m}(\vec{v} \times \vec{B})$$
    where $q$ is the electron charge, $m$ is the electron mass, $\vec{v}$ is the velocity, and $\vec{B}$ is the magnetic field.

2.  **Collision processes** between electrons and gas molecules:
    -   Elastic collisions
    -   Molecular excitation
    -   Ionization

3.  **Energy transfer** from electrons to the gas modeled according to:
    $$\Delta T = \frac{2}{3k_B n_{gas}V_{cell}} \Delta E_{transferred}$$
    where $k_B$ is the Boltzmann constant, $n_{gas}$ is the gas density, $V_{cell}$ is the cell volume, and $\Delta E_{transferred}$ is the transferred energy.

---

## 2. Usage Guide

### Prerequisites
1. **Julia**: Ensure you have Julia installed (version 1.8 or higher)
2. **Required Packages**:
   ```julia
   using Pkg
   Pkg.add(["Plots", "Interpolations", "DataFrames", "Statistics", "Random", "Dates"])
   ```

### Initial Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/pic-mcc-simulation.git
   cd pic-mcc-simulation
   ```

2. Open the main file `main.jl` to configure basic parameters:
   ```julia
   # --- Chamber Parameters ---
   chamber_width = 0.1    # Chamber width [m]
   chamber_length = 0.1   # Chamber length [m]
   chamber_height = 0.1   # Chamber height [m]
   
   # --- Simulation Parameters ---
   initial_temperature = 800.0    # Initial temperature [K]
   target_temperature = 2200.0    # Target temperature [K]
   dt = 1e-12                     # Time step [s]
   ```

### Running the Simulation
1. To execute the complete simulation:
   ```bash
   julia main.jl
   ```

2. The simulation will perform:
   - Automatic optimization of parameters
   - Complete PIC-MCC simulation
   - Generation of graphs and reports

### Results and Outputs
The program will generate the following files:
1. **Data Files**:
   - `optimal_parameters.txt`: Best parameters found
   - `simulation_data.csv`: Detailed simulation data

2. **Graphs** (in the `plots/` folder):
   - `temperature_vs_time.png`: Temperature evolution
   - `efficiency_vs_time.png`: Heating efficiency
   - `density_heatmap.png`: Electron density distribution
   - `temperature_heatmap.png`: Temperature distribution

3. **Final Report**:
   - `plots/simulation_report.txt`: Detailed simulation summary

### Customizing the Simulation
You can modify the following aspects:
1. **Physical Parameters**:
   ```julia
   # In main.jl
   electron_injection_energy_eV = 200    # Electron injection energy [eV]
   initial_pressure = 1e6                # Initial pressure [Pa]
   magnetic_field_strength = 0.5         # Magnetic field [T]
   ```

2. **Grid Configuration**:
   ```julia
   num_x_cells = 20    # Number of cells in x direction
   num_y_cells = 20    # Number of cells in y direction
   num_z_cells = 50    # Number of cells in z direction
   ```

3. **Simulation Conditions**:
   ```julia
   max_steps = 1000                     # Maximum number of steps
   simulated_electrons_per_step = 100    # Simulated electrons per step
   ```

### Data Analysis with Jupyter Notebook
The project includes a comprehensive data analysis notebook (`DataAnalysis.ipynb`) that provides advanced insights into the simulation results. To use it:

1. Ensure you have Jupyter Notebook installed
2. Open the notebook:
   ```bash
   jupyter notebook DataAnalysis.ipynb
   ```

The notebook includes:
- Statistical analysis of simulation results
- Efficiency trend visualizations
- Correlation heatmaps
- 3D relationship plots between parameters
- Advanced regression models for efficiency prediction
- Principal Component Analysis (PCA) for dimensionality reduction

### Parallel Execution (Optional)
For larger simulations, you can run in parallel:
```bash
julia -p 4 main.jl  # Use 4 processes
```

### Cleanup
To remove generated files:
```bash
rm -rf plots/*.png plots/simulation_report.txt optimal_parameters.txt simulation_data.csv
```

Now you're ready to run and modify the PIC-MCC simulation!


## 3. Code Description

The code implements a three-dimensional simulation that tracks the movement of individual electrons in a gas (air) under the influence of a magnetic field. It uses a hybrid PIC-MCC approach where:

-   Particles (electrons) move continuously in 3D space.
-   The gas is modeled as a continuous medium with uniform density.
-   Collisions are modeled using stochastic Monte Carlo processes.
-   Energy is deposited on a discrete grid for temperature calculation.

### Physical Constants and Air Properties

```julia
const k_b = 1.380649e-23       # Boltzmann constant [J/K]
const amu = 1.66054e-27        # Atomic mass unit [kg]
const electron_mass = 9.109e-31 # Electron mass [kg]
const electron_charge = -1.60218e-19 # Electron charge [C]

# Air composition modeled with N2 and O2
air_composition = Dict(
    "N2" => Dict{String, Any}(
        "mass" => 28.0134 * amu,        # Molecular mass [kg]
        "fraction" => 0.7808,           # Volume fraction
        "ionization_energy_eV" => 15.6  # Ionization potential [eV]
    ),
    "O2" => Dict{String, Any}(
        "mass" => 31.9988 * amu,
        "fraction" => 0.2095,
        "ionization_energy_eV" => 12.07
    ),
)
```

### Grid Parameters

```julia
chamber_width = 0.1    # Chamber width [m]
chamber_length = 0.1   # Chamber length [m]
chamber_height = 0.1   # Chamber height [m]
num_x_cells = 20       # Number of cells in x direction
num_y_cells = 20       # Number of cells in y direction
num_z_cells = 50       # Number of cells in z direction
```

---

## 4. Main Components

### 4.1 Collision Cross Sections

The code uses experimental data for the collision cross sections of Nitrogen (N2) and Oxygen (O2) based on Itikawa's studies. These cross sections describe the probability of different types of electron-molecule interactions as a function of electron energy.

```julia
# Interpolated cross sections for continuous calculation
n2_total_cross_section_func = LinearInterpolation(n2_energy_eV, n2_total_cross_section; extrapolation_bc = Flat())
n2_ionization_cross_section_func = LinearInterpolation(n2_energy_eV, n2_ionization_cross_section; extrapolation_bc = Flat())

o2_total_cross_section_func = LinearInterpolation(o2_energy_eV, o2_total_cross_section; extrapolation_bc = Flat())
o2_ionization_cross_section_func = LinearInterpolation(o2_energy_eV, o2_ionization_cross_section; extrapolation_bc = Flat())
```

The excitation cross section is estimated with:

```julia
function estimate_excitation_cross_section(total_cross_section, ionization_cross_section, energy_eV, ionization_energy_eV)
    cross_section_diff = total_cross_section - ionization_cross_section
    excitation_cross_section = max(0.0, cross_section_diff)
    energy_above_threshold = max(0.0, energy_eV - ionization_energy_eV)
    excitation_fraction = exp(-0.1 * energy_above_threshold)
    excitation_cross_section = excitation_cross_section * excitation_fraction
    return excitation_cross_section
end
```

This function estimates the excitation cross section as the difference between the total and ionization cross sections, modulated by a factor that decreases exponentially with energy above the ionization threshold.

### 4.2 Calculation of Electron Density and Velocity

```julia
function calculate_air_density_n(pressure, temperature)
    return pressure / (k_b * temperature)
end

function electron_velocity_from_energy(electron_energy_eV)
    energy_joules = electron_energy_eV * 1.60218e-19
    return sqrt(2 * energy_joules / electron_mass)
end

function electron_energy_from_velocity(velocity_magnitude)
    return 0.5 * electron_mass * (velocity_magnitude ^ 2)
end
```

These functions implement:
-   The ideal gas law:  $n = \frac{P}{k_B T}$
-   The relationship between kinetic energy and velocity: $E_k = \frac{1}{2}mv^2$

### 4.3 Electron Initialization

```julia
function initialize_electrons(num_electrons, chamber_width, chamber_length, initial_electron_velocity)
    rng = MersenneTwister(0)
    x_positions = rand(rng, num_electrons) * chamber_width
    y_positions = rand(rng, num_electrons) * chamber_length
    z_positions = zeros(num_electrons)

    vx = zeros(num_electrons)
    vy = zeros(num_electrons)
    vz = fill(initial_electron_velocity, num_electrons)

    positions = hcat(x_positions, y_positions, z_positions)
    velocities = hcat(vx, vy, vz)
    return positions, velocities
end
```

Electrons are initialized with a uniform spatial distribution in the XY plane at the base of the chamber (z=0), with initial velocities directed upwards (+z).

---

## 5. Key Algorithms

### 5.1 Boris Algorithm for Movement in a Magnetic Field

```julia
function move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
    new_velocities = copy(velocities)
    B_magnitude = norm(magnetic_field)

    # If there is no significant magnetic field, simple movement
    if B_magnitude < 1e-10
        new_positions = positions + velocities * dt
        return new_positions, new_velocities
    end

    # Unit vector in the direction of the magnetic field
    b_hat = magnetic_field / B_magnitude

    # Rotation angle during half a step
    theta = electron_charge * B_magnitude * dt / (2 * electron_mass)

    # Factors for rotation
    cos_2theta = cos(2 * theta)
    sin_2theta = sin(2 * theta)

    # For each electron
    for i in 1:size(velocities, 1)
        v = velocities[i, :]
        v_mag_initial = norm(v)

        # Decompose velocity: parallel and perpendicular components to B
        v_parallel = dot(v, b_hat) * b_hat
        v_perp = v - v_parallel

        # Rotate only the perpendicular component
        v_perp_mag = norm(v_perp)

        if v_perp_mag > 1e-10
            # Unit vectors perpendicular in the rotation plane
            e1 = v_perp / v_perp_mag
            e2 = cross(b_hat, e1)

            # Rotation of the perpendicular component
            v_perp_rotated = v_perp_mag * (cos_2theta * e1 + sin_2theta * e2)

            # New velocity: parallel + rotated perpendicular
            new_velocities[i, :] = v_parallel + v_perp_rotated

            # Ensure exact energy conservation
            v_mag_final = norm(new_velocities[i, :])
            if abs(v_mag_final - v_mag_initial) / v_mag_initial > 1e-10
                new_velocities[i, :] *= (v_mag_initial / v_mag_final)
            end
        end
    end

    # Update positions (second-order method)
    new_positions = positions + 0.5 * (velocities + new_velocities) * dt

    return new_positions, new_velocities
end
```

The Boris algorithm provides:
1.  Exact energy conservation (only changes the direction, not the magnitude of the velocity).
2.  Long-term numerical stability.
3.  Second-order accuracy in time.

The electron cyclotron frequency is calculated as:
$$\omega_c = \frac{|q|B}{m}$$

And the period:
$$T_c = \frac{2\pi}{\omega_c} = \frac{2\pi m}{|q|B}$$

### 5.2 Boundary Conditions

```julia
function apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)
    x = positions[:, 1]
    y = positions[:, 2]
    z = positions[:, 3]
    vx = velocities[:, 1]
    vy = velocities[:, 2]
    vz = velocities[:, 3]

    reflect_x_min = x .< 0
    reflect_x_max = x .> chamber_width
    x = ifelse.(reflect_x_min, .-x, ifelse.(reflect_x_max, chamber_width .- (x .- chamber_width), x))
    vx = ifelse.(reflect_x_min .| reflect_x_max, .-vx, vx)

    # [similar procedures for y, z]

    positions_updated = hcat(x, y, z)
    velocities_updated = hcat(vx, vy, vz)

    return positions_updated, velocities_updated
end
```

It implements elastic reflections on the walls of the rectangular chamber, inverting the velocity component perpendicular to the contact surface.

### 5.3 Monte Carlo Collisions

```julia
function monte_carlo_collision(positions, velocities, air_density_n, dt, rng, efficiency)
    num_particles = size(velocities, 1)

    # Vectors for results
    inelastic_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    # Initial energy calculations
    v_magnitudes = sqrt.(sum(velocities.^2, dims=2))
    E_e_eV = electron_energy_from_velocity.(v_magnitudes) ./ 1.60218e-19
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)

    # For each particle
    for i in 1:num_particles
        # Calculation of collision probability with each gas
        total_collision_prob = 0.0
        gas_probs = Dict{String, Float64}()

        for gas_name in keys(air_composition)
            gas_info = air_composition[gas_name]
            gas_fraction = gas_info["fraction"]
            total_cross_section_func = gas_info["total_cross_section_func"]
            total_cross_section = total_cross_section_func(E_e_eV[i])

            # Collision probability: P = 1 - exp(-n*σ*v*dt)
            gas_prob = (1.0 - exp(-(air_density_n * gas_fraction) * total_cross_section * v_magnitudes[i] * dt))
            gas_probs[gas_name] = gas_prob
            total_collision_prob += gas_prob * gas_fraction
        end

        # Check if collision occurs
        if rand(rng) < total_collision_prob
            # [Selection of gas and collision type]

            # Decide collision type based on relative cross sections
            collision_rand = rand(rng)

            if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
                # Ionization collision
                energy_loss_joules = min(ionization_energy_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > excitation_energy_joules
                # Excitation collision
                energy_loss_joules = min(excitation_energy_loss_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            else
                # Elastic collision
                fraction_energy_elastic = (2 * electron_mass / gas_mass)
                energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
                elastic_energy_transfer[i] = energy_loss_joules
            end

            # Update velocity with remaining energy
            remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
            new_speed = sqrt(2 * remaining_energy_joules / electron_mass)

            # New random direction for scattering
            new_direction = randn(rng, 3)
            new_direction = new_direction / norm(new_direction)

            # Assign new velocity
            velocities_new[i, :] = new_direction * new_speed
            collided_flags[i] = true
        end
    end

    return positions, velocities_new, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer
end
```

The collision probability for a particle with velocity $v$ in a time $dt$ is:

$$P_{collision} = 1 - \exp(-n \sigma v dt)$$

Where:
-   $n$ is the gas density.
-   $\sigma$ is the total cross section.
-   $v$ is the electron velocity.
-   $dt$ is the time step.

For elastic collisions, the fraction of energy transferred is:

$$\frac{\Delta E}{E} = \frac{2m_e}{M_{gas}}$$

Where $m_e$ is the electron mass and $M_{gas}$ is the molecular mass of the gas.

### 5.4 Energy Deposition on the Grid

```julia
function deposit_energy(positions, energy_transfer, x_grid, y_grid, z_grid)
    bins = (collect(x_grid), collect(y_grid), collect(z_grid))
    w = Weights(energy_transfer)
    h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), w, bins)
    return h.weights
end

function update_temperature_grid(temperature_grid, energy_deposition_grid, elastic_energy_deposition_grid, n_air, cell_volume)
    n_air_cell = n_air * cell_volume
    delta_T_inelastic = (2.0 / (3.0 * k_b)) * (energy_deposition_grid ./ n_air_cell)
    delta_T_elastic = (2.0 / (3.0 * k_b)) * (elastic_energy_deposition_grid ./ n_air_cell)
    new_temperature_grid = temperature_grid .+ delta_T_inelastic .+ delta_T_elastic
    return new_temperature_grid
end
```
The temperature change is calculated according to:

$$\Delta T = \frac{2}{3 k_B n V_{cell}} \Delta E$$

This equation comes from the relationship between internal energy and temperature for an ideal gas:
$$E_{internal} = \frac{3}{2} n k_B T V_{cell}$$

### 5.5 Main Simulation Algorithm

```julia
function run_pic_simulation(initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass, initial_electron_velocity; verbose=true)

    # Initialization of histories

    while avg_temps_history[end] < target_temperature && step < max_steps
        step += 1

        # 1. Inject new electrons
        new_positions, new_velocities = initialize_electrons(simulated_electrons_per_step, chamber_width, chamber_length, initial_electron_velocity)
        positions = vcat(positions, new_positions)
        velocities = vcat(velocities, new_velocities)

        # 2. Calculate input energy
        step_input_energy = sum(electron_energy_from_velocity.(sqrt.(sum(new_velocities.^2, dims=2)))) * particle_weight

        # 3. Electron movement in magnetic field
        positions, velocities = move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)

        # 4. Apply boundary conditions
        positions, velocities = apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)

        # 5. Collision simulation
        positions, velocities, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer = monte_carlo_collision(
            positions, velocities, initial_air_density_n, dt, rng, 1.0
        )

        # 6. Deposit energy on the grid
        energy_deposition_grid_step = deposit_energy(positions, inelastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)
        elastic_energy_deposition_grid_step = deposit_energy(positions, elastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)

        # 7. Update temperature
        temperature_grid = update_temperature_grid(temperature_grid, energy_deposition_grid_step, elastic_energy_deposition_grid_step, initial_air_density_n, cell_volume)

        # 8. Calculate efficiency
        current_internal_energy = (3/2) * k_b * Statistics.mean(temperature_grid) * initial_air_density_n * cell_volume * TOTAL_CELLS
        energy_transferred_to_gas = current_internal_energy - previous_internal_energy
        step_efficiency = (energy_transferred_to_gas / step_input_energy) * 100

        # 9. Update histories
        push!(avg_temps_history, Statistics.mean(temperature_grid))
        push!(efficiency_history, step_efficiency)
    end

    return results...
end
```

---

## 6. Validation and Optimization

### 6.1 Energy Conservation Verification

The code implements several checks to ensure that energy is properly conserved during the simulation:

1.  **In the Boris algorithm**: Ensuring that the movement in the magnetic field does not alter the kinetic energy.
2.  **In the collision process**: Verifying that the transferred energy does not exceed the electron's energy.
3.  **In the global balance**: Checking that the sum of energy transferred to the gas plus the remaining energy in the electrons equals the total input energy.

The expression that verifies conservation is:

$$\left|\frac{E_{loss} - E_{transferred}}{E_{transferred}}\right| < \epsilon$$

where $\epsilon$ is a small tolerance (typically 0.01).

### 5.2 Time Step Validity

The time step must satisfy two conditions:

1.  **Courant-Friedrichs-Lewy (CFL) condition** for spatial stability:
    $$dt < \frac{\Delta x}{v_{max}}$$

2.  **Cyclotron condition** for accuracy in a magnetic field:
    $$dt < \frac{0.1 \cdot 2\pi m_e}{|q|B}$$

### 6.3 Parametric Search for Optimization
The `parameter_search()` function systematically explores the parameter space:

```julia
function parameter_search()
    # Exploration ranges
    electron_energies = [50.0, 100.0, 150.0, 200.0]  # eV
    pressures = [1e6, 2e6, 3e6, 4e6]  # Pa
    magnetic_fields = [0.5, 1.0, 1.5, 2.0]  # Tesla

    # For each combination of parameters
    for energy in electron_energies
        for pressure in pressures
            for field in magnetic_fields
                # Evaluate efficiency
                efficiency = estimate_efficiency(energy, pressure, field)

                # Record results
                push!(results, (energy, pressure, field, efficiency))
            end
        end
    end

    # Sort by efficiency
    sort!(results, :Efficiency, rev=true)
    return results, best_params, best_efficiency
end
```
The efficiency is calculated as:
$$\eta = \frac{E_{transferred \, to \, gas}}{E_{total \, input}} \times 100\%$$

---
## 7. Conclusions

The implemented PIC-MCC simulation allows modeling the heating of a gas by electron impact under the influence of magnetic fields. The main conclusions are:

1.  **Boris Algorithm**: Fundamental for the correct simulation of electron movement in a magnetic field, guaranteeing energy conservation.

2.  **Monte Carlo Collision Model**: The stochastic approach provides an adequate physical representation of electron-molecule interactions.

3.  **Conservation Verification**: The implemented validations ensure that the results are physically plausible.

4.  **Parametric Optimization**: Systematic search allows identifying the optimal parameters (electron energy, pressure, magnetic field) to maximize heating efficiency.

5.  **Detailed Diagnostics**: Step-by-step tracking of energy and temperature allows understanding the physics of the process.
The results show that the heating efficiency strongly depends on the operating parameters, and the model provides a valuable tool for optimizing these parameters in practical applications of plasma generation and gas heating.

## 8. Simulation Results

### --- PIC-MCC Simulation Results ---

**Date and Time:** 2025-02-26T14:29:02.183

**--- Simulation Parameters ---**
* Initial Temperature: 800.0 K
* Initial Pressure: 1.0 MPa
* Initial Air Density: 9.0537131450499e25 m^-3
* Initial Electron Energy: 50.0 eV
* Axial Magnetic Field (Bz): 0.5 T
* Time Step (dt): 1.0e-12 s
* Simulation Time per Step: 1.0e-12 s
* Simulated Electrons per Step: 100
* Target Temperature: 2200.0 K
* Maximum Number of Steps: 1000

**--- Results ---**
* Final Average Temperature: 2201.1098822691433 K
* Reached Target Temperature: true
* Number of Simulated Steps: 686
* Total Simulation Time: 6.86e-10 s
* Total Increase in Air Internal Energy: 2627.0810292546444 J
* Total Energy Introduced by Electrons: 2747.7387000000463 J
* Simulation Efficiency: 95.07649872269027 %
* Average Efficiency per Step: 95.07649872269027 %

**--- Plots ---**
* Average Temperature vs Time: temperature_vs_time.png
* Average Efficiency vs Time: efficiency_vs_time.png
* Final Electron Density (Slice): density_heatmap.png
* Final Air Temperature (Slice): temperature_heatmap.png

### Analysis of Results
The PIC-MCC simulation of air heating to a target temperature of 2200.0 K has been successful. The results indicate the following:

*   **Efficient Heating:** The simulation reached the target temperature (2200.0 K) with a final average temperature of 2201.1 K, slightly exceeding the target. This was achieved in 686 simulation steps, corresponding to a total simulation time of 6.86e-10 seconds.

*    **High Energy Efficiency:** The overall simulation efficiency is remarkably high, reaching 95.08%.  This means that 95.08% of the total energy introduced by the electrons was transferred to the air, resulting in an increase in temperature. The average efficiency per step matches the overall efficiency, suggesting consistent energy transfer throughout the simulation.

*   **Visualization of Temperature vs. Time:** The [Average Temperature vs. Time](temperature_vs_time.png) graph shows the evolution of the air temperature throughout the simulation. A steady and rapid increase in temperature is observed from the initial temperature of 800.0 K until reaching and slightly exceeding the target temperature of 2200.0 K. The dashed red line in the graph clearly indicates the target temperature, allowing visualization of when it is reached.

*   **Visualization of Efficiency vs. Time:** The [Average Efficiency vs. Time](efficiency_vs_time.png) graph presents the heating efficiency per step over time. It can be seen that the efficiency quickly stabilizes at a high value, close to 95%, with some minor fluctuations. This confirms the high efficiency of the simulated heating process and its consistency throughout the simulation.

*   **Next Steps:** The [Final Electron Density (Slice)](density_heatmap.png) and [Final Air Temperature (Slice)](temperature_heatmap.png) graphs (not included in this report) would provide detailed spatial information on the electron density distribution and air temperature at the end of the simulation. These graphs would be useful for analyzing the uniformity of heating and the spatial distribution of the generated plasma.

In summary, the simulation demonstrates an efficient and rapid method for heating air using a plasma generated by gas discharge, with a high efficiency of energy transfer from electrons to the gas. The results obtained are consistent with the simulation parameters and provide valuable information for the optimization of plasma heating systems.