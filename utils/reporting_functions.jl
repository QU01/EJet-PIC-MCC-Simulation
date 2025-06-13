# ---------------------------------------------------------------------------
# FILE: reporting_functions.jl (Functions for reporting and data export)
# ---------------------------------------------------------------------------

using Dates
using DataFrames
using CSV
using CUDA # Needed to check for CuArray type

# Assumes K_B is a global constant defined elsewhere
# Assumes a to_cpu utility function is defined (e.g., in visualization.jl)
# to_cpu(data::AbstractArray) = Array(data)
# to_cpu(data) = data

# --- Energy and Efficiency Calculations ---

"""
Calculates the overall thermal efficiency based on the change in the gas's internal energy.
"""
function calculate_energy_based_efficiency(initial_temperature, final_temperature,
                                           initial_air_density_n, cell_volume, total_cells,
                                           accumulated_input_energy)
    # Calculate initial and final internal energy of the ideal gas
    initial_internal_energy = (3/2) * K_B * initial_temperature * initial_air_density_n * cell_volume * total_cells
    final_internal_energy = (3/2) * K_B * final_temperature * initial_air_density_n * cell_volume * total_cells
    delta_internal_energy = final_internal_energy - initial_internal_energy

    # Calculate efficiency
    efficiency = 0.0
    if accumulated_input_energy > 1e-20 # Avoid division by zero
        efficiency = (delta_internal_energy / accumulated_input_energy) * 100
    end

    return efficiency, delta_internal_energy, initial_internal_energy, final_internal_energy
end

# --- Simulation Report Generation ---

"""
Generates a summary text report of the simulation run.
Handles both CPU (Array) and GPU (CuArray) data by converting to CPU first.
"""
function generate_report(filename, params, results)
    # GPU: Ensure all data that might come from the GPU is moved to the CPU
    final_avg_temp = to_cpu(results.avg_temps_history[end])

    magnetic_field_strength = norm(params.magnetic_field)
    
    # Open file to write the report
    open(filename, "w") do io
        println(io, "--- PIC-MCC Simulation Report ---")
        println(io, "Date and Time: $(Dates.now())")
        println(io, "\n--- Simulation Parameters ---")
        println(io, "Initial Temperature: $(params.initial_temperature) K")
        println(io, "Target Temperature: $(params.target_temperature) K")
        println(io, "Initial Pressure: $(params.initial_pressure / 1e6) MPa")
        println(io, "Initial Air Density: $(params.initial_air_density_n) m^-3")
        println(io, "Electron Injection Energy: $(params.electron_injection_energy_eV) eV")
        println(io, "Axial Magnetic Field (Bz): $(magnetic_field_strength) T")
        println(io, "Anode Voltage: $(params.anode_voltage) V")
        println(io, "Time Step (dt): $(params.dt) s")
        println(io, "Simulated Electrons per Step: $(params.simulated_electrons_per_step)")
        println(io, "Max Simulation Steps: $(params.max_steps)")

        println(io, "\n--- Simulation Results ---")
        println(io, "Final Average Temperature: $(round(final_avg_temp, digits=2)) K")
        println(io, "Reached Target Temperature: $(results.reached_target_temp)")
        println(io, "Number of Steps Simulated: $(results.final_step)")
        println(io, "Total Simulated Time: $(results.final_step * params.dt * 1e6) µs")
        println(io, "Average Step Efficiency: $(round(results.avg_efficiency, digits=2)) %")
        println(io, "Average Electron Lifetime: $(round(results.avg_electron_lifetime * 1e9, digits=3)) ns")
        println(io, "Estimated Final Plasma Conductivity: $(round(results.plasma_conductivity, sigdigits=3)) S/m")

        # Detailed energy balance calculation
        eff, delta_U, initial_U, final_U = calculate_energy_based_efficiency(
            params.initial_temperature, final_avg_temp, params.initial_air_density_n,
            params.cell_volume, params.total_cells, results.accumulated_input_energy
        )
        
        println(io, "\n--- Overall Energy Balance ---")
        println(io, "Total Injected Electron Energy: $(results.accumulated_input_energy) J")
        println(io, "Initial Gas Internal Energy: $(initial_U) J")
        println(io, "Final Gas Internal Energy: $(final_U) J")
        println(io, "Change in Internal Energy (ΔU): $(delta_U) J")
        println(io, "Overall Thermal Efficiency (ΔU / E_in): $(round(eff, digits=2)) %")

        println(io, "\n--- Generated Files ---")
        println(io, "Temperature vs. Time Plot: plots/temperature_vs_time.png")
        println(io, "Efficiency vs. Time Plot: plots/efficiency_vs_time.png")
        println(io, "Final Electron Density Heatmap: plots/density_heatmap.png")
        println(io, "Final Temperature Heatmap: plots/temperature_heatmap.png")
    end
    println("\nSimulation report saved to: $(filename)")
end

# --- Data Export to CSV ---

"""
Exports simulation time-series, final grids, and parameter search results to CSV files.
Handles both CPU (Array) and GPU (CuArray) data.
"""
function export_simulation_data_to_csv(data_dir, params, results, parameter_search_df)
    println("\n--- Exporting data to CSV for external visualization ---")
    isdir(data_dir) || mkdir(data_dir)

    # GPU: Ensure final grid data is on the CPU before processing
    final_density_grid_cpu = to_cpu(results.final_density_grid)
    final_temperature_grid_cpu = to_cpu(results.final_temperature_grid)

    # 1. Export temperature vs. time
    time_points = (0:results.final_step) .* (params.dt * 1e6)
    temp_time_df = DataFrame(Time_microseconds=time_points, Temperature_K=to_cpu(results.avg_temps_history))
    CSV.write(joinpath(data_dir, "temperature_vs_time.csv"), temp_time_df)

    # 2. Export efficiency vs. time
    if !isempty(results.efficiency_history)
        eff_time_points = (1:results.final_step) .* (params.dt * 1e6)
        efficiency_time_df = DataFrame(Time_microseconds=eff_time_points, Efficiency_percent=results.efficiency_history)
        CSV.write(joinpath(data_dir, "efficiency_vs_time.csv"), efficiency_time_df)
    end

    # 3. Export final temperature grid (slice)
    z_slice_index = size(final_temperature_grid_cpu, 3) ÷ 2
    temp_slice = final_temperature_grid_cpu[:, :, z_slice_index]
    temp_grid_df = DataFrame(
        [(x, y, temp_slice[i, j]) for (i, x) in enumerate(params.x_grid[1:end-1]) for (j, y) in enumerate(params.y_grid[1:end-1])],
        [:x_m, :y_m, :temperature_K]
    )
    CSV.write(joinpath(data_dir, "temperature_grid_slice.csv"), temp_grid_df)

    # 4. Export final electron density grid (slice)
    density_slice = final_density_grid_cpu[:, :, z_slice_index]
    density_grid_df = DataFrame(
        [(x, y, density_slice[i, j]) for (i, x) in enumerate(params.x_grid[1:end-1]) for (j, y) in enumerate(params.y_grid[1:end-1])],
        [:x_m, :y_m, :electron_density_count]
    )
    CSV.write(joinpath(data_dir, "density_grid_slice.csv"), density_grid_df)

    # 5. Export simulation parameters
    sim_params_df = DataFrame(Parameter=string.(keys(params)), Value=string.(values(params)))
    CSV.write(joinpath(data_dir, "simulation_parameters.csv"), sim_params_df)

    # 6. Export detailed step-by-step data
    if !isempty(results.detailed_data)
        detailed_df = DataFrame(results.detailed_data)
        detailed_df.Time_microseconds = (1:nrow(detailed_df)) .* (params.dt * 1e6)
        CSV.write(joinpath(data_dir, "detailed_simulation_data.csv"), detailed_df)
    end

    # 7. Export parameter search results
    if !isnothing(parameter_search_df)
        CSV.write(joinpath(data_dir, "parameter_search_results.csv"), parameter_search_df)
        
        # Create comparative analysis datasets
        # (This part is complex and depends on the exact columns of parameter_search_df)
        println("Parameter search results exported.")
    end

    println("Data successfully exported to the '$(data_dir)' folder.")
end