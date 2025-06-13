# ===========================================================================
# main.jl - Main Orchestrator for the PIC-MCC Simulation
# ===========================================================================

# --- External Dependencies ---
using Plots
using DataFrames
using CSV
using Dates
using Statistics
using CUDA
using LinearAlgebra

# --- Set Up Plotting Backend ---
# Use a headless backend to avoid errors in server/container environments
#pyplot()
println("Plots.jl backend set to: ", backend_name())

# --- Include Simulation Modules ---
# The order is important to ensure dependencies are met.

# ===========================================================================
# main.jl - Main Orchestrator for the PIC-MCC Simulation
# ===========================================================================

# --- External Dependencies ---
using Plots
using DataFrames
using CSV
using Dates
using Statistics
using CUDA
using LinearAlgebra

# --- Set Up Plotting Backend ---
# Use a headless backend to avoid errors in server/container environments
#pyplot()
println("Plots.jl backend set to: ", backend_name())

# --- Include Simulation Modules ---
# The order is important to ensure dependencies are met.

include("utils/constants.jl")
include("utils/cross_sections.jl")
include("utils/particles.jl")
include("utils/collisions.jl")
include("utils/grid_functions.jl")
include("utils/electric_fields.jl")
include("utils/simulation_functions.jl")
include("utils/plotting_functions.jl")
include("utils/reporting_functions.jl")


# --- Global Simulation Flag ---
const USE_GPU = CUDA.functional()
if USE_GPU
    println("✅ NVIDIA GPU detected. Simulation will run on the GPU.")
else
    println("⚠️ No functional NVIDIA GPU detected. Simulation will run on the CPU.")
end

# ===========================================================================
# Simulation Setup
# ===========================================================================

function setup_simulation_environment()
    println("\n--- Setting up Simulation Environment ---")

    # --- Chamber and Grid Parameters ---
    chamber_dims = (width=0.1, length=0.1, height=0.001) # m
    grid_cells = (nx=6, ny=3, nz=50)
    total_cells = grid_cells.nx * grid_cells.ny * grid_cells.nz

    x_grid = LinRange(0, chamber_dims.width, grid_cells.nx + 1)
    y_grid = LinRange(0, chamber_dims.length, grid_cells.ny + 1)
    z_grid = LinRange(0, chamber_dims.height, grid_cells.nz + 1)
    
    x_cell_size = x_grid[2] - x_grid[1]
    y_cell_size = y_grid[2] - y_grid[1]
    z_cell_size = z_grid[2] - z_grid[1]
    cell_volume = x_cell_size * y_cell_size * z_cell_size

# --- Global Simulation Flag ---
const USE_GPU = CUDA.functional()
if USE_GPU
    println("✅ NVIDIA GPU detected. Simulation will run on the GPU.")
else
    println("⚠️ No functional NVIDIA GPU detected. Simulation will run on the CPU.")
end

# ===========================================================================
# Simulation Setup
# ===========================================================================

function setup_simulation_environment()
    println("\n--- Setting up Simulation Environment ---")

    # --- Chamber and Grid Parameters ---
    chamber_dims = (width=0.1, length=0.1, height=0.001) # m
    grid_cells = (nx=6, ny=3, nz=50)
    total_cells = grid_cells.nx * grid_cells.ny * grid_cells.nz

    x_grid = LinRange(0, chamber_dims.width, grid_cells.nx + 1)
    y_grid = LinRange(0, chamber_dims.length, grid_cells.ny + 1)
    z_grid = LinRange(0, chamber_dims.height, grid_cells.nz + 1)
    
    x_cell_size = x_grid[2] - x_grid[1]
    y_cell_size = y_grid[2] - y_grid[1]
    z_cell_size = z_grid[2] - z_grid[1]
    cell_volume = x_cell_size * y_cell_size * z_cell_size

    # --- PIC Simulation Parameters ---
    initial_temperature = 800.0     # K
    target_temperature = 2200.0    # K
    dt = 1e-13                      # s (timestep)
    simulated_electrons_per_step = 100
    physical_electrons_per_step = 5e12
    particle_weight = physical_electrons_per_step / simulated_electrons_per_step
    
    # --- Prepare CPU and GPU data structures ---
    # CPU data (always needed for some parts like conductivity calculation)
    populate_cpu_cross_sections!(air_composition_cpu)
    cpu_data = (air_composition=air_composition_cpu,)

    # GPU data (will be `nothing` if no GPU is available)
    gpu_data = setup_gpu_gas_data(USE_GPU, N2_CS_DATA, O2_CS_DATA)

    # --- Create directories for output ---
    isdir("plots") || mkdir("plots")
    isdir("simulation_data") || mkdir("simulation_data")

    # --- Base Parameters (will be updated by parameter search) ---
    base_params = (
        use_gpu = USE_GPU,
        dt = dt,
        initial_temperature = initial_temperature,
        target_temperature = target_temperature,
        simulated_electrons_per_step = simulated_electrons_per_step,
        particle_weight = particle_weight,
        chamber_dims = chamber_dims,
        x_grid = x_grid, y_grid = y_grid, z_grid = z_grid,
        x_cell_size = x_cell_size, y_cell_size = y_cell_size, z_cell_size = z_cell_size,
        cell_volume = cell_volume,
        total_cells = total_cells,
        min_energy_eV = 0.1,
        max_energy_eV = 1000.0,
        store_animation_data = false # Set to true for animations, but uses more memory
    )

    return base_params, cpu_data, gpu_data
end

# ===========================================================================
# Main Execution Block
# ===========================================================================

function main()
    base_params, cpu_data, gpu_data = setup_simulation_environment()

    # --- 1. Parameter Search ---
    println("\n--- Stage 1: Parameter Search ---")
    search_params = (
        energies = [50.0, 100.0],       # eV
        pressures = [1e6, 3e6],         # Pa
        fields = [0.5, 1.5],            # Tesla
        voltages = [50000, 10000, 200000, 500000],         # Volts
        max_steps = 50,                 # Use fewer steps for the search
        field_update_interval = 10
    )
    
    # The parameter search function is now cleaner
    search_df = parameter_search(search_params, base_params, cpu_data, gpu_data)

    # --- 2. Run Final Simulation with Optimal Parameters ---
    if isempty(search_df)
        println("\nParameter search yielded no results. Exiting.")
        return
    end
    
    best = first(search_df)
    println("\n--- Stage 2: Full Simulation with Optimal Parameters ---")
    println("Running with: E=$(best.ElectronEnergy)eV, P=$(best.Pressure/1e6)MPa, B=$(best.MagneticField)T, V=$(best.AnodeVoltage)V")

    # Create the final set of parameters for the detailed run
    final_run_params = merge(base_params, (
        electron_injection_energy_eV = best.ElectronEnergy,
        initial_pressure = best.Pressure,
        magnetic_field = [0.0, 0.0, best.MagneticField],
        anode_voltage = best.AnodeVoltage,
        max_steps = 5000, # Use more steps for the final run
        field_update_interval = 10,
        store_animation_data = true # Enable for final run
    ))
    
    # Add initial conditions to the final params
    initial_air_density_n = calculate_air_density_n(final_run_params.initial_pressure, final_run_params.initial_temperature)
    initial_electron_velocity = electron_velocity_from_energy(final_run_params.electron_injection_energy_eV)
    initial_positions, initial_velocities = initialize_electrons(0, final_run_params.chamber_dims, initial_electron_velocity)
    initial_temperature_grid = fill(final_run_params.initial_temperature, (length(final_run_params.x_grid)-1, length(final_run_params.y_grid)-1, length(final_run_params.z_grid)-1))

    final_run_params = merge(final_run_params, (
        initial_air_density_n = initial_air_density_n,
        initial_electron_velocity = initial_electron_velocity,
        initial_positions = initial_positions,
        initial_velocities = initial_velocities,
        initial_temperature_grid = initial_temperature_grid
    ))

    # Execute the main simulation
    final_results = run_pic_simulation(final_run_params, cpu_data, gpu_data, verbose=true)

    # --- 3. Post-Processing: Reporting and Visualization ---
    println("\n--- Stage 3: Post-Processing ---")
    
    # Generate text report
    generate_report("simulation_report.txt", final_run_params, final_results)
    
    # Export data to CSV files
    export_simulation_data_to_csv("simulation_data", final_run_params, final_results, search_df)
    
    # Generate plots
    if final_results.final_step > 0
        time_points = (0:final_results.final_step) .* (final_run_params.dt * 1e6)
        
        p1 = plot_temperature_vs_time(time_points, final_results.avg_temps_history, final_run_params.target_temperature)
        savefig(p1, "plots/temperature_vs_time.png")
        display(p1)

        # ... (add other plotting calls as needed) ...
        
        # Animate results if data was stored
        if final_run_params.store_animation_data
            animate_potential_slice(final_results.potential_history, final_run_params.x_grid, final_run_params.z_grid, 1)
            animate_electron_positions(final_results.position_history, final_run_params.chamber_dims.width, final_run_params.chamber_dims.height)
        end
    else
        println("Final simulation did not complete any steps. No plots will be generated.")
    end

    println("\nSimulation complete. Results, plots, and report saved.")
end

# --- Execute the main function ---
main()