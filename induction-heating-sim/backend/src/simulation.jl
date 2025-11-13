"""
Main Simulation Module: Electromagnetic Induction Heating Simulation
Orchestrates the complete 2D axisymmetric simulation
"""
module Simulation

include("physics.jl")
include("solver.jl")

using .Physics
using .Solver
using JSON3
using Printf

"""
SimulationConfig: Configuration for simulation run
"""
struct SimulationConfig
    # Time control
    t_final::Float64
    CFL::Float64
    save_interval::Float64

    # Grid resolution
    Nr::Int
    Nz::Int

    # Output
    output_dir::String
end

"""
SimulationResults: Container for simulation outputs
"""
struct SimulationResults
    times::Vector{Float64}
    efficiencies::Vector{Float64}
    max_temperatures::Vector{Float64}
    max_pressures::Vector{Float64}
    avg_exit_temp::Vector{Float64}

    # Final state snapshots
    r::Vector{Float64}
    z::Vector{Float64}
    ρ_final::Matrix{Float64}
    T_final::Matrix{Float64}
    p_final::Matrix{Float64}
    u_z_final::Matrix{Float64}
    u_r_final::Matrix{Float64}
    q_v_final::Matrix{Float64}
end

"""
    create_default_params()

Create default physical parameters for hydrogen flow
"""
function create_default_params()
    return PhysicalParams(
        # Geometry
        R_in = 0.02,      # 2 cm inner radius
        R_out = 0.025,    # 2.5 cm outer radius (0.5 cm wall)
        L_z = 0.5,        # 50 cm length

        # Coil
        N_coil = 100,
        I₀ = 500.0,       # 500 A peak
        freq = 50e3,      # 50 kHz
        R_coil = 0.5,     # 0.5 Ω
        R_b = 0.03,       # 3 cm coil radius
        z_coil = 0.25,    # Centered

        # Fluid (preionized hydrogen)
        σ_fluid = 500.0,  # 500 S/m (with preionization)
        c_p = 14300.0,    # J/(kg·K)
        c_v = 10200.0,    # J/(kg·K)
        γ = 1.41,
        R_gas = 4124.0,   # J/(kg·K)
        μ = Physics.μ₀,

        # Inlet
        p_in = 101325.0,  # 1 atm
        T_in = 300.0,     # 300 K
        u_in = 50.0       # 50 m/s
    )
end

"""
    run_simulation(params, config)

Run complete simulation
"""
function run_simulation(params::PhysicalParams, config::SimulationConfig)
    println("=" ^ 70)
    println("ELECTROMAGNETIC INDUCTION HEATING SIMULATION")
    println("=" ^ 70)
    println()

    # Initialize grid
    @printf("Creating grid: %d x %d cells\n", config.Nr, config.Nz)
    grid = Grid(params.R_in, params.R_out, params.L_z, config.Nr, config.Nz)

    # Initialize flow state
    println("Initializing flow field...")
    state = FluidState(grid)
    initialize_flow!(state, grid, params)

    # Calculate initial heat source
    calculate_heat_source!(state, grid, params)

    # Results storage
    times = Float64[]
    efficiencies = Float64[]
    max_temps = Float64[]
    max_pressures = Float64[]
    avg_exit_temps = Float64[]

    # Time integration
    t = 0.0
    iter = 0
    last_save = 0.0

    println("\nStarting time integration...")
    println("-" ^ 70)

    while t < config.t_final
        iter += 1

        # Calculate timestep
        dt = calculate_cfl_timestep(state, grid, params, config.CFL)

        # Take step
        step_simulation!(state, grid, params, dt)
        calculate_heat_source!(state, grid, params)

        t += dt

        # Save statistics
        if t - last_save >= config.save_interval
            η = calculate_efficiency(state, grid, params)
            T_max = maximum(state.T)
            p_max = maximum(state.p)
            T_exit_avg = mean(state.T[:, end])

            push!(times, t)
            push!(efficiencies, η)
            push!(max_temps, T_max)
            push!(max_pressures, p_max)
            push!(avg_exit_temps, T_exit_avg)

            @printf("t = %.4f s | T_max = %.1f K | T_exit = %.1f K | η = %.1f%% | dt = %.2e s\n",
                    t, T_max, T_exit_avg, η * 100, dt)

            last_save = t
        end
    end

    println("-" ^ 70)
    println("Simulation completed!")
    println()

    # Final statistics
    η_final = efficiencies[end]
    T_max_final = max_temps[end]
    ΔT_exit = avg_exit_temps[end] - params.T_in

    println("FINAL RESULTS:")
    println("  Thermal efficiency: ", @sprintf("%.2f%%", η_final * 100))
    println("  Maximum temperature: ", @sprintf("%.1f K", T_max_final))
    println("  Exit temperature rise: ", @sprintf("%.1f K", ΔT_exit))
    println("  Total iterations: ", iter)
    println()

    # Package results
    results = SimulationResults(
        times, efficiencies, max_temps, max_pressures, avg_exit_temps,
        grid.r, grid.z,
        copy(state.ρ), copy(state.T), copy(state.p),
        copy(state.u_z), copy(state.u_r), copy(state.q_v)
    )

    return results
end

"""
    save_results(results, filename)

Save simulation results to JSON file
"""
function save_results(results::SimulationResults, filename::String)
    data = Dict(
        "times" => results.times,
        "efficiencies" => results.efficiencies,
        "max_temperatures" => results.max_temperatures,
        "max_pressures" => results.max_pressures,
        "avg_exit_temp" => results.avg_exit_temp,
        "grid" => Dict(
            "r" => results.r,
            "z" => results.z
        ),
        "final_state" => Dict(
            "density" => results.ρ_final,
            "temperature" => results.T_final,
            "pressure" => results.p_final,
            "velocity_z" => results.u_z_final,
            "velocity_r" => results.u_r_final,
            "heat_source" => results.q_v_final
        )
    )

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end

    println("Results saved to: $filename")
end

"""
    results_to_dict(results)

Convert results to dictionary for API response
"""
function results_to_dict(results::SimulationResults)
    return Dict(
        "times" => results.times,
        "efficiencies" => results.efficiencies,
        "max_temperatures" => results.max_temperatures,
        "max_pressures" => results.max_pressures,
        "avg_exit_temp" => results.avg_exit_temp,
        "grid" => Dict(
            "r" => results.r,
            "z" => results.z,
            "Nr" => length(results.r),
            "Nz" => length(results.z)
        ),
        "final_state" => Dict(
            "density" => results.ρ_final,
            "temperature" => results.T_final,
            "pressure" => results.p_final,
            "velocity_z" => results.u_z_final,
            "velocity_r" => results.u_r_final,
            "heat_source" => results.q_v_final
        )
    )
end

export SimulationConfig, SimulationResults
export create_default_params, run_simulation, save_results, results_to_dict

end # module
