"""
Main Simulation Module: Electromagnetic Induction Heating (v2)
With non-uniform mesh and CUDA support
"""
module SimulationV2

include("physics.jl")
include("mesh.jl")
include("solver_cuda.jl")

using .Physics
using .Mesh
using .SolverCUDA
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

    # Mesh refinement
    radial_stretching::Float64   # Clustering near outer wall
    axial_stretching::Float64    # Clustering near outlet

    # GPU settings
    use_gpu::Bool

    # Output
    output_dir::String
end

function SimulationConfig(;
    t_final=5.0,
    CFL=0.5,
    save_interval=0.1,
    Nr=30,
    Nz=100,
    radial_stretching=2.0,
    axial_stretching=1.0,
    use_gpu=SolverCUDA.USE_CUDA[],
    output_dir="./data"
)
    SimulationConfig(t_final, CFL, save_interval, Nr, Nz,
                    radial_stretching, axial_stretching, use_gpu, output_dir)
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
    dr::Vector{Float64}
    dz::Vector{Float64}
    ρ_final::Matrix{Float64}
    T_final::Matrix{Float64}
    p_final::Matrix{Float64}
    u_z_final::Matrix{Float64}
    u_r_final::Matrix{Float64}
    q_v_final::Matrix{Float64}

    # Grid info
    grid_info::String
    use_gpu::Bool
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

Run complete simulation with refined mesh and optional GPU acceleration
"""
function run_simulation(params::PhysicalParams, config::SimulationConfig)
    println("=" ^ 70)
    println("ELECTROMAGNETIC INDUCTION HEATING SIMULATION v2")
    println("=" ^ 70)
    println()

    # Display configuration
    println("Configuration:")
    println("  Mesh type: Non-uniform (wall-refined)")
    println("  Radial clustering: ", config.radial_stretching)
    println("  Axial clustering: ", config.axial_stretching)
    println("  GPU acceleration: ", config.use_gpu ? "ENABLED ✓" : "DISABLED (CPU)")
    println()

    # Initialize grid with refinement
    @printf("Creating refined grid: %d x %d cells\n", config.Nr, config.Nz)
    grid = create_stretched_grid(
        params.R_in, params.R_out, params.L_z,
        config.Nr, config.Nz,
        radial_stretching = config.radial_stretching,
        axial_stretching = config.axial_stretching
    )

    print_grid_info(grid)
    println()

    # Initialize flow state
    println("Initializing flow field...")
    state = SolverCUDA.FluidState(grid, use_gpu=config.use_gpu)
    SolverCUDA.initialize_flow!(state, grid, params)

    # Calculate initial heat source
    SolverCUDA.calculate_heat_source!(state, grid, params)

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

    # Warm-up GPU if enabled
    if config.use_gpu
        println("Warming up GPU...")
        for _ in 1:5
            dt = SolverCUDA.calculate_cfl_timestep(state, grid, params, config.CFL)
            SolverCUDA.step_simulation!(state, grid, params, dt)
        end
        println("✓ GPU warm-up complete")
        println()
    end

    # Reset for actual simulation
    SolverCUDA.initialize_flow!(state, grid, params)
    SolverCUDA.calculate_heat_source!(state, grid, params)
    t = 0.0
    iter = 0

    # Main time loop
    start_time = time()

    while t < config.t_final
        iter += 1

        # Calculate timestep
        dt = SolverCUDA.calculate_cfl_timestep(state, grid, params, config.CFL)

        # Take step
        SolverCUDA.step_simulation!(state, grid, params, dt)
        SolverCUDA.calculate_heat_source!(state, grid, params)

        t += dt

        # Save statistics
        if t - last_save >= config.save_interval
            η = SolverCUDA.calculate_efficiency(state, grid, params)

            # Transfer to CPU for statistics
            T_cpu = SolverCUDA.to_host(state.T)
            p_cpu = SolverCUDA.to_host(state.p)

            T_max = maximum(T_cpu)
            p_max = maximum(p_cpu)
            T_exit_avg = sum(T_cpu[:, end]) / grid.Nr

            push!(times, t)
            push!(efficiencies, η)
            push!(max_temps, T_max)
            push!(max_pressures, p_max)
            push!(avg_exit_temps, T_exit_avg)

            elapsed = time() - start_time
            iter_per_sec = iter / elapsed

            @printf("t = %.4f s | T_max = %.1f K | T_exit = %.1f K | η = %.1f%% | %.1f iter/s\n",
                    t, T_max, T_exit_avg, η * 100, iter_per_sec)

            last_save = t
        end
    end

    elapsed_total = time() - start_time

    println("-" ^ 70)
    println("Simulation completed!")
    println()

    # Performance metrics
    println("PERFORMANCE:")
    println("  Total time: ", @sprintf("%.2f s", elapsed_total))
    println("  Total iterations: ", iter)
    println("  Average iterations/sec: ", @sprintf("%.1f", iter / elapsed_total))
    println("  Time per iteration: ", @sprintf("%.2f ms", 1000 * elapsed_total / iter))
    println()

    # Final statistics
    η_final = efficiencies[end]
    T_max_final = max_temps[end]
    ΔT_exit = avg_exit_temps[end] - params.T_in

    println("FINAL RESULTS:")
    println("  Thermal efficiency: ", @sprintf("%.2f%%", η_final * 100))
    println("  Maximum temperature: ", @sprintf("%.1f K", T_max_final))
    println("  Exit temperature rise: ", @sprintf("%.1f K", ΔT_exit))
    println()

    # Transfer final state to CPU
    ρ_final = SolverCUDA.to_host(state.ρ)
    T_final = SolverCUDA.to_host(state.T)
    p_final = SolverCUDA.to_host(state.p)
    u_z_final = SolverCUDA.to_host(state.u_z)
    u_r_final = SolverCUDA.to_host(state.u_r)
    q_v_final = SolverCUDA.to_host(state.q_v)

    # Grid info string
    grid_info = "Refined mesh: $(grid.Nr)x$(grid.Nz), " *
                "dr ∈ [$(grid.dr_min*1000)mm, $(grid.dr_max*1000)mm], " *
                "dz ∈ [$(grid.dz_min*1000)mm, $(grid.dz_max*1000)mm]"

    # Package results
    results = SimulationResults(
        times, efficiencies, max_temps, max_pressures, avg_exit_temps,
        grid.r, grid.z, grid.dr, grid.dz,
        ρ_final, T_final, p_final, u_z_final, u_r_final, q_v_final,
        grid_info, config.use_gpu
    )

    return results
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
            "dr" => results.dr,
            "dz" => results.dz,
            "Nr" => length(results.r),
            "Nz" => length(results.z),
            "info" => results.grid_info
        ),
        "final_state" => Dict(
            "density" => results.ρ_final,
            "temperature" => results.T_final,
            "pressure" => results.p_final,
            "velocity_z" => results.u_z_final,
            "velocity_r" => results.u_r_final,
            "heat_source" => results.q_v_final
        ),
        "gpu_used" => results.use_gpu
    )
end

export SimulationConfig, SimulationResults
export create_default_params, run_simulation, results_to_dict

end # module
