"""
Solver Module with CUDA Support and Non-uniform Mesh
Implements 2D axisymmetric finite difference with GPU acceleration
"""
module SolverCUDA

using ..Physics
using ..Mesh
using LinearAlgebra
using Statistics

# Try to load CUDA, fall back to CPU if not available
const USE_CUDA = Ref(false)
const ArrayType = Ref{Type}(Array)

function __init__()
    try
        @eval using CUDA
        if CUDA.functional()
            USE_CUDA[] = true
            ArrayType[] = CuArray
            println("✓ CUDA detected and functional - GPU acceleration enabled")
        else
            println("⚠ CUDA.jl loaded but GPU not functional - using CPU")
        end
    catch e
        println("ℹ CUDA.jl not available - using CPU only")
    end
end

"""
    to_device(x)

Transfer array to GPU if CUDA is available
"""
function to_device(x::AbstractArray)
    if USE_CUDA[]
        return ArrayType[](x)
    else
        return x
    end
end

"""
    to_host(x)

Transfer array back to CPU
"""
function to_host(x::AbstractArray)
    if USE_CUDA[] && x isa CuArray
        return Array(x)
    else
        return x
    end
end

"""
FluidState: Contains all fluid variables at mesh points
Supports both CPU and GPU arrays
"""
mutable struct FluidState{T<:AbstractArray}
    ρ::T    # Density [kg/m³]
    m_r::T  # Radial momentum [kg/(m²·s)]
    m_z::T  # Axial momentum [kg/(m²·s)]
    E_v::T  # Total energy density [J/m³]
    q_v::T  # Volumetric heat source [W/m³]

    # Primitive variables
    u_r::T  # Radial velocity [m/s]
    u_z::T  # Axial velocity [m/s]
    p::T    # Pressure [Pa]
    T::T    # Temperature [K]
end

function FluidState(grid::Mesh.Grid; use_gpu::Bool=USE_CUDA[])
    AType = use_gpu ? ArrayType[] : Array

    FluidState(
        AType(zeros(grid.Nr, grid.Nz)),  # ρ
        AType(zeros(grid.Nr, grid.Nz)),  # m_r
        AType(zeros(grid.Nr, grid.Nz)),  # m_z
        AType(zeros(grid.Nr, grid.Nz)),  # E_v
        AType(zeros(grid.Nr, grid.Nz)),  # q_v
        AType(zeros(grid.Nr, grid.Nz)),  # u_r
        AType(zeros(grid.Nr, grid.Nz)),  # u_z
        AType(zeros(grid.Nr, grid.Nz)),  # p
        AType(zeros(grid.Nr, grid.Nz))   # T
    )
end

"""
    initialize_flow!(state, grid, params)

Initialize flow field with uniform inlet conditions
"""
function initialize_flow!(state::FluidState, grid::Mesh.Grid, params::PhysicalParams)
    ρ_init = params.p_in / (params.R_gas * params.T_in)

    # Initialize on CPU
    ρ_cpu = zeros(grid.Nr, grid.Nz)
    u_z_cpu = zeros(grid.Nr, grid.Nz)
    u_r_cpu = zeros(grid.Nr, grid.Nz)
    p_cpu = zeros(grid.Nr, grid.Nz)
    T_cpu = zeros(grid.Nr, grid.Nz)
    m_z_cpu = zeros(grid.Nr, grid.Nz)
    m_r_cpu = zeros(grid.Nr, grid.Nz)
    E_v_cpu = zeros(grid.Nr, grid.Nz)

    for j in 1:grid.Nz, i in 1:grid.Nr
        ρ_cpu[i, j] = ρ_init
        u_z_cpu[i, j] = params.u_in
        u_r_cpu[i, j] = 0.0
        p_cpu[i, j] = params.p_in
        T_cpu[i, j] = params.T_in

        # Conservative variables
        m_z_cpu[i, j] = ρ_cpu[i, j] * u_z_cpu[i, j]
        m_r_cpu[i, j] = 0.0
        E_v_cpu[i, j] = total_energy_density(ρ_cpu[i, j], T_cpu[i, j],
                                             u_r_cpu[i, j], u_z_cpu[i, j],
                                             params.c_v)
    end

    # Transfer to device
    state.ρ .= to_device(ρ_cpu)
    state.u_z .= to_device(u_z_cpu)
    state.u_r .= to_device(u_r_cpu)
    state.p .= to_device(p_cpu)
    state.T .= to_device(T_cpu)
    state.m_z .= to_device(m_z_cpu)
    state.m_r .= to_device(m_r_cpu)
    state.E_v .= to_device(E_v_cpu)
    state.q_v .= to_device(zeros(grid.Nr, grid.Nz))
end

"""
    calculate_heat_source!(state, grid, params)

Calculate volumetric heat source from induction heating
Optimized for both CPU and GPU
"""
function calculate_heat_source!(state::FluidState, grid::Mesh.Grid, params::PhysicalParams)
    # Transfer grid data to device
    r_dev = to_device(grid.r)
    z_dev = to_device(grid.z)

    # Kernel-like operation (works on both CPU and GPU)
    for j in 1:grid.Nz
        for i in 1:grid.Nr
            r_local = grid.r[i]
            z_local = grid.z[j]

            state.q_v[i, j] = volumetric_power_density(
                r_local, z_local, params.σ_fluid,
                params.N_coil, params.R_b, params.ω, params.I₀,
                params.R_out, params.μ
            )
        end
    end
end

"""
    update_primitives!(state, params)

Update primitive variables from conservative ones
GPU-compatible
"""
function update_primitives!(state::FluidState, params::PhysicalParams)
    Nr, Nz = size(state.ρ)

    # Broadcast operations work on both CPU and GPU
    # Ensure minimum density
    state.ρ .= max.(state.ρ, 1e-6)

    # Velocities
    state.u_r .= state.m_r ./ state.ρ
    state.u_z .= state.m_z ./ state.ρ

    # Internal energy
    e_kinetic = 0.5 .* (state.u_r.^2 .+ state.u_z.^2)
    e_internal = state.E_v ./ state.ρ .- e_kinetic

    # Temperature
    e_internal .= max.(e_internal, 1000.0)
    state.T .= e_internal ./ params.c_v

    # Pressure (ideal gas law)
    state.p .= state.ρ .* params.R_gas .* state.T

    # Physical limits
    state.T .= clamp.(state.T, 50.0, 5000.0)
    state.p .= max.(state.p, 1000.0)
end

"""
    apply_boundary_conditions!(state, grid, params)

Apply boundary conditions
Works on both CPU and GPU
"""
function apply_boundary_conditions!(state::FluidState, grid::Mesh.Grid, params::PhysicalParams)
    Nr, Nz = grid.Nr, grid.Nz

    # Inlet boundary (j = 1)
    ρ_in = params.p_in / (params.R_gas * params.T_in)
    for i in 1:Nr
        state.T[i, 1] = params.T_in
        state.p[i, 1] = params.p_in
        state.ρ[i, 1] = ρ_in
        state.u_z[i, 1] = params.u_in
        state.u_r[i, 1] = 0.0

        # Update conservative
        state.m_z[i, 1] = state.ρ[i, 1] * state.u_z[i, 1]
        state.m_r[i, 1] = 0.0
        state.E_v[i, 1] = total_energy_density(state.ρ[i, 1], state.T[i, 1],
                                               state.u_r[i, 1], state.u_z[i, 1],
                                               params.c_v)
    end

    # Outlet boundary (j = Nz) - extrapolation
    p_out = 0.5 * params.p_in
    for i in 1:Nr
        state.p[i, Nz] = p_out
        state.ρ[i, Nz] = state.ρ[i, Nz-1]
        state.u_z[i, Nz] = state.u_z[i, Nz-1]
        state.u_r[i, Nz] = state.u_r[i, Nz-1]
        state.T[i, Nz] = state.p[i, Nz] / (state.ρ[i, Nz] * params.R_gas)

        state.m_z[i, Nz] = state.ρ[i, Nz] * state.u_z[i, Nz]
        state.m_r[i, Nz] = state.ρ[i, Nz] * state.u_r[i, Nz]
        state.E_v[i, Nz] = total_energy_density(state.ρ[i, Nz], state.T[i, Nz],
                                                state.u_r[i, Nz], state.u_z[i, Nz],
                                                params.c_v)
    end

    # Axis boundary (i = 1) - symmetry
    for j in 1:Nz
        state.u_r[1, j] = 0.0
        state.m_r[1, j] = 0.0
        state.ρ[1, j] = state.ρ[2, j]
        state.u_z[1, j] = state.u_z[2, j]
        state.p[1, j] = state.p[2, j]
        state.T[1, j] = state.T[2, j]

        state.m_z[1, j] = state.ρ[1, j] * state.u_z[1, j]
        state.E_v[1, j] = total_energy_density(state.ρ[1, j], state.T[1, j],
                                               state.u_r[1, j], state.u_z[1, j],
                                               params.c_v)
    end

    # Wall boundary (i = Nr) - slip wall
    for j in 1:Nz
        state.u_r[Nr, j] = 0.0
        state.m_r[Nr, j] = 0.0
        state.u_z[Nr, j] = state.u_z[Nr-1, j]
        state.p[Nr, j] = state.p[Nr-1, j]
        state.T[Nr, j] = state.T[Nr-1, j]
        state.ρ[Nr, j] = state.p[Nr, j] / (params.R_gas * state.T[Nr, j])

        state.m_z[Nr, j] = state.ρ[Nr, j] * state.u_z[Nr, j]
        state.E_v[Nr, j] = total_energy_density(state.ρ[Nr, j], state.T[Nr, j],
                                                state.u_r[Nr, j], state.u_z[Nr, j],
                                                params.c_v)
    end
end

"""
    calculate_cfl_timestep(state, grid, params, CFL)

Calculate timestep based on CFL condition
Handles non-uniform grids
"""
function calculate_cfl_timestep(state::FluidState, grid::Mesh.Grid,
                               params::PhysicalParams, CFL::Float64)::Float64
    dt_min = 1e-3

    # Transfer to CPU for reduction (faster than GPU for small operations)
    ρ_cpu = to_host(state.ρ)
    u_r_cpu = to_host(state.u_r)
    u_z_cpu = to_host(state.u_z)
    T_cpu = to_host(state.T)

    for j in 2:grid.Nz-1
        for i in 2:grid.Nr-1
            a_sound = speed_of_sound(params.γ, params.R_gas, T_cpu[i, j])

            λ_r = abs(u_r_cpu[i, j]) + a_sound
            λ_z = abs(u_z_cpu[i, j]) + a_sound

            # Use local cell sizes for non-uniform grid
            dt_local = CFL * min(grid.dr[i] / λ_r, grid.dz[j] / λ_z)
            dt_min = min(dt_min, dt_local)
        end
    end

    return max(dt_min, 1e-8)
end

"""
    euler_step!(state, state_new, grid, params, dt)

Single Euler step for conservative variables
Handles non-uniform grid spacing
"""
function euler_step!(state::FluidState, state_new::FluidState, grid::Mesh.Grid,
                    params::PhysicalParams, dt::Float64)
    Nr, Nz = grid.Nr, grid.Nz

    # Interior points only
    for j in 2:Nz-1
        for i in 2:Nr-1
            # Central differences with non-uniform spacing
            dr_avg = 0.5 * (grid.dr[i] + grid.dr[i-1])
            dz_avg = 0.5 * (grid.dz[j] + grid.dz[j-1])

            # Continuity
            flux_r = (state.m_r[i+1, j] - state.m_r[i-1, j]) / (2 * dr_avg)
            flux_z = (state.m_z[i, j+1] - state.m_z[i, j-1]) / (2 * dz_avg)
            state_new.ρ[i, j] = state.ρ[i, j] - dt * (flux_r + flux_z)

            # Momentum
            dp_dr = (state.p[i+1, j] - state.p[i-1, j]) / (2 * dr_avg)
            dp_dz = (state.p[i, j+1] - state.p[i, j-1]) / (2 * dz_avg)

            state_new.m_r[i, j] = state.m_r[i, j] - dt * dp_dr
            state_new.m_z[i, j] = state.m_z[i, j] - dt * dp_dz

            # Energy with heat source
            state_new.E_v[i, j] = state.E_v[i, j] + dt * state.q_v[i, j]
        end
    end
end

"""
    step_simulation!(state, grid, params, dt)

Advance simulation by one timestep using RK2
"""
function step_simulation!(state::FluidState, grid::Mesh.Grid,
                         params::PhysicalParams, dt::Float64)
    state_temp = FluidState(grid, use_gpu=USE_CUDA[])

    # Predictor step
    euler_step!(state, state_temp, grid, params, dt/2)
    apply_boundary_conditions!(state_temp, grid, params)
    update_primitives!(state_temp, params)

    # Corrector step
    euler_step!(state_temp, state, grid, params, dt)
    apply_boundary_conditions!(state, grid, params)
    update_primitives!(state, params)
end

"""
    calculate_efficiency(state, grid, params)

Calculate thermal efficiency of the system
"""
function calculate_efficiency(state::FluidState, grid::Mesh.Grid,
                             params::PhysicalParams)::Float64
    P_fluid = 0.0

    # Transfer to CPU for integration
    q_v_cpu = to_host(state.q_v)

    for j in 1:grid.Nz
        for i in 1:grid.Nr
            # Volume of cylindrical cell with non-uniform spacing
            vol_cell = grid.r[i] * grid.dr[i] * grid.dz[j] * 2π
            P_fluid += q_v_cpu[i, j] * vol_cell
        end
    end

    P_coil = coil_losses(params.I₀, params.R_coil)

    return thermal_efficiency(P_fluid, P_coil)
end

export FluidState, USE_CUDA, to_device, to_host
export initialize_flow!, calculate_heat_source!
export update_primitives!, apply_boundary_conditions!
export calculate_cfl_timestep, step_simulation!, calculate_efficiency

end # module
