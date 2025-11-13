"""
Solver Module: 2D Axisymmetric Fluid Dynamics with Heat Source
Implements finite difference method for compressible inviscid flow
"""
module Solver

using ..Physics
using LinearAlgebra
using Statistics

"""
Grid structure for 2D axisymmetric mesh
"""
struct Grid
    Nr::Int            # Number of radial cells
    Nz::Int            # Number of axial cells
    r::Vector{Float64} # Radial positions [m]
    z::Vector{Float64} # Axial positions [m]
    dr::Float64        # Radial spacing [m]
    dz::Float64        # Axial spacing [m]
end

function Grid(R_in::Float64, R_out::Float64, L_z::Float64, Nr::Int, Nz::Int)
    dr = (R_out - R_in) / (Nr - 1)
    dz = L_z / (Nz - 1)

    # Cell centers
    r = [R_in + (i - 0.5) * dr for i in 1:Nr]
    z = [(j - 0.5) * dz for j in 1:Nz]

    Grid(Nr, Nz, r, z, dr, dz)
end

"""
FluidState: Contains all fluid variables at mesh points
"""
mutable struct FluidState
    ρ::Matrix{Float64}    # Density [kg/m³]
    m_r::Matrix{Float64}  # Radial momentum [kg/(m²·s)]
    m_z::Matrix{Float64}  # Axial momentum [kg/(m²·s)]
    E_v::Matrix{Float64}  # Total energy density [J/m³]
    q_v::Matrix{Float64}  # Volumetric heat source [W/m³]

    # Primitive variables
    u_r::Matrix{Float64}  # Radial velocity [m/s]
    u_z::Matrix{Float64}  # Axial velocity [m/s]
    p::Matrix{Float64}    # Pressure [Pa]
    T::Matrix{Float64}    # Temperature [K]
end

function FluidState(grid::Grid)
    FluidState(
        zeros(grid.Nr, grid.Nz),  # ρ
        zeros(grid.Nr, grid.Nz),  # m_r
        zeros(grid.Nr, grid.Nz),  # m_z
        zeros(grid.Nr, grid.Nz),  # E_v
        zeros(grid.Nr, grid.Nz),  # q_v
        zeros(grid.Nr, grid.Nz),  # u_r
        zeros(grid.Nr, grid.Nz),  # u_z
        zeros(grid.Nr, grid.Nz),  # p
        zeros(grid.Nr, grid.Nz)   # T
    )
end

"""
    initialize_flow!(state, grid, params)

Initialize flow field with uniform inlet conditions
"""
function initialize_flow!(state::FluidState, grid::Grid, params::PhysicalParams)
    ρ_init = params.p_in / (params.R_gas * params.T_in)

    for j in 1:grid.Nz, i in 1:grid.Nr
        state.ρ[i, j] = ρ_init
        state.u_z[i, j] = params.u_in
        state.u_r[i, j] = 0.0
        state.p[i, j] = params.p_in
        state.T[i, j] = params.T_in

        # Conservative variables
        state.m_z[i, j] = state.ρ[i, j] * state.u_z[i, j]
        state.m_r[i, j] = 0.0
        state.E_v[i, j] = total_energy_density(state.ρ[i, j], state.T[i, j],
                                               state.u_r[i, j], state.u_z[i, j],
                                               params.c_v)
        state.q_v[i, j] = 0.0
    end
end

"""
    calculate_heat_source!(state, grid, params)

Calculate volumetric heat source from induction heating
"""
function calculate_heat_source!(state::FluidState, grid::Grid, params::PhysicalParams)
    for j in 1:grid.Nz, i in 1:grid.Nr
        r_local = grid.r[i]
        z_local = grid.z[j]

        state.q_v[i, j] = volumetric_power_density(
            r_local, z_local, params.σ_fluid,
            params.N_coil, params.R_b, params.ω, params.I₀,
            params.R_out, params.μ
        )
    end
end

"""
    update_primitives!(state, params)

Update primitive variables from conservative ones
"""
function update_primitives!(state::FluidState, params::PhysicalParams)
    Nr, Nz = size(state.ρ)

    for j in 1:Nz, i in 1:Nr
        # Avoid division by zero
        if state.ρ[i, j] < 1e-6
            state.ρ[i, j] = 1e-6
        end

        # Velocities
        state.u_r[i, j] = state.m_r[i, j] / state.ρ[i, j]
        state.u_z[i, j] = state.m_z[i, j] / state.ρ[i, j]

        # Internal energy
        e_kinetic = 0.5 * (state.u_r[i, j]^2 + state.u_z[i, j]^2)
        e_internal = state.E_v[i, j] / state.ρ[i, j] - e_kinetic

        # Temperature
        if e_internal < 0.0
            e_internal = 1000.0  # Minimum energy
        end
        state.T[i, j] = e_internal / params.c_v

        # Pressure (ideal gas law)
        state.p[i, j] = gas_pressure(state.ρ[i, j], params.R_gas, state.T[i, j])

        # Physical limits
        state.T[i, j] = clamp(state.T[i, j], 50.0, 5000.0)
        state.p[i, j] = max(state.p[i, j], 1000.0)
    end
end

"""
    apply_boundary_conditions!(state, grid, params)

Apply boundary conditions: inlet, outlet, axis, wall
"""
function apply_boundary_conditions!(state::FluidState, grid::Grid, params::PhysicalParams)
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
    p_out = 0.5 * params.p_in  # Expansion
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
"""
function calculate_cfl_timestep(state::FluidState, grid::Grid,
                               params::PhysicalParams, CFL::Float64)::Float64
    dt_min = 1e-3

    for j in 2:grid.Nz-1, i in 2:grid.Nr-1
        a_sound = speed_of_sound(params.γ, params.R_gas, state.T[i, j])

        λ_r = abs(state.u_r[i, j]) + a_sound
        λ_z = abs(state.u_z[i, j]) + a_sound

        dt_local = CFL * min(grid.dr / λ_r, grid.dz / λ_z)
        dt_min = min(dt_min, dt_local)
    end

    return max(dt_min, 1e-8)
end

"""
    euler_step!(state, state_new, grid, params, dt)

Single Euler step for conservative variables with operator splitting
"""
function euler_step!(state::FluidState, state_new::FluidState, grid::Grid,
                    params::PhysicalParams, dt::Float64)
    Nr, Nz = grid.Nr, grid.Nz

    # Interior points only
    for j in 2:Nz-1, i in 2:Nr-1
        # Simple upwind scheme for demonstration
        # (In production, use MUSCL or WENO)

        # Continuity
        flux_r = (state.m_r[i+1, j] - state.m_r[i-1, j]) / (2 * grid.dr)
        flux_z = (state.m_z[i, j+1] - state.m_z[i, j-1]) / (2 * grid.dz)
        state_new.ρ[i, j] = state.ρ[i, j] - dt * (flux_r + flux_z)

        # Momentum (simplified - no convective terms for stability)
        dp_dr = (state.p[i+1, j] - state.p[i-1, j]) / (2 * grid.dr)
        dp_dz = (state.p[i, j+1] - state.p[i, j-1]) / (2 * grid.dz)

        state_new.m_r[i, j] = state.m_r[i, j] - dt * dp_dr
        state_new.m_z[i, j] = state.m_z[i, j] - dt * dp_dz

        # Energy with heat source
        state_new.E_v[i, j] = state.E_v[i, j] + dt * state.q_v[i, j]
    end
end

"""
    step_simulation!(state, grid, params, dt)

Advance simulation by one timestep using RK2
"""
function step_simulation!(state::FluidState, grid::Grid,
                         params::PhysicalParams, dt::Float64)
    state_temp = FluidState(grid)

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
function calculate_efficiency(state::FluidState, grid::Grid,
                             params::PhysicalParams)::Float64
    P_fluid = 0.0

    for j in 1:grid.Nz, i in 1:grid.Nr
        vol_cell = grid.r[i] * grid.dr * grid.dz * 2π
        P_fluid += state.q_v[i, j] * vol_cell
    end

    P_coil = coil_losses(params.I₀, params.R_coil)

    return thermal_efficiency(P_fluid, P_coil)
end

export Grid, FluidState
export initialize_flow!, calculate_heat_source!
export update_primitives!, apply_boundary_conditions!
export calculate_cfl_timestep, step_simulation!, calculate_efficiency

end # module
