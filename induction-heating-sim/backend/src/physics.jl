"""
Physics Module: Electromagnetic Induction Heating
Contains all physical equations for Joule heating by eddy currents
"""
module Physics

using LinearAlgebra

# Physical constants
const μ₀ = 4π * 1e-7  # Magnetic permeability [H/m]
const π² = π^2

"""
    skin_depth(ω, μ, σ)

Calculate electromagnetic skin depth (penetration depth)
δ = √(2/(ωμσ))
"""
function skin_depth(ω::Float64, μ::Float64, σ::Float64)::Float64
    return sqrt(2.0 / (ω * μ * σ))
end

"""
    magnetic_field_coefficient(N, R_b, z)

Calculate geometric coefficient K_B for magnetic field from coil
K_B = μ₀NR_b²/[2(R_b² + z²)^(3/2)]
"""
function magnetic_field_coefficient(N::Int, R_b::Float64, z::Float64)::Float64
    denominator = 2.0 * (R_b^2 + z^2)^1.5
    return (μ₀ * N * R_b^2) / denominator
end

"""
    electric_field_rms(r, z, N, R_b, ω, I₀)

Calculate RMS electric field induced by AC coil
E_rms(r) = [μ₀NR_b²ωI₀/(4√2(R_b² + z²)^(3/2))] * r
"""
function electric_field_rms(r::Float64, z::Float64, N::Int, R_b::Float64,
                           ω::Float64, I₀::Float64)::Float64
    K_B = magnetic_field_coefficient(N, R_b, z)
    C₀ = (K_B * ω * I₀) / (4.0 * sqrt(2.0))
    return C₀ * r
end

"""
    volumetric_power_density(r, z, σ, N, R_b, ω, I₀, R_out, μ)

Calculate volumetric Joule heating power density
p_vol = σE_rms² * exp(-2z_skin/δ)
"""
function volumetric_power_density(r::Float64, z::Float64, σ::Float64,
                                 N::Int, R_b::Float64, ω::Float64, I₀::Float64,
                                 R_out::Float64, μ::Float64)::Float64
    # Electric field RMS
    E_rms = electric_field_rms(r, z, N, R_b, ω, I₀)

    # Skin depth
    δ = skin_depth(ω, μ, σ)

    # Distance from wall (skin effect)
    z_skin = max(0.0, R_out - r)

    # Volumetric power with exponential decay
    p_vol = σ * E_rms^2 * exp(-2.0 * z_skin / δ)

    # Limit to physically reasonable values
    return clamp(p_vol, 0.0, 1e9)
end

"""
    power_per_unit_length(a, σ, δ, N, R_b, z, ω, I₀)

Calculate total power dissipated per unit length
P' = πaσδ[C₀a]²
"""
function power_per_unit_length(a::Float64, σ::Float64, δ::Float64,
                              N::Int, R_b::Float64, z::Float64,
                              ω::Float64, I₀::Float64)::Float64
    K_B = magnetic_field_coefficient(N, R_b, z)
    C₀ = (K_B * ω * I₀) / (4.0 * sqrt(2.0))

    return π * a * σ * δ * (C₀ * a)^2
end

"""
    temperature_increase(P_prime, ρ, v, c_p, a)

Calculate fluid temperature increase
ΔT = P'/(ρπa²vc_p)
"""
function temperature_increase(P_prime::Float64, ρ::Float64, v::Float64,
                             c_p::Float64, a::Float64)::Float64
    mass_flow = ρ * π * a^2 * v
    return P_prime / (mass_flow * c_p)
end

"""
    thermal_efficiency(P_fluid, P_coil, P_pre)

Calculate system thermal efficiency
η = P_fluid / (P_fluid + P_coil + P_pre)
"""
function thermal_efficiency(P_fluid::Float64, P_coil::Float64,
                           P_pre::Float64=0.0)::Float64
    P_total = P_fluid + P_coil + P_pre
    return P_total > 0.0 ? P_fluid / P_total : 0.0
end

"""
    coil_losses(I₀, R_coil)

Calculate Joule losses in the coil
P_coil = I_rms² * R_coil = (I₀/√2)² * R_coil
"""
function coil_losses(I₀::Float64, R_coil::Float64)::Float64
    I_rms = I₀ / sqrt(2.0)
    return I_rms^2 * R_coil
end

"""
    speed_of_sound(γ, R, T)

Calculate speed of sound in gas
a = √(γRT)
"""
function speed_of_sound(γ::Float64, R::Float64, T::Float64)::Float64
    return sqrt(γ * R * T)
end

"""
    gas_pressure(ρ, R, T)

Calculate pressure from ideal gas law
p = ρRT
"""
function gas_pressure(ρ::Float64, R::Float64, T::Float64)::Float64
    return ρ * R * T
end

"""
    internal_energy(T, c_v)

Calculate specific internal energy
e = c_v * T
"""
function internal_energy(T::Float64, c_v::Float64)::Float64
    return c_v * T
end

"""
    total_energy_density(ρ, T, u_r, u_z, c_v)

Calculate total energy density
E_v = ρ(c_v*T + 0.5(u_r² + u_z²))
"""
function total_energy_density(ρ::Float64, T::Float64, u_r::Float64,
                             u_z::Float64, c_v::Float64)::Float64
    kinetic = 0.5 * (u_r^2 + u_z^2)
    return ρ * (c_v * T + kinetic)
end

"""
Physical parameters structure for convenience
"""
struct PhysicalParams
    # Geometry
    R_in::Float64      # Inner radius [m]
    R_out::Float64     # Outer radius [m]
    L_z::Float64       # Axial length [m]

    # Coil parameters
    N_coil::Int        # Number of turns
    I₀::Float64        # Peak current [A]
    freq::Float64      # Frequency [Hz]
    R_coil::Float64    # Coil resistance [Ω]
    R_b::Float64       # Coil radius [m]
    z_coil::Float64    # Coil axial position [m]

    # Fluid properties
    σ_fluid::Float64   # Electrical conductivity [S/m]
    c_p::Float64       # Specific heat at const pressure [J/(kg·K)]
    c_v::Float64       # Specific heat at const volume [J/(kg·K)]
    γ::Float64         # Heat capacity ratio
    R_gas::Float64     # Gas constant [J/(kg·K)]
    μ::Float64         # Magnetic permeability [H/m]

    # Inlet conditions
    p_in::Float64      # Inlet pressure [Pa]
    T_in::Float64      # Inlet temperature [K]
    u_in::Float64      # Inlet velocity [m/s]

    # Derived
    ω::Float64         # Angular frequency [rad/s]

    function PhysicalParams(;R_in, R_out, L_z, N_coil, I₀, freq, R_coil, R_b, z_coil,
                           σ_fluid, c_p, c_v, γ, R_gas, μ, p_in, T_in, u_in)
        ω = 2π * freq
        new(R_in, R_out, L_z, N_coil, I₀, freq, R_coil, R_b, z_coil,
            σ_fluid, c_p, c_v, γ, R_gas, μ, p_in, T_in, u_in, ω)
    end
end

export skin_depth, magnetic_field_coefficient, electric_field_rms
export volumetric_power_density, power_per_unit_length, temperature_increase
export thermal_efficiency, coil_losses
export speed_of_sound, gas_pressure, internal_energy, total_energy_density
export PhysicalParams, μ₀

end # module
