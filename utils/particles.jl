# ---------------------------------------------------------------------------
# ARCHIVO: particles.jl (Inicialización y movimiento de partículas)
# ---------------------------------------------------------------------------

using LinearAlgebra
using Random

# GPU: Importar CUDA y la bandera de uso
using CUDA
# Asumimos que USE_GPU y constantes como ELECTRON_MASS, ELECTRON_CHARGE están definidas.

# ===========================================================================
# KERNELS Y FUNCIONES DE DISPOSITIVO (GPU)
# ===========================================================================

# GPU: Versión "device" de la interpolación del campo eléctrico.
# Cada hilo (partícula) la llama para encontrar el campo en su posición.
@inline function device_interpolate_field(px, py, pz, Ex, Ey, Ez, x_grid, y_grid, z_grid)
    # Asume malla uniforme para un cálculo de índice rápido
    dx, dy, dz = x_grid[2] - x_grid[1], y_grid[2] - y_grid[1], z_grid[2] - z_grid[1]
    
    # Nearest Grid Point (NGP) para simplicidad.
    ix = clamp(floor(Int, px / dx) + 1, 1, size(Ex, 1))
    iy = clamp(floor(Int, py / dy) + 1, 1, size(Ex, 2))
    iz = clamp(floor(Int, pz / dz) + 1, 1, size(Ex, 3))
    
    return Ex[ix, iy, iz], Ey[ix, iy, iz], Ez[ix, iy, iz]
end

# GPU: Kernel para el algoritmo de Boris.
function boris_pusher_kernel!(positions, velocities, dt, Bx, By, Bz,
                              Ex, Ey, Ez, # <-- Desempaquetado
                              x_grid, y_grid, z_grid)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > size(positions, 1); return; end

    # Cargar posición y velocidad de la partícula
    px, py, pz = positions[i, 1], positions[i, 2], positions[i, 3]
    vx, vy, vz = velocities[i, 1], velocities[i, 2], velocities[i, 3]

    # --- CORRECCIÓN 2: Usar los arrays de campo E directamente ---
    # La llamada a device_interpolate_field ahora usa los arrays directamente.
    Ex_local, Ey_local, Ez_local = device_interpolate_field(
        px, py, pz, Ex, Ey, Ez, # <-- Pasa los arrays
        x_grid, y_grid, z_grid
    )


    q_over_m = ELECTRON_CHARGE / ELECTRON_MASS
    dt_half = dt / 2.0

    vx_minus = vx + q_over_m * Ex_local * dt_half
    vy_minus = vy + q_over_m * Ey_local * dt_half
    vz_minus = vz + q_over_m * Ez_local * dt_half

    tx, ty, tz = q_over_m * Bx * dt_half, q_over_m * By * dt_half, q_over_m * Bz * dt_half

    t_mag_sq = tx*tx + ty*ty + tz*tz

    s_factor = 2.0 / (1.0 + t_mag_sq)

    sx, sy, sz = tx * s_factor, ty * s_factor, tz * s_factor

    v_prime_x = vx_minus + (vy_minus * tz - vz_minus * ty)
    v_prime_y = vy_minus + (vz_minus * tx - vx_minus * tz)
    v_prime_z = vz_minus + (vx_minus * ty - vy_minus * tx)

    vx_plus = vx_minus + (v_prime_y * sz - v_prime_z * sy)
    vy_plus = vy_minus + (v_prime_z * sx - v_prime_x * sz)
    vz_plus = vz_minus + (v_prime_x * sy - v_prime_y * sx)

    vx_new = vx_plus + q_over_m * Ex_local * dt_half
    vy_new = vy_plus + q_over_m * Ey_local * dt_half
    vz_new = vz_plus + q_over_m * Ez_local * dt_half

    velocities[i, 1], velocities[i, 2], velocities[i, 3] = vx_new, vy_new, vz_new

    positions[i, 1] += vx_new * dt
    positions[i, 2] += vy_new * dt
    positions[i, 3] += vz_new * dt
    
    return
end

# GPU: Kernel para limitar la energía de las partículas.
function limit_energy_kernel!(velocities, min_v_sq, max_v_sq)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > size(velocities, 1); return; end

    vx, vy, vz = velocities[i, 1], velocities[i, 2], velocities[i, 3]
    v_sq = vx*vx + vy*vy + vz*vz

    if v_sq > max_v_sq
        factor = CUDA.sqrt(max_v_sq / v_sq)
        velocities[i, 1] *= factor
        velocities[i, 2] *= factor
        velocities[i, 3] *= factor
    elseif v_sq < min_v_sq && v_sq > 1e-20
        factor = CUDA.sqrt(min_v_sq / v_sq)
        velocities[i, 1] *= factor
        velocities[i, 2] *= factor
        velocities[i, 3] *= factor
    end
    
    return
end


# ===========================================================================
# FUNCIONES DE ALTO NIVEL (DESPACHADORAS)
# ===========================================================================

function move_electrons(positions, velocities, dt, magnetic_field, E_grid,
                        x_grid, y_grid, z_grid)
    num_particles = size(positions, 1)
    if num_particles == 0; return positions, velocities; end

    if USE_GPU && isa(positions, CuArray)
        # Versión GPU
        threads = 256
        blocks = cld(num_particles, threads)
        
        x_grid_d, y_grid_d, z_grid_d = CuArray(x_grid), CuArray(y_grid), CuArray(z_grid)

        # --- CORRECCIÓN 4: Desempaquetar la struct ANTES de llamar a @cuda ---
        @cuda threads=threads blocks=blocks boris_pusher_kernel!(
            positions, velocities, dt, magnetic_field[1], magnetic_field[2], magnetic_field[3],
            E_grid.Ex, E_grid.Ey, E_grid.Ez, # <-- Pasa los componentes, no la struct
            x_grid_d, y_grid_d, z_grid_d
        )
        CUDA.synchronize()
        return positions, velocities
    else
        # Versión CPU (no necesita cambios, ya que puede manejar la struct)
        return move_electrons_cpu(positions, velocities, dt, magnetic_field, E_grid,
                                  x_grid, y_grid, z_grid)
    end
end

function apply_rectangular_boundary_conditions(positions, velocities, chamber_dims)
    # Esta función ahora devuelve (posiciones_filtradas, velocidades_filtradas, máscara_de_conservados)
    
    # La lógica para crear la máscara es la misma para CPU y GPU
    mask = (positions[:, 1] .>= 0.0) .& (positions[:, 1] .<= chamber_dims.width) .&
           (positions[:, 2] .>= 0.0) .& (positions[:, 2] .<= chamber_dims.length) .&
           (positions[:, 3] .>= 0.0) .& (positions[:, 3] .<= chamber_dims.height)
    
    # Filtra los arrays usando la máscara
    filtered_positions = positions[mask, :]
    filtered_velocities = velocities[mask, :]
    
    # Devuelve los arrays filtrados Y la máscara
    return filtered_positions, filtered_velocities, mask
end

function calculate_inside_chamber_mask(positions, chamber_dims)
    if isa(positions, CuArray)
        # Versión GPU con comprobación de límites
        return (positions[:, 1] .>= 0.0) .& 
               (positions[:, 1] .<= Float32(chamber_dims.width)) .&
               (positions[:, 2] .>= 0.0) .& 
               (positions[:, 2] .<= Float32(chamber_dims.length)) .&
               (positions[:, 3] .>= 0.0) .& 
               (positions[:, 3] .<= Float32(chamber_dims.height))
    else
        # Versión CPU
        return [
            0 <= x <= chamber_dims.width &&
            0 <= y <= chamber_dims.length &&
            0 <= z <= chamber_dims.height
            for (x, y, z) in eachrow(positions)
        ]
    end
end

function limit_electron_energy!(velocities, min_eV=0.1, max_eV=1000.0)
    if isempty(velocities); return velocities; end
    
    min_energy_J = min_eV * abs(ELECTRON_CHARGE)
    max_energy_J = max_eV * abs(ELECTRON_CHARGE)
    min_v_sq = 2 * min_energy_J / ELECTRON_MASS
    max_v_sq = 2 * max_energy_J / ELECTRON_MASS

    if USE_GPU && isa(velocities, CuArray)
        # Versión GPU
        threads = 256
        blocks = cld(size(velocities, 1), threads)
        @cuda threads=threads blocks=blocks limit_energy_kernel!(velocities, min_v_sq, max_v_sq)
    else
        # Versión CPU
        for i in 1:size(velocities, 1)
            v_sq = sum(velocities[i,:].^2)
            if v_sq > max_v_sq
                factor = sqrt(max_v_sq / v_sq)
                velocities[i,:] .*= factor
            elseif v_sq < min_v_sq && v_sq > 0
                factor = sqrt(min_v_sq / v_sq)
                velocities[i,:] .*= factor
            end
        end
    end
    return velocities
end


# ===========================================================================
# IMPLEMENTACIONES ORIGINALES PARA CPU Y FUNCIONES AUXILIARES
# ===========================================================================

function electron_velocity_from_energy(electron_energy_eV)
    energy_joules = electron_energy_eV * abs(ELECTRON_CHARGE)
    return sqrt(2 * max(0.0, energy_joules) / ELECTRON_MASS)
end

function lorentz_force(velocity, magnetic_field, charge)
    return charge .* cross(velocity, magnetic_field)
end

function initialize_electrons(num_electrons, chamber_dims, initial_electron_velocity)
    # La inicialización se hace en la CPU, los datos se mueven a la GPU después si es necesario.
    rng = MersenneTwister(0)
    x_pos = rand(rng, num_electrons) .* chamber_dims.width
    y_pos = rand(rng, num_electrons) .* chamber_dims.length
    z_pos = zeros(num_electrons)

    vx = zeros(num_electrons)
    vy = zeros(num_electrons)
    vz = fill(initial_electron_velocity, num_electrons)

    positions = hcat(x_pos, y_pos, z_pos)
    velocities = hcat(vx, vy, vz)
    return positions, velocities
end

function move_electrons_cpu(positions, velocities, dt, magnetic_field, E_grid,
                            x_grid, y_grid, z_grid)
    num_particles = size(positions, 1)
    new_positions = copy(positions)
    new_velocities = copy(velocities)

    q_over_m = ELECTRON_CHARGE / ELECTRON_MASS
    dt_half = dt / 2.0
    
    t = q_over_m * magnetic_field * dt_half
    t_mag_sq = dot(t, t)
    s = 2.0 * t / (1.0 + t_mag_sq)

    for i in 1:num_particles
        # Interpolar campo E en la CPU
        E_local = interpolate_electric_field(positions[i,:], E_grid, x_grid, y_grid, z_grid)
        
        # Algoritmo de Boris
        v_n = velocities[i, :]
        E_accel_half_dt = q_over_m * E_local * dt_half
        
        v_minus = v_n + E_accel_half_dt
        v_prime = v_minus + cross(v_minus, t)
        v_plus = v_minus + cross(v_prime, s)
        v_n_plus_1 = v_plus + E_accel_half_dt
        
        new_velocities[i, :] = v_n_plus_1
        new_positions[i, :] = positions[i, :] + v_n_plus_1 * dt
    end
    return new_positions, new_velocities
end

function check_timestep_validity(dt, magnetic_field_strength)
    if magnetic_field_strength > 1e-10
        cyclotron_period = 2π * ELECTRON_MASS / (abs(ELECTRON_CHARGE) * magnetic_field_strength)
        if dt > 0.1 * cyclotron_period
            @warn "Timestep $(dt)s puede ser demasiado grande para la frecuencia de ciclotrón. T_c ≈ $(cyclotron_period)s."
            return false
        end
    end
    return true
end