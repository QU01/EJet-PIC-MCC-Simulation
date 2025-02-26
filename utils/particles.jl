using LinearAlgebra
using Random

# --- Funciones Auxiliares (Sin Cambios Mayores) ---
function electron_velocity_from_energy(electron_energy_eV)
    energy_joules = electron_energy_eV * 1.60218e-19
    return sqrt(2 * energy_joules / electron_mass)
end

function electron_energy_from_velocity(velocity_magnitude)
    return 0.5 * electron_mass * (velocity_magnitude ^ 2)
end

function lorentz_force(velocity, magnetic_field, charge)
    v_cross_B = cross(velocity, magnetic_field)
    return charge * v_cross_B
end

# --- Inicialización de Electrones (Sin Cambios) ---
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

# --- Movimiento de Electrones con Fuerza de Lorentz (Corregido) ---
function move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
    new_velocities = copy(velocities)
    B_magnitude = norm(magnetic_field)

    # Verificar validez del timestep
    if B_magnitude > 1e-10
        cyclotron_period = 2π * electron_mass / (abs(electron_charge) * B_magnitude)
        if dt > 0.1 * cyclotron_period
            @warn "Timestep ($(dt)) es demasiado grande comparado con el período de ciclotón ($(cyclotron_period)) para B=$(B_magnitude)T"
        end
    end

    # Si no hay campo magnético significativo
    if B_magnitude < 1e-10
        new_positions = positions + velocities * dt
        return new_positions, new_velocities
    end

    # Vector unitario en dirección del campo magnético
    b_hat = magnetic_field / B_magnitude

    # Ángulo de rotación durante medio paso
    theta = electron_charge * B_magnitude * dt / (2 * electron_mass)

    # Factores para rotación
    cos_2theta = cos(2 * theta)
    sin_2theta = sin(2 * theta)

    # Para cada electrón
    for i in 1:size(velocities, 1)
        v = velocities[i, :]
        v_mag_initial = norm(v)

        # Descomponer velocidad: componentes paralela y perpendicular a B
        v_parallel = dot(v, b_hat) * b_hat
        v_perp = v - v_parallel

        # Rotar solo componente perpendicular
        v_perp_mag = norm(v_perp)

        if v_perp_mag > 1e-10
            # Vectores unitarios perpendiculares en plano de rotación
            e1 = v_perp / v_perp_mag
            e2 = cross(b_hat, e1)

            # Rotación de la componente perpendicular
            v_perp_rotated = v_perp_mag * (cos_2theta * e1 + sin_2theta * e2)

            # Nueva velocidad: componente paralela (sin cambio) + perpendicular rotada
            new_velocities[i, :] = v_parallel + v_perp_rotated

            # Verificar conservación de energía
            v_mag_final = norm(new_velocities[i, :])
            if abs(v_mag_final - v_mag_initial) / v_mag_initial > 1e-10
                # Corregir para garantizar conservación exacta de energía
                new_velocities[i, :] *= (v_mag_initial / v_mag_final)
            end
        end
    end

    # Actualizar posiciones usando promedio de velocidades (método de segundo orden)
    new_positions = positions + 0.5 * (velocities + new_velocities) * dt

    return new_positions, new_velocities
end

function check_timestep_validity(dt, magnetic_field_strength, electron_mass, electron_charge)
    if magnetic_field_strength > 1e-10
        cyclotron_period = 2π * electron_mass / (abs(electron_charge) * magnetic_field_strength)
        dt_max_recommended = 0.1 * cyclotron_period

        if dt > dt_max_recommended
            println("⚠️ ADVERTENCIA DE ESTABILIDAD:")
            println("  Timestep actual: $(dt) segundos")
            println("  Período de ciclotón: $(cyclotron_period) segundos")
            println("  Timestep máximo recomendado: $(dt_max_recommended) segundos")
            println("  La simulación puede ser inestable. Considere reducir dt o implementar un método implícito.")
            return false
        end
    end
    return true
end
# --- Condiciones de Frontera (Sin Cambios) ---
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

    reflect_y_min = y .< 0
    reflect_y_max = y .> chamber_length
    y = ifelse.(reflect_y_min, .-y, ifelse.(reflect_y_max, chamber_length .- (y .- chamber_length), y))
    vy = ifelse.(reflect_y_min .| reflect_y_max, .-vy, vy)

    reflect_bottom = z .< 0
    z = ifelse.(reflect_bottom, .-z, z)
    vz = ifelse.(reflect_bottom, .-vz, vz)

    reflect_top = z .> chamber_height
    z = ifelse.(reflect_top, chamber_height .- (z .- chamber_height), z)
    vz = ifelse.(reflect_top, .-vz, vz)

    positions_updated = hcat(x, y, z)
    velocities_updated = hcat(vx, vy, vz)
    positions_alive = positions_updated
    velocities_alive = velocities_updated
    mask_alive = positions_updated[:, 3] .>= 0
    positions_alive = positions_updated[mask_alive, :]
    velocities_alive = velocities_updated[mask_alive, :]
    return positions_alive, velocities_alive
end

# Añadir después de inicializar nuevos electrones
# Al principio de cada paso de simulación
function limit_electron_energy!(velocities, min_eV=0.1, max_eV=1000.0)
    min_energy = min_eV * 1.60218e-19  # Convertir a Joules
    max_energy = max_eV * 1.60218e-19

    min_velocity = sqrt(2 * min_energy / electron_mass)
    max_velocity = sqrt(2 * max_energy / electron_mass)

    for i in 1:size(velocities, 1)
        v_mag = norm(velocities[i,:])
        if v_mag > max_velocity
            velocities[i,:] *= (max_velocity / v_mag)
        elseif v_mag < min_velocity && v_mag > 0
            velocities[i,:] *= (min_velocity / v_mag)
        end
    end
    return velocities
end