# --- particles.jl ---
using LinearAlgebra
using Random

# --- Funciones Auxiliares (Asumiendo que están aquí o accesibles) ---
function electron_velocity_from_energy(electron_energy_eV)
    energy_joules = electron_energy_eV * 1.60218e-19
    # Asegurar que la energía no sea negativa antes de sqrt
    if energy_joules < 0.0
        energy_joules = 0.0
    end
    return sqrt(2 * energy_joules / electron_mass) # electron_mass debe ser global o constante
end

function electron_energy_from_velocity(velocity_magnitude)
    return 0.5 * electron_mass * (velocity_magnitude ^ 2) # electron_mass debe ser global o constante
end

function lorentz_force(velocity, magnetic_field, charge)
    v_cross_B = cross(velocity, magnetic_field)
    return charge * v_cross_B
end

# --- Inicialización de Electrones (Sin Cambios, pero necesaria para el contexto) ---
function initialize_electrons(num_electrons, chamber_width, chamber_length, initial_electron_velocity)
    rng = MersenneTwister(0) # Puedes usar un RNG global si prefieres
    x_positions = rand(rng, num_electrons) .* chamber_width
    y_positions = rand(rng, num_electrons) .* chamber_length
    z_positions = zeros(num_electrons) # Inyectados en z=0

    vx = zeros(num_electrons)
    vy = zeros(num_electrons)
    vz = fill(initial_electron_velocity, num_electrons) # Velocidad inicial en z

    positions = hcat(x_positions, y_positions, z_positions)
    velocities = hcat(vx, vy, vz)
    return positions, velocities
end


# --- Movimiento de Electrones con Fuerza Eléctrica y Magnética (Algoritmo de Boris) ---
# !!! VERSIÓN COMPLETA MODIFICADA para incluir electric_field y usar Algoritmo de Boris !!!
function move_electrons(positions, velocities, dt, magnetic_field, electric_field, electron_charge, electron_mass)
    num_particles = size(positions, 1)
    if num_particles == 0
        return positions, velocities # Devuelve arrays vacíos si no hay partículas
    end

    new_velocities = copy(velocities)
    new_positions = copy(positions) # Inicializamos new_positions

    # Precalcular términos constantes que no dependen de la partícula
    q_over_m = electron_charge / electron_mass
    dt_half = dt / 2.0
    E_accel_half_dt = q_over_m * electric_field * dt_half # Vector de aceleración E para medio paso

    # --- Algoritmo de Boris ---
    for i in 1:num_particles
        v_n = velocities[i, :] # Velocidad al inicio del paso (t)

        # 1. Primera mitad de la aceleración eléctrica (v_n -> v_minus en t+dt/2)
        v_minus = v_n + E_accel_half_dt

        # 2. Rotación magnética (v_minus -> v_plus en t+dt/2)
        # Vector t = (q/m) * B * dt/2
        t = q_over_m * magnetic_field * dt_half
        t_mag_sq = dot(t, t) # |t|^2

        # Vector s = 2*t / (1 + |t|^2) (maneja el caso |t|=0 implícitamente)
        s = 2.0 * t / (1.0 + t_mag_sq)

        # v' = v⁻ + v⁻ x t
        v_prime = v_minus + cross(v_minus, t)

        # v⁺ = v⁻ + v' x s
        v_plus = v_minus + cross(v_prime, s)

        # 3. Segunda mitad de la aceleración eléctrica (v_plus -> v_n+1 en t+dt)
        v_n_plus_1 = v_plus + E_accel_half_dt # Velocidad al final del paso
        new_velocities[i, :] = v_n_plus_1

        # 4. Actualización de la posición
        # Usamos el método promedio (promedio de velocidad inicial y final del paso)
        # Alternativa Leapfrog estricta: new_positions[i, :] = positions[i, :] + v_n_plus_1 * dt
        new_positions[i, :] = positions[i, :] + 0.5 * (v_n + v_n_plus_1) * dt
    end

    # NOTA: La conservación de energía ya no se cumple exactamente cuando E != 0,
    # porque el campo eléctrico realiza trabajo sobre las partículas.
    # La verificación de energía sólo tiene sentido si E = 0.

    return new_positions, new_velocities
end


# --- Verificación de Timestep (Sin Cambios, pero relevante) ---
function check_timestep_validity(dt, magnetic_field_strength, electron_mass, electron_charge)
    if magnetic_field_strength > 1e-10 # Evita división por cero si B=0
        cyclotron_period = 2π * electron_mass / (abs(electron_charge) * magnetic_field_strength)
        dt_max_recommended = 0.1 * cyclotron_period # Recomendación común (factor 0.1-0.2)

        if dt > dt_max_recommended
            println("⚠️ ADVERTENCIA DE ESTABILIDAD:")
            println("  Timestep actual (dt):         $(dt) s")
            println("  Período de ciclotrón (T_c):   $(cyclotron_period) s")
            println("  Timestep máx. recomendado:  ~$(round(dt_max_recommended, sigdigits=3)) s (e.g., 0.1 * T_c)")
            println("  Ratio dt / T_c:             $(round(dt/cyclotron_period, sigdigits=3))")
            println("  La simulación puede ser inestable o imprecisa para el movimiento giroscópico.")
            println("  Considere reducir dt.")
            return false
        end
    end
    # Podrías añadir aquí una condición para el campo eléctrico si fuera necesario
    # Por ejemplo, basado en la frecuencia de plasma o el tiempo de cruce de celda.
    return true
end

# --- Condiciones de Frontera (Sin Cambios) ---
function apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)
    x = positions[:, 1]
    y = positions[:, 2]
    z = positions[:, 3]

    # Crear máscara para electrones dentro de los límites de la cámara
    mask_alive = (x .>= 0.0) .& (x .<= chamber_width) .&
                 (y .>= 0.0) .& (y .<= chamber_length) .&
                 (z .>= 0.0) .& (z .<= chamber_height)

    # Filtrar electrones que permanecen dentro
    positions_alive = positions[mask_alive, :]
    velocities_alive = velocities[mask_alive, :]

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