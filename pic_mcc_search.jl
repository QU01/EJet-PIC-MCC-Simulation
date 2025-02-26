using Interpolations
using Plots
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Dates
using DataFrames
using CSV

# --- Constantes Físicas y Propiedades del Aire ---
const k_b = 1.380649e-23
const amu = 1.66054e-27
const electron_mass = 9.109e-31
const electron_charge = -1.60218e-19

# --- Composición del Aire (Ajustada y Simplificada) ---
air_composition = Dict(
    "N2" => Dict{String, Any}(
        "mass" => 28.0134 * amu,
        "fraction" => 0.7808,
        "ionization_energy_eV" => 15.6
    ),
    "O2" => Dict{String, Any}(
        "mass" => 31.9988 * amu,
        "fraction" => 0.2095,
        "ionization_energy_eV" => 12.07
    ),
)

# --- Datos de Sección Eficaz de Itikawa (N2) ---
n2_energy_eV = [
    0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 17.0,
    20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0,
    150.0, 170.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0,
    800.0, 900.0, 1000.0
]

n2_total_cross_section = [
    4.88, 5.13, 5.56, 5.85, 6.25, 6.84, 7.32, 7.72, 8.06, 8.33, 8.61, 8.96, 9.25, 9.48,
    9.66, 9.85, 10.2, 11.2, 13.3, 25.7, 28.5, 21.0, 14.6, 13.2, 12.3, 11.8, 11.4, 11.4, 11.5,
    11.7, 12.0, 12.4, 13.2, 13.5, 13.7, 13.5, 13.0, 12.4, 12.0, 11.6, 11.3, 10.7, 10.2,
    9.72, 9.30, 8.94, 8.33, 7.48, 7.02, 6.43, 5.66, 5.04, 4.54, 4.15, 3.82, 3.55, 3.14,
    2.79, 2.55, 2.32, 2.13
] * 1e-20

n2_ionization_cross_section = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0211, 0.640, 0.962, 1.25, 1.54, 1.77, 1.91, 2.16, 2.30, 2.40, 2.47, 2.51, 2.48,
    2.28, 2.19, 1.98, 1.82, 1.68, 1.56, 1.45, 1.36, 1.20, 1.07, 0.971, 0.907, 0.847
] * 1e-20

# --- Datos de Sección Eficaz de Itikawa (O2) ---
o2_energy_eV = [
    0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,
    15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 120.0, 150.0, 170.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
    600.0, 700.0, 800.0, 900.0, 1000.0
]

o2_total_cross_section = [
    3.83, 4.02, 4.22, 4.33, 4.47, 4.65, 4.79, 4.91, 5.07, 5.20, 5.31, 5.49, 5.64, 5.77,
    5.87, 5.97, 6.18, 6.36, 6.45, 6.56, 6.68, 6.84, 7.01, 7.18, 7.36, 7.55, 7.93, 8.39,
    9.16, 9.91, 10.4, 10.8, 10.7, 10.7, 10.8, 11.0, 11.0, 10.9, 10.7, 10.5, 10.3, 9.87,
    9.52, 9.23, 8.98, 8.68, 7.97, 7.21, 6.78, 6.24, 5.51, 4.94, 4.55, 4.17, 3.85, 3.58,
    3.11, 2.76, 2.49, 2.26, 2.08
] * 1e-20

o2_ionization_cross_section = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0730, 0.383, 0.641, 0.927, 1.19, 1.42, 1.61, 1.78, 1.91, 2.04, 2.15, 2.22, 2.26,
    2.34, 2.38, 2.42, 2.43, 2.45, 2.42, 2.19, 2.01, 1.94, 1.80, 1.68, 1.56, 1.46, 1.38,
    1.30, 1.24
] * 1e-20

n2_total_cross_section_func = LinearInterpolation(n2_energy_eV, n2_total_cross_section; extrapolation_bc = Flat())
n2_ionization_cross_section_func = LinearInterpolation(n2_energy_eV, n2_ionization_cross_section; extrapolation_bc = Flat())

o2_total_cross_section_func = LinearInterpolation(o2_energy_eV, o2_total_cross_section; extrapolation_bc = Flat())
o2_ionization_cross_section_func = LinearInterpolation(o2_energy_eV, o2_ionization_cross_section; extrapolation_bc = Flat())

# --- Agregar las funciones de sección eficaz al diccionario de composición del aire ---
air_composition["N2"]["total_cross_section_func"] = n2_total_cross_section_func
air_composition["N2"]["ionization_cross_section_func"] = n2_ionization_cross_section_func

air_composition["O2"]["total_cross_section_func"] = o2_total_cross_section_func
air_composition["O2"]["ionization_cross_section_func"] = o2_ionization_cross_section_func

# ---  Cálculo de la Masa Promedio del Aire (Necesaria para colisiones elásticas) ---
avg_air_mass = sum(air_composition[gas]["mass"] * air_composition[gas]["fraction"] for gas in keys(air_composition))

# ---  Modelo de Excitación (Estimación) ---
function estimate_excitation_cross_section(total_cross_section, ionization_cross_section, energy_eV, ionization_energy_eV)
    cross_section_diff = total_cross_section - ionization_cross_section
    excitation_cross_section = max(0.0, cross_section_diff)
    energy_above_threshold = max(0.0, energy_eV - ionization_energy_eV)
    excitation_fraction = exp(-0.1 * energy_above_threshold)
    excitation_cross_section = excitation_cross_section * excitation_fraction
    return excitation_cross_section
end

# --- Funciones Auxiliares (Sin Cambios Mayores) ---
function calculate_air_density_n(pressure, temperature)
    return pressure / (k_b * temperature)
end

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

# --- Cálculo de la Densidad en la Malla (REVERTIDO A TUPLA DE VECTORES) ---
function calculate_grid_density(positions, x_grid, y_grid, z_grid)
    bins = (collect(x_grid), collect(y_grid), collect(z_grid))
    h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), bins)
    return h.weights
end

# --- Depositar Energía en la Malla (REVERTIDO A TUPLA DE VECTORES) ---
function deposit_energy(positions, energy_transfer, x_grid, y_grid, z_grid)
    bins = (collect(x_grid), collect(y_grid), collect(z_grid))
    w = Weights(energy_transfer)
    h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), w, bins)
    return h.weights
end

# --- Actualización de la Temperatura en la Malla (Sin Cambios) ---
function update_temperature_grid(temperature_grid, energy_deposition_grid, elastic_energy_deposition_grid, n_air, cell_volume)
    n_air_cell = n_air * cell_volume
    delta_T_inelastic = (2.0 / (3.0 * k_b)) * (energy_deposition_grid ./ n_air_cell)
    delta_T_elastic = (2.0 / (3.0 * k_b)) * (elastic_energy_deposition_grid ./ n_air_cell)
    new_temperature_grid = temperature_grid .+ delta_T_inelastic .+ delta_T_elastic
    return new_temperature_grid
end

# ---  Función de Colisión Monte Carlo (¡¡¡MODIFICADA!!!) ---
function monte_carlo_collision(positions, velocities, air_density_n, dt, rng, efficiency)
    num_particles = size(velocities, 1)
    if num_particles == 0
        return positions, velocities, rng, zeros(num_particles), zeros(Bool, num_particles), zeros(num_particles), zeros(num_particles)
    end

    # Vectores para resultados - ESENCIAL: inicializar a cero
    inelastic_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    # Cálculos de energía para diagnóstico
    v_magnitudes = sqrt.(sum(velocities.^2, dims=2))
    E_e_eV = electron_energy_from_velocity.(v_magnitudes) ./ 1.60218e-19  # Convertir J a eV
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)
    
    total_initial_energy_eV = sum(E_e_eV)
    total_initial_energy_J = sum(E_e_joules)
    """
    println("Diagnóstico Inicial:")
    println("  Número de electrones: $(num_particles)")
    println("  Media de energía de electrones: $(mean(E_e_eV)) eV")
    println("  Energía máxima de electrones: $(maximum(E_e_eV)) eV")
    println("  Energía mínima de electrones: $(minimum(E_e_eV)) eV")
    println("  Energía TOTAL inicial: $(total_initial_energy_eV) eV ($(total_initial_energy_J) J)")
    """
    # Constantes físicas
    excitation_energy_loss_eV = 3.0
    excitation_energy_loss_joules = excitation_energy_loss_eV * 1.60218e-19
    
    # Para cada partícula
    for i in 1:num_particles
        # IMPORTANTE: La energía disponible para este electrón
        electron_energy_joules = E_e_joules[i]
        
        # Si el electrón no tiene suficiente energía, saltar
        if electron_energy_joules < 1e-25
            continue
        end
        
        # Probabilidad total de colisión
        total_collision_prob = 0.0
        gas_probs = Dict{String, Float64}()
        
        for gas_name in keys(air_composition)
            gas_info = air_composition[gas_name]
            gas_fraction = gas_info["fraction"]
            total_cross_section_func = gas_info["total_cross_section_func"]
            total_cross_section = total_cross_section_func(E_e_eV[i])
            
            # Probabilidad de colisión por densidad, sección eficaz y tiempo
            gas_prob = (1.0 - exp(-(air_density_n * gas_fraction) * total_cross_section * v_magnitudes[i] * dt))
            gas_probs[gas_name] = gas_prob
            total_collision_prob += gas_prob * gas_fraction
        end
        
        # Limitar probabilidad por seguridad numérica
        total_collision_prob = min(total_collision_prob, 0.95)
        
        # Decidir si colisiona
        if rand(rng) < total_collision_prob
            # Seleccionar gas para colisión (ponderado por fracción)
            selected_gas = ""
            rand_val = rand(rng)
            cumulative_prob = 0.0
            
            for gas_name in keys(air_composition)
                gas_fraction = air_composition[gas_name]["fraction"]
                gas_prob = gas_probs[gas_name] * gas_fraction / total_collision_prob
                
                cumulative_prob += gas_prob
                if rand_val <= cumulative_prob
                    selected_gas = gas_name
                    break
                end
            end
            
            if selected_gas == ""
                selected_gas = last(keys(air_composition))
            end
            
            # Propiedades del gas seleccionado
            gas_info = air_composition[selected_gas]
            mass = gas_info["mass"]
            ionization_energy_eV = gas_info["ionization_energy_eV"]
            ionization_energy_joules = ionization_energy_eV * 1.60218e-19
            total_cross_section_func = gas_info["total_cross_section_func"]
            ionization_cross_section_func = gas_info["ionization_cross_section_func"]
            
            # Secciones eficaces
            total_cross_section = total_cross_section_func(E_e_eV[i])
            ionization_cross_section = ionization_cross_section_func(E_e_eV[i])
            excitation_cross_section = estimate_excitation_cross_section(
                total_cross_section, ionization_cross_section, E_e_eV[i], ionization_energy_eV)
            elastic_cross_section = max(0.0, total_cross_section - ionization_cross_section - excitation_cross_section)
            
            # Probabilidades relativas
            p_total = ionization_cross_section + excitation_cross_section + elastic_cross_section
            if p_total > 0
                p_ionize = ionization_cross_section / p_total
                p_excite = excitation_cross_section / p_total
                p_elastic = elastic_cross_section / p_total
            else
                p_elastic = 1.0
                p_ionize = 0.0
                p_excite = 0.0
            end
            
            # Decidir tipo de colisión
            collision_rand = rand(rng)
            energy_loss_joules = 0.0
            collision_type = "Elástica"
            
            # CRÍTICO: NO PUEDE TRANSFERIR MÁS ENERGÍA DE LA QUE TIENE
            max_transfer = 0.95 * electron_energy_joules
            
            # IONIZACIÓN (si tiene suficiente energía)
            if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(ionization_energy_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Ionización"
            
            # EXCITACIÓN (si tiene suficiente energía)
            elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > excitation_energy_loss_joules
                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(excitation_energy_loss_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Excitación"
                
            # ELÁSTICA
            else
                # Fracción teórica de transferencia elástica (ecuación física)
                fraction_energy_elastic = (2 * electron_mass / mass)
                
                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
                elastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Elástica"
            end
            
            # CRUCIAL: Actualizar velocidad con energía remanente
            remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
            new_speed = sqrt(2 * remaining_energy_joules / electron_mass)
            
            # Nueva dirección aleatoria para dispersión
            new_direction = randn(rng, 3)
            new_direction_norm = sqrt(sum(new_direction.^2))
            if new_direction_norm > 0
                new_direction = new_direction ./ new_direction_norm
            else
                new_direction = [0.0, 0.0, 1.0]
            end
            
            # Actualizar velocidad
            velocities_new[i, :] = new_direction .* new_speed
            
            # Registrar colisión
            collided_flags[i] = true
            collision_energy_transfers_eV[i] = energy_loss_joules / 1.60218e-19
            
            # Diagnóstico para primeras partículas
            """
            if i <= 3
                println("Colisión #$i con gas $selected_gas:")
                println("  E inicial = $(E_e_eV[i]) eV ($(electron_energy_joules) J)")
                println("  Tipo colisión: $collision_type")
                println("  Pérdida energía = $(energy_loss_joules/1.60218e-19) eV ($(energy_loss_joules) J)")
                println("  E final = $(remaining_energy_joules/1.60218e-19) eV ($(remaining_energy_joules) J)")
                println("  Ratio transferencia/inicial = $(energy_loss_joules/electron_energy_joules * 100)%")
            end
            """
        end
    end
    
    # Diagnóstico final con verificación de conservación de energía
    new_v_magnitudes = sqrt.(sum(velocities_new.^2, dims=2))
    new_E_e_eV = electron_energy_from_velocity.(new_v_magnitudes) ./ 1.60218e-19
    new_E_e_joules = electron_energy_from_velocity.(new_v_magnitudes)
    
    # Sumas totales para verificación
    inelastic_sum_J = sum(inelastic_energy_transfer)
    elastic_sum_J = sum(elastic_energy_transfer)
    inelastic_sum_eV = inelastic_sum_J / 1.60218e-19
    elastic_sum_eV = elastic_sum_J / 1.60218e-19
    
    total_final_energy_J = sum(new_E_e_joules)
    total_final_energy_eV = sum(new_E_e_eV)
    
    energy_lost_J = total_initial_energy_J - total_final_energy_J
    energy_lost_eV = total_initial_energy_eV - total_final_energy_eV
    
    energy_transferred_J = inelastic_sum_J + elastic_sum_J
    energy_transferred_eV = inelastic_sum_eV + elastic_sum_eV
    """
    println("Diagnóstico Final:")
    println("  Media de energía después de colisiones: $(mean(new_E_e_eV)) eV")
    println("  Energía máxima después de colisiones: $(maximum(new_E_e_eV)) eV")
    println("  Energía TOTAL final: $(total_final_energy_eV) eV ($(total_final_energy_J) J)")
    println("  Pérdida de energía total: $(energy_lost_eV) eV ($(energy_lost_J) J)")
    println("  Energía transferida inelástica total: $(inelastic_sum_eV) eV ($(inelastic_sum_J) J)")
    println("  Energía transferida elástica total: $(elastic_sum_eV) eV ($(elastic_sum_J) J)")
    println("  Energía transferida total: $(energy_transferred_eV) eV ($(energy_transferred_J) J)")
    """
    # VERIFICACIÓN ESENCIAL: La energía transferida debe ser igual (o muy cercana) a la energía perdida
    conservation_error = abs(energy_lost_J - energy_transferred_J) / max(1e-30, energy_transferred_J)
    if conservation_error > 0.01
       println("⚠️ ERROR DE CONSERVACIÓN: Diferencia entre energía perdida y transferida: $(energy_lost_eV - energy_transferred_eV) eV")
    else
        println("✓ Conservación de energía verificada (error relativo = $(conservation_error*100)%).")
    end

    return positions, velocities_new, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer
end

# --- Función para Generar Reporte de Simulación ---
function generate_report(filename, initial_temperature, initial_pressure, electron_injection_energy_eV, magnetic_field_strength, dt, simulation_time, simulated_electrons_per_step, efficiency_simulation, increase_internal_energy, total_input_energy, final_step, reached_target_temp, avg_temps_history_julia, avg_efficiency_julia)
    open(filename, "w") do io
        println(io, "--- Reporte de Simulación PIC-MCC ---")
        println(io, "Fecha y Hora: $(Dates.now())")
        println(io, "\n--- Parámetros de Simulación ---")
        println(io, "Temperatura Inicial: $(initial_temperature) K")
        println(io, "Presión Inicial: $(initial_pressure/1e6) MPa")
        println(io, "Densidad Inicial del Aire: $(initial_air_density_n) m^-3")
        println(io, "Energía Inicial de los Electrones: $(electron_injection_energy_eV) eV")
        println(io, "Campo Magnético Axial (Bz): $(magnetic_field_strength) T")
        println(io, "Paso de Tiempo (dt): $(dt) s")
        println(io, "Tiempo de Simulación por paso: $(dt) s") # Clarify time per step
        println(io, "Electrones Simulados por paso: $(simulated_electrons_per_step)")
        println(io, "Temperatura Objetivo: $(target_temperature) K")
        println(io, "Máximo número de pasos: $(max_steps)")
        println(io, "\n--- Resultados ---")
        println(io, "Temperatura Final Promedio: $(avg_temps_history_julia[end]) K")
        println(io, "Alcanzó Temperatura Objetivo: $(reached_target_temp)")
        println(io, "Número de Pasos Simulados: $(final_step)")
        println(io, "Tiempo Total de Simulación: $(final_step * dt) s") # Total simulation time
        println(io, "Aumento Total de Energía Interna del Aire: $(increase_internal_energy) J")
        println(io, "Energía Total Introducida por Electrones: $(total_input_energy) J")
        println(io, "Eficiencia de la Simulación: $(efficiency_simulation) %") # Now reports avg_efficiency_julia
        println(io, "Eficiencia Promedio por Paso: $(avg_efficiency_julia) %") # Keep for clarity
        println(io, "\n--- Plots ---")
        println(io, "Temperatura Promedio vs Tiempo: temperature_vs_time.png")
        println(io, "Eficiencia Promedio vs Tiempo: efficiency_vs_time.png") # Added efficiency vs time plot to report
        println(io, "Densidad de Electrones al Final (Slice): density_heatmap.png")
        println(io, "Temperatura del Aire al Final (Slice): temperature_heatmap.png")
    end
    println("\nReporte guardado en: $(filename)")
end

# --- Función para Graficar Temperatura vs Tiempo ---
function plot_temperature_vs_time(time_points, avg_temps_history_julia, target_temperature)
    p1 = plot(time_points, avg_temps_history_julia,
              label="Temperatura Promedio del Aire",
              xlabel="Tiempo (µs)", ylabel="Temperatura (K)",
              title="Simulación PIC-MCC: Calentamiento del Aire hasta Temperatura Objetivo",
              grid=true)
    hline!(p1, [target_temperature],
           color="red", linestyle=:dash,
           label="Temperatura Objetivo ($(target_temperature) K)")
    return p1
end

# --- Función para Graficar Eficiencia vs Tiempo ---
function plot_efficiency_vs_time(time_points, efficiency_history_julia)
    p4 = plot(time_points, efficiency_history_julia,
              label="Eficiencia por Paso",
              xlabel="Tiempo (µs)", ylabel="Eficiencia (%)",
              title="Eficiencia de Calentamiento por Paso",
              grid=true)
    return p4
end


# --- Funciones Heatmap para Densidad y Temperatura ---
function heatmap_density_slice(x_grid, y_grid, final_density_grid, z_slice_index)
    p2 = heatmap(x_grid*1000, y_grid*1000, final_density_grid[:, :, z_slice_index]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Densidad de Electrones al Final (Slice en z=mitad, con Campo B)",
                 aspect_ratio=:auto, color=:viridis,
                 colorbar_title="Densidad de Electrones (u.a.)")
    return p2
end

function heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, z_slice_index)
    p3 = heatmap(x_grid*1000, y_grid*1000, final_temperature_grid[:, :, z_slice_index]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Temperatura del Aire al Final (Slice en z=mitad, con Campo B)",
                 aspect_ratio=:auto, color=:magma,
                 colorbar_title="Temperatura (K)")
    return p3
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
# --- Función Principal de Simulación (modificada para calcular eficiencia por paso) ---
# Modificar run_pic_simulation para aceptar max_steps como parámetro opcional
function run_pic_simulation(initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass, initial_electron_velocity; verbose=true, max_steps_override=nothing)

    is_stable = check_timestep_validity(dt, magnetic_field_strength, electron_mass, electron_charge)
    if !is_stable
        println("Continuando con timestep potencialmente inestable...")
    end
    positions_history = [initial_positions]
    velocities_history = [initial_velocities]
    temperatures_history = [initial_temperature_grid]
    avg_temps_history = [Statistics.mean(initial_temperature_grid)]
    density_history = [calculate_grid_density(initial_positions, x_grid, y_grid, z_grid)]
    energy_deposition_history = [zeros(size(initial_temperature_grid))]
    elastic_energy_deposition_history = [zeros(size(initial_temperature_grid))]
    efficiency_history = Float64[]

    # Datos adicionales para análisis detallado
    electron_count_history = [size(initial_positions, 1)]
    inelastic_energy_history = Float64[]
    elastic_energy_history = Float64[]
    total_energy_transfer_history = Float64[]
    input_energy_history = Float64[]

    positions = initial_positions
    velocities = initial_velocities
    temperature_grid = initial_temperature_grid
    rng = MersenneTwister(0)

    current_time = 0.0
    step = 0
    reached_target_temp = false
    accumulated_input_energy = 0.0

    # Usar max_steps_override si se proporciona, de lo contrario usar max_steps global
    local_max_steps = isnothing(max_steps_override) ? max_steps : max_steps_override

    # Para calcular la energía interna inicial
    initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n * cell_volume * TOTAL_CELLS
    current_internal_energy = initial_internal_energy

    while avg_temps_history[end] < target_temperature && step < local_max_steps
        step += 1
        current_time += dt

        if verbose
            println("\n--- Simulación en el paso $(step), tiempo = $(current_time * 1e6) µs ---")
            println("Temperatura Promedio Inicial del Paso: $(avg_temps_history[end]) K")
        end

        # Inyectar nuevos electrones
        new_positions, new_velocities = initialize_electrons(simulated_electrons_per_step, chamber_width, chamber_length, initial_electron_velocity)

        # Calcular energía cinética de nuevos electrones inyectados
        step_input_energy = sum(electron_energy_from_velocity.(sqrt.(sum(new_velocities.^2, dims=2)))) * particle_weight
        accumulated_input_energy += step_input_energy
        push!(input_energy_history, step_input_energy)

        if verbose
            println("Energía Cinética de Nuevos Electrones (Input de este paso): $(step_input_energy) J")
        end

        positions = vcat(positions, new_positions)
        velocities = vcat(velocities, new_velocities)
        push!(electron_count_history, size(positions, 1))

        # Guardar energía interna antes de las colisiones
        previous_internal_energy = current_internal_energy

        # Movimiento de electrones y condiciones de frontera
        positions, velocities = move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
        positions, velocities = apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)

        total_initial_energy = sum(electron_energy_from_velocity.(sqrt.(sum(velocities.^2, dims=2))))

        # Simulación de colisiones - AHORA RECIBE inelastic_energy_transfer en lugar de total_energy_transfer
        positions, velocities, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer = monte_carlo_collision(
            positions, velocities, initial_air_density_n, dt, rng, 1.0
        )

        total_final_energy = sum(electron_energy_from_velocity.(sqrt.(sum(velocities.^2, dims=2))))
        energy_lost_by_electrons = total_initial_energy - total_final_energy

        velocities = limit_electron_energy!(velocities)

        inelastic_energy_step = sum(inelastic_energy_transfer)
        elastic_energy_step = sum(elastic_energy_transfer)
        total_energy_transfer = sum(inelastic_energy_transfer) + sum(elastic_energy_transfer)
        if total_energy_transfer > 0 && abs(energy_lost_by_electrons - total_energy_transfer)/total_energy_transfer > 0.01
            println("⚠️ ERROR DE CONSERVACIÓN: Energía perdida por electrones: $energy_lost_by_electrons J")
            println("⚠️ ERROR DE CONSERVACIÓN: Energía transferida reportada: $total_energy_transfer J")
            println("⚠️ Esto indica un error en el modelo de colisiones")
        end

        # Imprimir los valores de energía aquí
        if verbose
            println("  Energía de Entrada del Paso: $(step_input_energy) J")
            println("  Energía de Transferencia Elástica (electrones reales): $(elastic_energy_step * particle_weight) J")
            println("  Energía de Transferencia Inelástica (electrones reales): $(inelastic_energy_step * particle_weight) J")
        end

        adjusted_particle_weight = min(particle_weight, 1e10/sum(inelastic_energy_transfer + elastic_energy_transfer))

        energy_deposition_grid_step = deposit_energy(positions, inelastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)
        elastic_energy_deposition_grid_step = deposit_energy(positions, elastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)

        inelastic_energy_step_weighted = inelastic_energy_step * particle_weight
        elastic_energy_step_weighted = elastic_energy_step * particle_weight
        # Guardar historial de transferencia de energía
        
        push!(inelastic_energy_history, inelastic_energy_step_weighted)
        push!(elastic_energy_history, elastic_energy_step_weighted)
        push!(total_energy_transfer_history, inelastic_energy_step_weighted + elastic_energy_step_weighted)

        temperature_grid = update_temperature_grid(temperature_grid, energy_deposition_grid_step, elastic_energy_deposition_grid_step, initial_air_density_n, cell_volume)

        # Calcular la nueva energía interna del gas
        current_internal_energy = (3/2) * k_b * Statistics.mean(temperature_grid) * initial_air_density_n * cell_volume * TOTAL_CELLS
        energy_transferred_to_gas = current_internal_energy - previous_internal_energy

        total_transfer_step = inelastic_energy_step_weighted + elastic_energy_step_weighted
        if total_transfer_step > step_input_energy
            println("⚠️ Advertencia: Energía transferida ($(total_transfer_step) J) > energía de entrada ($(step_input_energy) J)")
            scaling_factor = step_input_energy / total_transfer_step
            inelastic_energy_step_weighted *= scaling_factor
            elastic_energy_step_weighted *= scaling_factor
            
            # Reajustar depósitos de energía
            energy_deposition_grid_step *= scaling_factor
            elastic_energy_deposition_grid_step *= scaling_factor
        end

        # MODIFICACIÓN: Verificar que la energía transferida no exceda la entrada
        if energy_transferred_to_gas > step_input_energy
            if verbose
                println("⚠️ Advertencia: Detección de energía transferida ($(energy_transferred_to_gas) J) mayor que la energía de entrada ($(step_input_energy) J)")
                println("   Limitando la energía transferida al valor físicamente posible.")
            end
            energy_transferred_to_gas = step_input_energy
        end

        # Calcular la eficiencia en este paso
        if step_input_energy > 0
            step_efficiency = (energy_transferred_to_gas / step_input_energy) * 100
        else
            step_efficiency = 0.0
        end
        push!(efficiency_history, step_efficiency)

        if verbose
            println("Energía Transferida al Gas: $(energy_transferred_to_gas) J")
            println("Eficiencia en este paso: $(step_efficiency) %")
        end

        push!(energy_deposition_history, energy_deposition_grid_step)
        push!(elastic_energy_deposition_history, elastic_energy_deposition_grid_step)
        push!(temperatures_history, temperature_grid)
        push!(avg_temps_history, Statistics.mean(temperature_grid))
        push!(density_history, calculate_grid_density(positions, x_grid, y_grid, z_grid))
        push!(positions_history, positions)
        push!(velocities_history, velocities)

        if avg_temps_history[end] >= target_temperature
            reached_target_temp = true
            if verbose
                println("Temperatura objetivo alcanzada en el paso $(step)!")
            end
            break
        end
    end

    avg_efficiency = isempty(efficiency_history) ? 0.0 : Statistics.mean(efficiency_history)
    if verbose
        println("\nEficiencia promedio a lo largo de la simulación: $(avg_efficiency) %")
    end

    # Recopilar datos detallados para análisis
    detailed_data = Dict(
        "electron_count_history" => electron_count_history,
        "inelastic_energy_history" => inelastic_energy_history,
        "elastic_energy_history" => elastic_energy_history,
        "total_energy_transfer_history" => total_energy_transfer_history,
        "input_energy_history" => input_energy_history
    )

    return temperatures_history, avg_temps_history, density_history,
           energy_deposition_history, elastic_energy_deposition_history, step, reached_target_temp,
           accumulated_input_energy, efficiency_history, avg_efficiency, detailed_data
end


# --- Exportación de Datos a CSV para Interfaz Gráfica ---
function export_simulation_data_to_csv(time_points, avg_temps_history, efficiency_history,
    final_density_grid, final_temperature_grid,
    x_grid, y_grid, z_grid, simulation_params,
    detailed_data, results_df, best_params)
    println("\n--- Exportando datos a CSV para visualización externa ---")

    # Asegurar que la carpeta de datos exista
    data_dir = "simulation_data"
    isdir(data_dir) || mkdir(data_dir)

    # 1. Exportar temperatura vs tiempo
    temp_time_df = DataFrame(
    Time_microseconds = time_points,
    Temperature_K = avg_temps_history
    )
    CSV.write("$data_dir/temperature_vs_time.csv", temp_time_df)

    # 2. Exportar eficiencia vs tiempo (quitamos el primer punto que no tiene eficiencia)
    efficiency_time_df = DataFrame(
    Time_microseconds = time_points[2:end],
    Efficiency_percent = efficiency_history
    )
    CSV.write("$data_dir/efficiency_vs_time.csv", efficiency_time_df)

    # 3. Exportar malla de temperatura final (plano central)
    z_slice_index = num_z_cells ÷ 2
    temp_slice = final_temperature_grid[:, :, z_slice_index]

    # Crear DataFrame con coordenadas x, y y temperatura
    temp_grid_data = []
    for i in 1:length(x_grid)-1
        for j in 1:length(y_grid)-1
        push!(temp_grid_data, (x_grid[i], y_grid[j], temp_slice[i, j]))
        end
        end

        temp_grid_df = DataFrame(temp_grid_data, [:x_mm, :y_mm, :temperature_K])
        temp_grid_df.x_mm .*= 1000  # Convertir a mm
        temp_grid_df.y_mm .*= 1000  # Convertir a mm
        CSV.write("$data_dir/temperature_grid.csv", temp_grid_df)

        # 4. Exportar malla de densidad de electrones final (plano central)
        density_slice = final_density_grid[:, :, z_slice_index]

        # Crear DataFrame con coordenadas x, y y densidad
        density_grid_data = []
        for i in 1:length(x_grid)-1
        for j in 1:length(y_grid)-1
        push!(density_grid_data, (x_grid[i], y_grid[j], density_slice[i, j]))
        end
    end

    density_grid_df = DataFrame(density_grid_data, [:x_mm, :y_mm, :electron_density])
    density_grid_df.x_mm .*= 1000  # Convertir a mm
    density_grid_df.y_mm .*= 1000  # Convertir a mm
    CSV.write("$data_dir/density_grid.csv", density_grid_df)

    # 5. Exportar parámetros de simulación para metadatos
    sim_params_df = DataFrame(
    Parameter = [
    "Initial Temperature (K)",
    "Target Temperature (K)",
    "Initial Pressure (MPa)",
    "Electron Energy (eV)",
    "Magnetic Field (T)",
    "Time Step (s)",
    "Chamber Width (mm)",
    "Chamber Length (mm)",
    "Chamber Height (mm)",
    "Final Average Temperature (K)",
    "Average Efficiency (%)",
    "Total Simulation Time (μs)"
    ],
    Value = [
    simulation_params["initial_temperature"],
    simulation_params["target_temperature"],
    simulation_params["initial_pressure"]/1e6,
    simulation_params["electron_energy_eV"],
    simulation_params["magnetic_field_strength"],
    simulation_params["dt"],
    simulation_params["chamber_width"]*1000,
    simulation_params["chamber_length"]*1000,
    simulation_params["chamber_height"]*1000,
    avg_temps_history[end],
    simulation_params["avg_efficiency"],
    time_points[end]
    ]
    )
    CSV.write("$data_dir/simulation_parameters.csv", sim_params_df)

    # 6. NUEVO: Exportar datos adicionales para análisis detallado
    detailed_df = DataFrame(
    Time_microseconds = time_points[2:end],
    Electron_Count = detailed_data["electron_count_history"][2:end],
    Input_Energy_J = detailed_data["input_energy_history"],
    Inelastic_Energy_Transfer_J = detailed_data["inelastic_energy_history"],
    Elastic_Energy_Transfer_J = detailed_data["elastic_energy_history"],
    Total_Energy_Transfer_J = detailed_data["total_energy_transfer_history"],
    Efficiency_percent = efficiency_history
    )
    CSV.write("$data_dir/detailed_simulation_data.csv", detailed_df)

    # 7. NUEVO: Exportar resultados de la búsqueda de parámetros
    CSV.write("$data_dir/parameter_search_results.csv", results_df)

    # 8. NUEVO: Exportar mejores parámetros encontrados
    best_params_df = DataFrame(
    Parameter = ["Electron Energy (eV)", "Pressure (MPa)", "Magnetic Field (T)", "Best Efficiency (%)"],
    Value = [best_params[1], best_params[2]/1e6, best_params[3], best_efficiency]
    )
    CSV.write("$data_dir/best_parameters.csv", best_params_df)

    # 9. NUEVO: Crear conjuntos de datos para gráficas comparativas específicas

    # 9.1 Eficiencia vs Energía de Electrones para diferentes presiones y campos magnéticos
    energy_comp_data = []
    for field in unique(results_df.MagneticField)
    for pressure in unique(results_df.Pressure)
    subset = results_df[(results_df.MagneticField .== field) .& (results_df.Pressure .== pressure), :]
    if !isempty(subset)
    for row in eachrow(subset)
    push!(energy_comp_data, (row.ElectronEnergy, row.Efficiency, pressure/1e6, field))
    end
    end
    end
    end
    energy_comp_df = DataFrame(energy_comp_data, [:ElectronEnergy_eV, :Efficiency_percent, :Pressure_MPa, :MagneticField_T])
    CSV.write("$data_dir/efficiency_vs_energy.csv", energy_comp_df)

    # 9.2 Eficiencia vs Presión para diferentes energías y campos magnéticos
    pressure_comp_data = []
    for field in unique(results_df.MagneticField)
    for energy in unique(results_df.ElectronEnergy)
    subset = results_df[(results_df.MagneticField .== field) .& (results_df.ElectronEnergy .== energy), :]
    if !isempty(subset)
    for row in eachrow(subset)
    push!(pressure_comp_data, (row.Pressure/1e6, row.Efficiency, energy, field))
    end
    end
    end
    end
    pressure_comp_df = DataFrame(pressure_comp_data, [:Pressure_MPa, :Efficiency_percent, :ElectronEnergy_eV, :MagneticField_T])
    CSV.write("$data_dir/efficiency_vs_pressure.csv", pressure_comp_df)

    # 9.3 Eficiencia vs Campo Magnético para diferentes energías y presiones
    field_comp_data = []
    for pressure in unique(results_df.Pressure)
    for energy in unique(results_df.ElectronEnergy)
    subset = results_df[(results_df.Pressure .== pressure) .& (results_df.ElectronEnergy .== energy), :]
    if !isempty(subset)
    for row in eachrow(subset)
    push!(field_comp_data, (row.MagneticField, row.Efficiency, energy, pressure/1e6))
    end
    end
    end
    end
    field_comp_df = DataFrame(field_comp_data, [:MagneticField_T, :Efficiency_percent, :ElectronEnergy_eV, :Pressure_MPa])
    CSV.write("$data_dir/efficiency_vs_magnetic_field.csv", field_comp_df)

    println("Datos exportados con éxito a la carpeta '$data_dir'")
    println("Archivos generados para análisis de eficiencia vs parámetros:")
    println("  - parameter_search_results.csv")
    println("  - best_parameters.csv")
    println("  - efficiency_vs_energy.csv")
    println("  - efficiency_vs_pressure.csv")
    println("  - efficiency_vs_magnetic_field.csv")
    println("Archivos generados para análisis detallado de simulación:")
    println("  - temperature_vs_time.csv")
    println("  - efficiency_vs_time.csv")
    println("  - detailed_simulation_data.csv")
    println("  - temperature_grid.csv")
    println("  - density_grid.csv")
    println("  - simulation_parameters.csv")
end



# Modificar estimate_efficiency para retornar avg_efficiency_julia como eficiencia principal
function estimate_efficiency(electron_injection_energy_eV, initial_pressure, magnetic_field_strength; max_steps_override=nothing, initial_electron_velocity=electron_velocity_from_energy(electron_injection_energy_eV))
    magnetic_field = [0.0, 0.0, magnetic_field_strength]

    initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature)
    initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
    initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
    initial_air_density_n_value = Float64(initial_air_density_n)

    (temperatures_history_julia, avg_temps_history_julia, density_history_julia,
     energy_deposition_history_julia, elastic_energy_deposition_history_julia, final_step, reached_target_temp, accumulated_input_energy, efficiency_history_julia, avg_efficiency_julia) = run_pic_simulation(
        initial_positions, initial_velocities, initial_temperature_grid,
        initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
        magnetic_field, electron_charge, electron_mass, initial_electron_velocity, verbose=false,
        max_steps_override=max_steps_override
    )

    # Efficiency is now the average step efficiency
    efficiency_simulation = avg_efficiency_julia

    return efficiency_simulation, avg_temps_history_julia[end], final_step, reached_target_temp, avg_efficiency_julia
end

function parameter_search()
    # Definir rangos de búsqueda para los parámetros
    electron_energies = [50.0, 100.0, 150.0, 200.0]  # eV
    pressures = [1e6, 2e6, 3e6, 4e6]  # Pa
    magnetic_fields = [0.5, 1.0, 1.5, 2.0]  # Tesla

    # Número reducido de pasos para la búsqueda rápida
    search_max_steps = 20

    # Inicializar almacenamiento de resultados
    results = DataFrame(
        ElectronEnergy = Float64[],
        Pressure = Float64[],
        MagneticField = Float64[],
        Efficiency = Float64[],
        FinalTemperature = Float64[],
        Steps = Int[],
        ReachedTarget = Bool[],
        AvgStepEfficiency = Float64[],
        SimulationTime = Float64[], # Tiempo total de simulación
        HeatingRate = Float64[]    # Tasa de calentamiento (K/μs)
    )

    # Realizar búsqueda en cuadrícula
    best_efficiency = 0.0
    best_params = (0.0, 0.0, 0.0)

    for energy in electron_energies
        for pressure in pressures
            for field in magnetic_fields
                println("\nEvaluando parámetros: Energía = $energy eV, Presión = $(pressure/1e6) MPa, Campo Magnético = $field T")

                # Calculate initial electron velocity here and pass it to estimate_efficiency
                initial_electron_velocity_param_search = electron_velocity_from_energy(energy)
                efficiency, final_temp, steps, reached_target, avg_step_efficiency =
                    estimate_efficiency(energy, pressure, field; max_steps_override=search_max_steps, initial_electron_velocity=initial_electron_velocity_param_search)

                # Calcular métricas adicionales
                simulation_time = steps * dt * 1e6  # Tiempo en microsegundos
                heating_rate = (final_temp - initial_temperature) / simulation_time  # K/μs

                println("  Resultado: Eficiencia Promedio por Paso = $(round(efficiency, digits=2))%, Temperatura Final = $(round(final_temp, digits=1)) K")
                println("  Tasa de Calentamiento = $(round(heating_rate, digits=2)) K/μs")

                # Guardar resultados
                push!(results, (energy, pressure, field, efficiency, final_temp, steps, reached_target, avg_step_efficiency, simulation_time, heating_rate))

                # Actualizar mejores parámetros
                if efficiency > best_efficiency
                    best_efficiency = efficiency
                    best_params = (energy, pressure, field)
                    println("  ¡NUEVO MEJOR RESULTADO!")
                end
            end
        end
    end

    # Ordenar resultados por eficiencia
    sort!(results, :Efficiency, rev=true)

    return results, best_params, best_efficiency
end


# --- Parámetros de la Cámara ---
chamber_width = 0.1
chamber_length = 0.1
chamber_height = 0.1
num_x_cells = 20
num_y_cells = 20
num_z_cells = 50
TOTAL_CELLS = Int(num_x_cells * num_y_cells * num_z_cells)

# --- Malla Rectangular ---
x_grid = LinRange(0, chamber_width, num_x_cells + 1)
y_grid = LinRange(0, chamber_length, num_y_cells + 1)
z_grid = LinRange(0, chamber_height, num_z_cells + 1)
x_cell_size = x_grid[2] - x_grid[1]
y_cell_size = y_grid[2] - y_grid[1]
z_cell_size = z_grid[2] - z_grid[1]
cell_volume = x_cell_size * y_cell_size * z_cell_size

# --- Parámetros de Simulación PIC (Ajustados) ---
initial_temperature = 800.0
target_temperature = 2200.0
dt = 1e-12
simulation_time = 1e-7 # No longer used for stopping condition
simulated_electrons_per_step = Int(100)
physical_electrons_per_step = 5e17
particle_weight = physical_electrons_per_step / simulated_electrons_per_step
max_steps = Int(1000) # Reducido para diagnóstico
electron_injection_energy_eV = 50.0 # Parameter from error message example
initial_pressure = 1e6 # Parameter from error message example
magnetic_field_strength = 0.5 # Parameter from error message example


# Ejecutar búsqueda de parámetros
println("\n--- Iniciando Búsqueda de Parámetros Óptimos ---")
results_df, best_params, best_efficiency = parameter_search()

# Mostrar mejores parámetros
println("\n--- Mejores Parámetros Encontrados ---")
println("Energía de Electrón: $(best_params[1]) eV")
println("Presión: $(best_params[2]/1e6) MPa")
println("Campo Magnético: $(best_params[3]) T")
println("Eficiencia Promedio por Paso: $(best_efficiency) %") # Updated message

# Mostrar los mejores resultados
println("\n--- Top 5 Combinaciones de Parámetros ---")
first_5_rows = first(results_df, 5)
for i in 1:nrow(first_5_rows)
    row = first_5_rows[i, :]
    println("Rank $i: Energía = $(row.ElectronEnergy) eV, Presión = $(row.Pressure/1e6) MPa, Campo = $(row.MagneticField) T, Eficiencia Promedio por Paso = $(round(row.Efficiency, digits=2)) %") # Updated message
end

# Guardar resultados en un archivo de texto
open("parametros_optimos.txt", "w") do io
    println(io, "--- Resultados de Búsqueda de Parámetros ---")
    println(io, "Fecha y Hora: $(Dates.now())")
    println(io, "\n--- Mejores Parámetros ---")
    println(io, "Energía de Electrón: $(best_params[1]) eV")
    println(io, "Presión: $(best_params[2]/1e6) MPa")
    println(io, "Campo Magnético: $(best_params[3]) T")
    println(io, "Eficiencia Promedio por Paso: $(best_efficiency) %") # Updated message

    println(io, "\n--- Top 10 Combinaciones de Parámetros ---")
    for i in 1:min(10, nrow(results_df))
        row = results_df[i, :]
        println(io, "Rank $i: Energía = $(row.ElectronEnergy) eV, Presión = $(row.Pressure/1e6) MPa, Campo = $(row.MagneticField) T, Eficiencia Promedio por Paso = $(round(row.Efficiency, digits=2)) %") # Updated message
    end
end
println("\nResultados guardados en parametros_optimos.txt")

# Ejecutar la simulación completa con los mejores parámetros
println("\n--- Ejecutando Simulación Completa con Parámetros Óptimos ---")
electron_injection_energy_eV = best_params[1]
initial_pressure = best_params[2]
magnetic_field_strength = best_params[3]

# Actualizar variables globales con los mejores parámetros
magnetic_field = [0.0, 0.0, magnetic_field_strength]
initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature)
initial_electron_velocity = electron_velocity_from_energy(electron_injection_energy_eV)
initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
initial_air_density_n_value = Float64(initial_air_density_n)

# Ejecutar la simulación completa con los mejores parámetros
(temperatures_history_julia, avg_temps_history_julia, density_history_julia,
 energy_deposition_history_julia, elastic_energy_deposition_history_julia, final_step, reached_target_temp,
 accumulated_input_energy, efficiency_history_julia, avg_efficiency_julia, detailed_data) = run_pic_simulation(
    initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass, initial_electron_velocity, verbose=true
)

# La eficiencia de simulación reportada ahora es avg_efficiency_julia
efficiency_simulation = avg_efficiency_julia

# Calcular métricas finales de eficiencia (manteniendo el cálculo de eficiencia total para referencia aunque no se use como principal)
total_input_energy = accumulated_input_energy
initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n_value * cell_volume * TOTAL_CELLS
final_internal_energy = (3/2) * k_b * avg_temps_history_julia[end] * initial_air_density_n_value * cell_volume * TOTAL_CELLS
increase_internal_energy = final_internal_energy - initial_internal_energy
efficiency_simulation_total_energy = (increase_internal_energy / total_input_energy) * 100

# --- Resultados y Visualización ---
time_points = LinRange(0, final_step * dt * 1e6, final_step+1)

# Graficar Temperatura Promedio vs Tiempo
p1 = plot_temperature_vs_time(time_points, avg_temps_history_julia, target_temperature)
savefig(p1, "temperature_vs_time.png")
display(p1)

# Graficar Eficiencia vs Tiempo
p4 = plot_efficiency_vs_time(time_points[2:end], efficiency_history_julia) # Time points adjusted to match efficiency history length
savefig(p4, "efficiency_vs_time.png")
display(p4)

final_density_grid = density_history_julia[end]
final_temperature_grid = temperatures_history_julia[end]
z_slice_index = num_z_cells ÷ 2

# Heatmap de Densidad (slice)
p2 = heatmap_density_slice(x_grid, y_grid, final_density_grid, z_slice_index)
savefig(p2, "density_heatmap.png")
display(p2)

# Heatmap de Temperatura (slice)
p3 = heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, z_slice_index)
savefig(p3, "temperature_heatmap.png")
display(p3)

# --- Generar Reporte Final ---
generate_report("simulation_report.txt", initial_temperature, initial_pressure, electron_injection_energy_eV, magnetic_field_strength, dt, simulation_time, simulated_electrons_per_step, efficiency_simulation, increase_internal_energy, total_input_energy, final_step, reached_target_temp, avg_temps_history_julia, avg_efficiency_julia) # Pass avg_efficiency_julia which is now the main efficiency

# Preparar los parámetros de simulación para exportación
simulation_params = Dict(
    "initial_temperature" => initial_temperature,
    "target_temperature" => target_temperature,
    "initial_pressure" => initial_pressure,
    "electron_energy_eV" => electron_injection_energy_eV,
    "magnetic_field_strength" => magnetic_field_strength,
    "dt" => dt,
    "chamber_width" => chamber_width,
    "chamber_length" => chamber_length,
    "chamber_height" => chamber_height,
    "avg_efficiency" => avg_efficiency_julia
)

# Exportar datos a CSV para visualización externa
using CSV
export_simulation_data_to_csv(
    time_points,
    avg_temps_history_julia,
    efficiency_history_julia,
    final_density_grid,
    final_temperature_grid,
    x_grid, y_grid, z_grid,
    simulation_params,
    detailed_data,
    results_df,
    best_params
)


println("\nSimulación completada. Resultados, plots y reporte guardados.")