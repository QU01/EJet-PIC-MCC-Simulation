# ---------------------------------------------------------------------------
# ARCHIVO: collisions.jl (Versión con aceleración GPU)
# ---------------------------------------------------------------------------

using Random
using LinearAlgebra

# GPU: Importar la librería de CUDA y definir una bandera global para su uso.
using CUDA
const USE_GPU = CUDA.functional() # Se pondrá a 'true' si tienes una GPU NVIDIA.

# --- Constantes Físicas (necesarias en este archivo) ---
const ELECTRON_MASS = 9.109e-31
const ELECTRON_CHARGE_CONST = -1.60218e-19
const EXCITATION_ENERGY_LOSS_EV = 3.0
const EXCITATION_ENERGY_LOSS_JOULES = EXCITATION_ENERGY_LOSS_EV * 1.60218e-19

# --- Funciones Auxiliares para CPU ---
# (Se necesitan para la versión CPU y para cálculos de diagnóstico)

function electron_energy_from_velocity(velocity_magnitude)
    return 0.5 * ELECTRON_MASS * (velocity_magnitude^2)
end

function estimate_excitation_cross_section(total_cross_section, ionization_cross_section, energy_eV, ionization_energy_eV)
    cross_section_diff = total_cross_section - ionization_cross_section
    excitation_cross_section = max(0.0, cross_section_diff)
    energy_above_threshold = max(0.0, energy_eV - ionization_energy_eV)
    excitation_fraction = exp(-0.1 * energy_above_threshold)
    excitation_cross_section = excitation_cross_section * excitation_fraction
    return excitation_cross_section
end


# ===========================================================================
# SECCIÓN DE ACELERACIÓN POR GPU
# ===========================================================================

# GPU: Función de interpolación lineal simple que puede ejecutarse en la GPU.
# Es una "device function" y reemplaza a `Interpolations.jl` dentro del kernel.
@inline function device_interpolate(x_vals, y_vals, x_query)
    len = length(x_vals)
    # Manejar extrapolación (comportamiento "Flat")
    if x_query <= x_vals[1]; return y_vals[1]; end
    if x_query >= x_vals[end]; return y_vals[end]; end

    # Búsqueda lineal (suficiente y simple para la GPU)
    idx = 1
    while idx < len && x_vals[idx+1] < x_query
        idx += 1
    end
    
    x1, y1 = x_vals[idx], y_vals[idx]
    x2, y2 = x_vals[idx+1], y_vals[idx+1]
    
    return y1 + (x_query - x1) * (y2 - y1) / (x2 - x1)
end

# GPU: Versión "device" del modelo de excitación.
@inline function device_estimate_excitation_cross_section(total_cs, ion_cs, energy_eV, ion_energy_eV)
    diff = max(0.0, total_cs - ion_cs)
    energy_above = max(0.0, energy_eV - ion_energy_eV)
    fraction = CUDA.exp(-0.1 * energy_above) # Usar la función exp de CUDA
    return diff * fraction
end

# GPU: Versión "device" para generar una dirección 3D aleatoria.
@inline function device_rand_direction(r1, r2, r3)
    norm_val = CUDA.sqrt(r1*r1 + r2*r2 + r3*r3)
    if norm_val > 1e-9
        inv_norm = 1.0 / norm_val
        return (r1 * inv_norm, r2 * inv_norm, r3 * inv_norm)
    end
    return (0.0, 0.0, 1.0)
end

# GPU: El KERNEL. Esta es la función que se ejecuta en paralelo para cada partícula.
function monte_carlo_collision_kernel!(
        # Arrays de E/S que se modifican
        velocities, inelastic_energy_transfer, elastic_energy_transfer,
        collided_flags, collision_energy_transfers_eV,
        # Parámetros de la simulación
        air_density_n, dt,
        # Datos de los gases (aplanados en arrays)
        gas_masses, gas_fractions, gas_ionization_energies_eV,
        # Datos de sección eficaz (punteros a CuArrays)
        n2_E, n2_total_cs, n2_ion_cs, o2_E, o2_total_cs, o2_ion_cs,
        # Números aleatorios pre-generados
        rand_numbers)

    # Índice global del hilo, que corresponde a una partícula
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > size(velocities, 1); return; end

    # 1. Calcular energía de la partícula
    vx, vy, vz = velocities[i, 1], velocities[i, 2], velocities[i, 3]
    v_mag_sq = vx*vx + vy*vy + vz*vz
    v_magnitude = CUDA.sqrt(v_mag_sq)
    electron_energy_joules = 0.5 * ELECTRON_MASS * v_mag_sq
    if electron_energy_joules < 1e-25; return; end
    electron_energy_eV = electron_energy_joules / abs(ELECTRON_CHARGE_CONST)

    # 2. Calcular probabilidad de colisión
    # Gas 1: N2 (índice 1)
    n2_frac = gas_fractions[1]
    n2_total_cs_val = device_interpolate(n2_E, n2_total_cs, electron_energy_eV)
    n2_prob = 1.0 - CUDA.exp(-(air_density_n * n2_frac) * n2_total_cs_val * v_magnitude * dt)
    # Gas 2: O2 (índice 2)
    o2_frac = gas_fractions[2]
    o2_total_cs_val = device_interpolate(o2_E, o2_total_cs, electron_energy_eV)
    o2_prob = 1.0 - CUDA.exp(-(air_density_n * o2_frac) * o2_total_cs_val * v_magnitude * dt)
    
    total_collision_prob = min(0.95, n2_prob * n2_frac + o2_prob * o2_frac)

    # 3. Decidir si colisiona
    if rand_numbers[i, 1] < total_collision_prob
        # 4. Seleccionar el gas
        selected_gas_idx = 2 # Por defecto O2
        # Normalizar probabilidades relativas para la selección
        total_gas_prob = n2_prob * n2_frac + o2_prob * o2_frac
        if total_gas_prob > 1e-9 && rand_numbers[i, 2] <= (n2_prob * n2_frac / total_gas_prob)
            selected_gas_idx = 1 # N2
        end

        # 5. Obtener propiedades y secciones eficaces del gas seleccionado
        mass = gas_masses[selected_gas_idx]
        ionization_energy_eV = gas_ionization_energies_eV[selected_gas_idx]
        ionization_energy_joules = ionization_energy_eV * abs(ELECTRON_CHARGE_CONST)
        
        total_cs, ion_cs = 0.0, 0.0
        if selected_gas_idx == 1 # N2
            total_cs = n2_total_cs_val
            ion_cs = device_interpolate(n2_E, n2_ion_cs, electron_energy_eV)
        else # O2
            total_cs = o2_total_cs_val
            ion_cs = device_interpolate(o2_E, o2_ion_cs, electron_energy_eV)
        end
        excite_cs = device_estimate_excitation_cross_section(total_cs, ion_cs, electron_energy_eV, ionization_energy_eV)
        elastic_cs = max(0.0, total_cs - ion_cs - excite_cs)

        # 6. Decidir tipo de colisión
        p_total = ion_cs + excite_cs + elastic_cs
        p_ionize, p_excite = 0.0, 0.0
        if p_total > 1e-30
            p_ionize = ion_cs / p_total
            p_excite = excite_cs / p_total
        end

        collision_rand = rand_numbers[i, 3]
        energy_loss_joules = 0.0
        max_transfer = 0.95 * electron_energy_joules

        if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
            energy_loss_joules = min(ionization_energy_joules, max_transfer)
            inelastic_energy_transfer[i] = energy_loss_joules
        elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > EXCITATION_ENERGY_LOSS_JOULES
            energy_loss_joules = min(EXCITATION_ENERGY_LOSS_JOULES, max_transfer)
            inelastic_energy_transfer[i] = energy_loss_joules
        else
            fraction_energy_elastic = (2 * ELECTRON_MASS / mass)
            energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
            elastic_energy_transfer[i] = energy_loss_joules
        end

        # 7. Actualizar velocidad
        remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
        new_speed = CUDA.sqrt(2 * remaining_energy_joules / ELECTRON_MASS)
        dir_x, dir_y, dir_z = device_rand_direction(rand_numbers[i, 4], rand_numbers[i, 5], rand_numbers[i, 6])
        
        velocities[i, 1] = dir_x * new_speed
        velocities[i, 2] = dir_y * new_speed
        velocities[i, 3] = dir_z * new_speed

        # 8. Registrar resultados
        collided_flags[i] = true
        collision_energy_transfers_eV[i] = energy_loss_joules / abs(ELECTRON_CHARGE_CONST)
    end
    
    return
end

# GPU: Función "envoltorio" que prepara datos y llama al kernel.
function monte_carlo_collision_gpu(positions, velocities, air_density_n, dt, gas_data)
    num_particles = size(velocities, 1)
    if num_particles == 0
        # Devuelve arrays vacíos del tipo correcto (CPU)
        return velocities, Float64[], Float64[]
    end

    # 1. El array de velocidades ya está en la GPU, no es necesario copiarlo.
    vel_d = velocities

    # 2. Crear arrays de resultados en la GPU
    inelastic_d = CUDA.zeros(Float64, num_particles)
    elastic_d = CUDA.zeros(Float64, num_particles)
    collided_d = CUDA.zeros(Bool, num_particles)
    transfers_eV_d = CUDA.zeros(Float64, num_particles)

    # 3. Generar números aleatorios en la GPU
    rand_numbers_d = CUDA.rand(Float64, num_particles, 6)

    # 4. Lanzar el kernel (modifica vel_d en el sitio)
    threads = 256
    blocks = cld(num_particles, threads)
    @cuda threads=threads blocks=blocks monte_carlo_collision_kernel!(
        vel_d, inelastic_d, elastic_d, collided_d, transfers_eV_d,
        air_density_n, dt,
        gas_data.masses, gas_data.fractions, gas_data.ion_energies,
        gas_data.n2_E, gas_data.n2_total_cs, gas_data.n2_ion_cs,
        gas_data.o2_E, gas_data.o2_total_cs, gas_data.o2_ion_cs,
        rand_numbers_d
    )

    # 5. Copiar SOLO los resultados necesarios (energías) de vuelta a la CPU.
    CUDA.synchronize()
    inelastic_transfer = Array(inelastic_d)
    elastic_transfer = Array(elastic_d)

    # Devuelve el array de velocidades de la GPU y los arrays de energía de la CPU.
    return vel_d, inelastic_transfer, elastic_transfer
end


# ===========================================================================
# SECCIÓN DE CÓDIGO ORIGINAL PARA CPU
# ===========================================================================

# CPU: Tu función original, renombrada para claridad.
# He eliminado `rng` y `efficiency` de los argumentos para que coincida mejor.
function monte_carlo_collision_cpu(positions, velocities, air_density_n, dt, air_composition)
    rng = Random.default_rng()
    num_particles = size(velocities, 1)
    if num_particles == 0
        return positions, velocities, Float64[], Bool[], Float64[], Float64[]
    end

    inelastic_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    v_magnitudes = vec(sqrt.(sum(velocities.^2, dims=2)))
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)
    E_e_eV = E_e_joules ./ abs(ELECTRON_CHARGE_CONST)
    
    gas_names_keys = collect(keys(air_composition))

    for i in 1:num_particles
        electron_energy_joules = E_e_joules[i]
        if electron_energy_joules < 1e-25; continue; end

        total_collision_prob = 0.0
        gas_probs = Dict{String, Float64}()
        for gas_name in gas_names_keys
            gas_info = air_composition[gas_name]
            total_cross_section = gas_info["total_cross_section_func"](E_e_eV[i])
            gas_prob = 1.0 - exp(-(air_density_n * gas_info["fraction"]) * total_cross_section * v_magnitudes[i] * dt)
            gas_probs[gas_name] = gas_prob
            total_collision_prob += gas_prob * gas_info["fraction"]
        end
        total_collision_prob = min(total_collision_prob, 0.95)

        if rand(rng) < total_collision_prob
            selected_gas = ""
            rand_val = rand(rng)
            cumulative_prob = 0.0
            norm_factor = sum(gas_probs[g] * air_composition[g]["fraction"] for g in gas_names_keys)
            
            if norm_factor > 1e-9
                for gas_name in gas_names_keys
                    prob_norm = (gas_probs[gas_name] * air_composition[gas_name]["fraction"]) / norm_factor
                    cumulative_prob += prob_norm
                    if rand_val <= cumulative_prob
                        selected_gas = gas_name
                        break
                    end
                end
            end
            if selected_gas == ""; selected_gas = last(gas_names_keys); end
            
            gas_info = air_composition[selected_gas]
            mass = gas_info["mass"]
            ionization_energy_eV = gas_info["ionization_energy_eV"]
            total_cs = gas_info["total_cross_section_func"](E_e_eV[i])
            ion_cs = gas_info["ionization_cross_section_func"](E_e_eV[i])
            excite_cs = estimate_excitation_cross_section(total_cs, ion_cs, E_e_eV[i], ionization_energy_eV)
            elastic_cs = max(0.0, total_cs - ion_cs - excite_cs)
            
            p_total = ion_cs + excite_cs + elastic_cs
            p_ionize, p_excite = 0.0, 0.0
            if p_total > 0
                p_ionize = ion_cs / p_total
                p_excite = excite_cs / p_total
            end

            collision_rand = rand(rng)
            energy_loss_joules = 0.0
            max_transfer = 0.95 * electron_energy_joules
            ionization_energy_joules = ionization_energy_eV * abs(ELECTRON_CHARGE_CONST)
            
            if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
                energy_loss_joules = min(ionization_energy_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > EXCITATION_ENERGY_LOSS_JOULES
                energy_loss_joules = min(EXCITATION_ENERGY_LOSS_JOULES, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            else
                fraction_energy_elastic = (2 * ELECTRON_MASS / mass)
                energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
                elastic_energy_transfer[i] = energy_loss_joules
            end

            remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
            new_speed = sqrt(2 * remaining_energy_joules / ELECTRON_MASS)
            new_direction = normalize(randn(rng, 3))
            velocities_new[i, :] = new_direction .* new_speed
            
            collided_flags[i] = true
            collision_energy_transfers_eV[i] = energy_loss_joules / abs(ELECTRON_CHARGE_CONST)
        end
    end
    
    return velocities_new, inelastic_energy_transfer, elastic_energy_transfer
end


# ===========================================================================
# FUNCIÓN "DESPACHADORA" PRINCIPAL
# ===========================================================================

# Esta es la única función que tu bucle de simulación principal necesitará llamar.
# Elige automáticamente la mejor implementación (GPU o CPU).
function monte_carlo_collision(positions, velocities, air_density_n, dt, air_composition, gpu_gas_data)
    if USE_GPU && size(velocities, 1) > 0
        # La versión GPU ahora devuelve (CuArray, Array, Array)
        return monte_carlo_collision_gpu(positions, velocities, air_density_n, dt, gpu_gas_data)
    else
        # La versión CPU ahora devuelve (Array, Array, Array)
        return monte_carlo_collision_cpu(positions, velocities, air_density_n, dt, air_composition)
    end
end

# Función de diagnóstico que puedes llamar si es necesario
function verify_energy_conservation(initial_velocities, final_velocities, inelastic_transfer, elastic_transfer)
    if isempty(initial_velocities); return; end

    v_mag_initial = vec(sqrt.(sum(initial_velocities.^2, dims=2)))
    energy_initial_J = sum(electron_energy_from_velocity.(v_mag_initial))
    
    v_mag_final = vec(sqrt.(sum(final_velocities.^2, dims=2)))
    energy_final_J = sum(electron_energy_from_velocity.(v_mag_final))

    energy_lost_J = energy_initial_J - energy_final_J
    energy_transferred_J = sum(inelastic_transfer) + sum(elastic_transfer)

    error_rel = abs(energy_lost_J - energy_transferred_J) / max(1e-20, energy_transferred_J)

    if error_rel > 0.01
        println("⚠️ ERROR DE CONSERVACIÓN: ΔE=$(energy_lost_J), E_transf=$(energy_transferred_J), error=$(round(error_rel*100, digits=2))%")
    else
        # Comentado para no llenar la consola
        # println("✓ Conservación de energía verificada (error relativo = $(round(error_rel*100, digits=2))%).")
    end
end