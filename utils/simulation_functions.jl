using Statistics
using Random
using DataFrames

# --- Función Principal de Simulación (modificada para calcular eficiencia por paso) ---
function run_pic_simulation(initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electric_field, electron_charge, electron_mass, 
    initial_electron_velocity; verbose=true, max_steps_override=nothing)

    # Acceso a variables globales (asegúrate que estén definidas)
    global target_temperature, max_steps, k_b, cell_volume, TOTAL_CELLS
    global chamber_width, chamber_length, chamber_height
    global x_grid, y_grid, z_grid # Necesarios para calculate_grid_density, deposit_energy
    global particle_weight # Necesario para escalar energía depositada
    global monte_carlo_collision # Necesario para llamar la función de colisión
    global apply_rectangular_boundary_conditions # Necesario para las fronteras
    global limit_electron_energy! # Para limitar energía
    global calculate_plasma_conductivity # Para calcular sigma

    # --- Inicialización ---
    magnetic_field_strength = norm(magnetic_field) # Usado en check_timestep y conductividad
    is_stable = check_timestep_validity(dt, magnetic_field_strength, electron_mass, electron_charge)
    if !is_stable && verbose
        println("Continuando con timestep potencialmente inestable...")
    end

    # Historiales
    temperatures_history = [copy(initial_temperature_grid)]
    avg_temps_history = [Statistics.mean(initial_temperature_grid)]
    density_history = [calculate_grid_density(initial_positions, x_grid, y_grid, z_grid)] # Asume calculate_grid_density existe
    energy_deposition_history = [zeros(size(initial_temperature_grid))] # Historial de depósito inelástico
    elastic_energy_deposition_history = [zeros(size(initial_temperature_grid))] # Historial de depósito elástico
    efficiency_history = Float64[] # Eficiencia por paso

    # Datos detallados
    electron_count_history = [size(initial_positions, 1)]
    inelastic_energy_history = Float64[] # Energía inelástica transferida *por paso* (escalada)
    elastic_energy_history = Float64[]   # Energía elástica transferida *por paso* (escalada)
    total_energy_transfer_history = Float64[] # Suma de las dos anteriores
    input_energy_history = Float64[] # Energía cinética inyectada *por paso* (escalada)
    conductivity_history = Float64[] # Conductividad estimada *por paso*

    # Variables de estado
    positions = copy(initial_positions)
    velocities = copy(initial_velocities)
    temperature_grid = copy(initial_temperature_grid)
    rng = MersenneTwister(0) # O usar un RNG global

    current_time = 0.0
    step = 0
    final_step = 0
    reached_target_temp = false
    accumulated_input_energy = 0.0 # Energía total inyectada acumulada (escalada)

    # Tracking de vida de electrones
    electron_creation_times = Float64[] # Tiempo de creación para los electrones activos
    electron_lifetimes = Float64[] # Tiempos de vida de electrones eliminados
    # NOTA: El tracking por ID se omitió aquí para simplificar, pero sería más robusto

    local_max_steps = isnothing(max_steps_override) ? max_steps : max_steps_override

    # Energía interna inicial para cálculo de eficiencia
    initial_temperature = Statistics.mean(initial_temperature_grid) # Tomar la T inicial promedio
    initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n * cell_volume * TOTAL_CELLS
    previous_internal_energy = initial_internal_energy # Energía interna al inicio del paso anterior

    # --- Bucle Principal ---
    while avg_temps_history[end] < target_temperature && step < local_max_steps
        step += 1
        current_time += dt

        if verbose
            println("\n--- Simulación Paso $(step), Tiempo = $(round(current_time * 1e6, digits=3)) µs ---")
            println("Temperatura Promedio Actual: $(round(avg_temps_history[end], digits=1)) K")
            println("Nº Electrones Activos (inicio paso): $(size(positions, 1))")
        end

        # 1. Inyectar Nuevos Electrones
        new_positions, new_velocities = initialize_electrons(simulated_electrons_per_step, chamber_width, chamber_length, initial_electron_velocity)
        append!(electron_creation_times, fill(current_time, size(new_positions, 1))) # Registrar tiempo creación

        # Calcular energía cinética inyectada en este paso (escalada por peso)
        step_input_energy = sum(electron_energy_from_velocity.(sqrt.(sum(new_velocities.^2, dims=2)))) * particle_weight
        accumulated_input_energy += step_input_energy
        push!(input_energy_history, step_input_energy)

        if verbose
            println("Inyectando $(size(new_positions, 1)) electrones simulados.")
            println("Energía Cinética Inyectada (este paso): $(round(step_input_energy, sigdigits=3)) J")
        end

        # Combinar electrones viejos y nuevos
        positions = vcat(positions, new_positions)
        velocities = vcat(velocities, new_velocities)
        push!(electron_count_history, size(positions, 1)) # Contar después de añadir

        # Guardar energía interna al inicio del paso para calcular ΔU_gas
        current_avg_temp = Statistics.mean(temperature_grid)
        internal_energy_start_step = (3/2) * k_b * current_avg_temp * initial_air_density_n * cell_volume * TOTAL_CELLS

        # 2. Mover Electrones (Fuerza de Lorentz: E + v x B)
        # !!! MODIFICADO: Se pasa electric_field a move_electrons !!!
        positions, velocities = move_electrons(
            positions,
            velocities,
            dt,
            magnetic_field,   # Campo B global
            electric_field,   # Campo E global (nuevo)
            electron_charge,
            electron_mass
        )

        # 3. Aplicar Condiciones de Frontera y Registrar Pérdidas
        prev_count = size(positions, 1)
        positions, velocities = apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)
        current_count = size(positions, 1)

        if current_count < prev_count
            num_lost = prev_count - current_count
            if verbose
                 println("$(num_lost) electrones eliminados en fronteras.")
            end
            # Registrar tiempos de vida (simplificado: asume que los primeros son los perdidos)
            lost_times = electron_creation_times[1:num_lost]
            lifetimes_step = current_time .- lost_times
            append!(electron_lifetimes, lifetimes_step)
            # Eliminar tiempos de creación de los perdidos
            deleteat!(electron_creation_times, 1:num_lost)
        end

        # 4. Calcular Conductividad Estimada (antes de colisiones para usar T actual)
        current_e_density_grid = calculate_grid_density(positions, x_grid, y_grid, z_grid)
        avg_e_density = mean(current_e_density_grid) / cell_volume # Densidad numérica promedio
        current_pressure = initial_air_density_n * k_b * current_avg_temp # Presión actual estimada
        # Asegurar valores no nulos para evitar errores en conductividad
        safe_e_density = max(avg_e_density, 1e-5) # Densidad mínima pequeña
        safe_temp = max(current_avg_temp, 1.0) # Temperatura mínima pequeña
        safe_pressure = max(current_pressure, 1e-3) # Presión mínima pequeña

        sigma = calculate_plasma_conductivity(
            safe_e_density,
            safe_temp,
            safe_pressure,
            magnetic_field_strength # Usa la magnitud de B
        )
        push!(conductivity_history, sigma)
        if verbose && step % 10 == 0 # Muestra cada 10 pasos
             println("Conductividad estimada σ: $(round(sigma, sigdigits=3)) S/m (n_e ≈ $(round(avg_e_density, sigdigits=3)) m⁻³)")
        end


        # 5. Colisiones Monte Carlo
        if size(positions, 1) > 0 # Solo si quedan electrones
            # Nota: monte_carlo_collision debe devolver las energías transferidas *por partícula*
            positions, velocities, rng, inelastic_transfer_particle, collided_flags, _, elastic_transfer_particle = monte_carlo_collision(
                positions, velocities, initial_air_density_n, dt, rng, 1.0 # Eficiencia 1.0 aquí
            )

            # Limitar energía después de colisiones si es necesario
            velocities = limit_electron_energy!(velocities)

            # Calcular energía total transferida en el paso (escalada por peso)
            inelastic_energy_step_weighted = sum(inelastic_transfer_particle) * particle_weight
            elastic_energy_step_weighted = sum(elastic_transfer_particle) * particle_weight
            total_transfer_step_weighted = inelastic_energy_step_weighted + elastic_energy_step_weighted

            push!(inelastic_energy_history, inelastic_energy_step_weighted)
            push!(elastic_energy_history, elastic_energy_step_weighted)
            push!(total_energy_transfer_history, total_transfer_step_weighted)

            if verbose
                println("Energía Transferida (Colisiones): Inelástica=$(round(inelastic_energy_step_weighted, sigdigits=3)) J, Elástica=$(round(elastic_energy_step_weighted, sigdigits=3)) J")
            end

            # 6. Depositar Energía en la Malla
            energy_deposition_grid_step = deposit_energy(positions, inelastic_transfer_particle .* particle_weight, x_grid, y_grid, z_grid)
            elastic_energy_deposition_grid_step = deposit_energy(positions, elastic_transfer_particle .* particle_weight, x_grid, y_grid, z_grid)

            # 7. Actualizar Temperatura de la Malla
            temperature_grid = update_temperature_grid(temperature_grid, energy_deposition_grid_step, elastic_energy_deposition_grid_step, initial_air_density_n, cell_volume) # Asume update_temperature_grid existe

        else
            # Si no hay electrones, no hay transferencia ni depósito
             push!(inelastic_energy_history, 0.0)
             push!(elastic_energy_history, 0.0)
             push!(total_energy_transfer_history, 0.0)
             energy_deposition_grid_step = zeros(size(temperature_grid))
             elastic_energy_deposition_grid_step = zeros(size(temperature_grid))
             if verbose
                 println("No quedan electrones para colisiones o depósito.")
             end
        end

        # 8. Calcular Eficiencia del Paso
        # Energía ganada por el gas en este paso
        internal_energy_end_step = (3/2) * k_b * Statistics.mean(temperature_grid) * initial_air_density_n * cell_volume * TOTAL_CELLS
        energy_gained_by_gas = internal_energy_end_step - internal_energy_start_step

        # Eficiencia = (Energía ganada por gas) / (Energía inyectada)
        step_efficiency = 0.0
        if step_input_energy > 1e-20 # Evitar división por cero
            # Asegurar que la ganancia no exceda la entrada (podría pasar por errores numéricos)
            actual_energy_transfer = min(energy_gained_by_gas, step_input_energy)
            if energy_gained_by_gas > step_input_energy + 1e-9 # Pequeña tolerancia
                 if verbose
                    println("⚠️ Advertencia: Ganancia de energía del gas ($(energy_gained_by_gas) J) > Entrada ($(step_input_energy) J). Eficiencia limitada a 100%.")
                 end
                 actual_energy_transfer = step_input_energy
            elseif energy_gained_by_gas < 0.0 && verbose # No debería perder energía si la entrada es positiva
                 println("⚠️ Advertencia: El gas perdió energía ($(energy_gained_by_gas) J) a pesar de la entrada de energía.")
                 actual_energy_transfer = 0.0
            end

            step_efficiency = (actual_energy_transfer / step_input_energy) * 100.0
        end
        push!(efficiency_history, step_efficiency)

        if verbose
            println("Energía Ganada por Gas (ΔU): $(round(energy_gained_by_gas, sigdigits=3)) J")
            println("Eficiencia del Paso: $(round(step_efficiency, digits=2)) %")
        end

        # 9. Guardar Estado para el Siguiente Paso/Historial
        push!(temperatures_history, copy(temperature_grid))
        push!(avg_temps_history, Statistics.mean(temperature_grid))
        # Guardar densidad y depósitos *después* de que ocurrieron en el paso
        push!(density_history, calculate_grid_density(positions, x_grid, y_grid, z_grid))
        push!(energy_deposition_history, energy_deposition_grid_step) # Depósito inelástico del paso
        push!(elastic_energy_deposition_history, elastic_energy_deposition_grid_step) # Depósito elástico del paso

        # 10. Comprobar Condición de Parada
        if avg_temps_history[end] >= target_temperature
            reached_target_temp = true
            final_step = step
            if verbose
                println("\n🏁 Temperatura objetivo alcanzada en el paso $(final_step)! Temperatura final: $(round(avg_temps_history[end], digits=1)) K")
            end
            break # Salir del bucle while
        end

        # Si se alcanza el máximo de pasos sin llegar a la T objetivo
        if step >= local_max_steps
            final_step = step
            reached_target_temp = false
            if verbose
                println("\n🏁 Máximo número de pasos ($(final_step)) alcanzado antes de la temperatura objetivo. Temperatura final: $(round(avg_temps_history[end], digits=1)) K")
            end
            break # Salir del bucle while
        end

    end # Fin del bucle while

    # --- Cálculos Post-Simulación ---
    if final_step == 0 # Si el bucle no corrió ni una vez
        final_step = step
    end

    # Calcular tiempo promedio de vida
    avg_electron_lifetime = if !isempty(electron_lifetimes)
        mean(electron_lifetimes)
    elseif !isempty(electron_creation_times) # Si quedaron electrones pero ninguno se perdió
        mean(current_time .- electron_creation_times) # Vida promedio de los que quedan
         if verbose; println("⚠️ Ningún electrón fue eliminado; avg_lifetime basado en los restantes."); end
         current_time # O simplemente el tiempo total como fallback
    else # Si no hubo electrones o ninguno se perdió
        0.0 # O NaN? Depende de cómo quieras manejarlo
    end

    # Calcular eficiencia promedio
    avg_efficiency = isempty(efficiency_history) ? 0.0 : Statistics.mean(filter(isfinite, efficiency_history)) # Filtra NaNs/Infs si ocurren

    if verbose
        println("\n--- Resumen Final Simulación ---")
        println("Pasos totales: $(final_step)")
        println("Tiempo total simulado: $(round(final_step * dt * 1e6, digits=3)) µs")
        println("Temperatura final promedio: $(round(avg_temps_history[end], digits=1)) K")
        println("¿Alcanzó T objetivo?: $(reached_target_temp)")
        println("Energía total inyectada: $(round(accumulated_input_energy, sigdigits=3)) J")
        println("Eficiencia promedio por paso: $(round(avg_efficiency, digits=2)) %")
        println("Tiempo de vida promedio e⁻: $(round(avg_electron_lifetime * 1e9, digits=3)) ns")
    end

    # Recopilar datos detallados
    detailed_data = Dict(
        "electron_count_history" => electron_count_history,
        "inelastic_energy_history" => inelastic_energy_history,
        "elastic_energy_history" => elastic_energy_history,
        "total_energy_transfer_history" => total_energy_transfer_history,
        "input_energy_history" => input_energy_history
        # Podrías añadir más datos aquí si los necesitaras
    )

    # --- Retorno ---
    # Asegúrate que el orden coincida con cómo se usa en el script principal
    return temperatures_history, avg_temps_history, density_history,
           energy_deposition_history, elastic_energy_deposition_history, final_step, reached_target_temp,
           accumulated_input_energy, efficiency_history, avg_efficiency, detailed_data, avg_electron_lifetime, conductivity_history
end

# Modificar estimate_efficiency para retornar avg_efficiency_julia como eficiencia principal
function estimate_efficiency(electron_injection_energy_eV, initial_pressure, magnetic_field_strength; 
        max_steps_override=nothing, 
        initial_electron_velocity=electron_velocity_from_energy(electron_injection_energy_eV))
    # Configurar campo magnético (axial en z)
    magnetic_field = [0.0, 0.0, magnetic_field_strength]

    # Inicializar parámetros de simulación
    initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature)
    initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
    initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
    initial_air_density_n_value = Float64(initial_air_density_n)

    # Ejecutar simulación completa (modo no-verbose para búsqueda de parámetros)
    (temperatures_history, avg_temps_history, density_history,
    energy_deposition_history, elastic_energy_deposition_history, final_step, 
    reached_target_temp, accumulated_input_energy, efficiency_history, 
    avg_efficiency, detailed_data, avg_electron_lifetime, 
    conductivity_history) = run_pic_simulation(
    initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass, initial_electron_velocity, 
    verbose=false, max_steps_override=max_steps_override
    )

    # Calcular métricas adicionales
    final_temperature = avg_temps_history[end]

    # Tasa de cambio de conductividad (Δσ/Δt)
    conductivity_change_rate = if length(conductivity_history) > 1
    (conductivity_history[end] - conductivity_history[1]) / (final_step * dt)  # [S/(m·s)]
    else
    0.0
    end

    # Eficiencia como métrica principal
    efficiency_simulation = avg_efficiency

    return (efficiency_simulation, final_temperature, final_step, reached_target_temp,
    avg_efficiency, avg_electron_lifetime, conductivity_change_rate)
end

function calculate_plasma_conductivity(electron_density, temperature_e, pressure, magnetic_field_strength)
    # Constantes
    e = 1.60218e-19  # Carga del electrón (C)
    m_e = 9.109e-31   # Masa del electrón (kg)
    
    # Densidad numérica del gas
    n_gas = pressure / (k_b * temperature_e)  # [m^-3]
    
    # Calcular energía térmica promedio en eV
    E_avg_eV = (3/2) * k_b * temperature_e / e
    
    # Calcular sección eficaz total promedio ponderada por composición del aire
    total_cross_section = 0.0
    for gas_name in keys(air_composition)
        gas_fraction = air_composition[gas_name]["fraction"]
        cross_section_func = air_composition[gas_name]["total_cross_section_func"]
        total_cross_section += gas_fraction * cross_section_func(E_avg_eV)
    end
    
    # Velocidad térmica electrónica
    v_th = sqrt(3 * k_b * temperature_e / m_e)
    
    # Frecuencia de colisiones electrón-neutro (ahora usando secciones eficaces reales)
    ν_en = n_gas * total_cross_section * v_th
    
    # Frecuencia de ciclotrón electrón
    ω_ce = (e * magnetic_field_strength) / m_e  # [rad/s]
    
    # Conductividad usando modelo de Drude con campo magnético
    σ = (e^2 * electron_density * ν_en) / (m_e * (ν_en^2 + ω_ce^2))  # [S/m]
    
    return σ
end

function parameter_search() # <-- Removido el argumento de potencial fijo

    # Acceso a variables globales (asegúrate que estén definidas)
    global initial_temperature, chamber_width, chamber_length, chamber_height
    global num_x_cells, num_y_cells, num_z_cells, TOTAL_CELLS
    global air_composition, dt, simulated_electrons_per_step
    global electron_charge, electron_mass, k_b, cell_volume, particle_weight
    global calculate_air_density_n, initialize_electrons, calculate_plasma_conductivity
    global electron_velocity_from_energy # Asegurarse que esta esté accesible

    # --- Definir Rangos de Búsqueda ---
    # (Ajusta estos rangos según necesites)
    electron_energies = [25.0, 50.0, 100.0, 200.0] # eV
    pressures = [1e6, 3e6]                   # Pa
    magnetic_fields = [0.5, 1.5]             # Tesla
    # !!! NUEVO: Rango para el potencial atractivo !!!
    attractive_potentials = [0.0, 25.0, 50.0, 100.0] # Volts (0.0 = sin campo E externo)

    println("--- Iniciando Búsqueda de Parámetros ---")
    println("Rangos de Búsqueda:")
    println("  Energía (eV): $electron_energies")
    println("  Presión (MPa): $(pressures ./ 1e6)")
    println("  Campo B (T): $magnetic_fields")
    println("  Potencial Atractivo (V): $attractive_potentials") # Mostrar nuevo rango

    # Número reducido de pasos para la búsqueda rápida
    search_max_steps = 250 # Ajusta según el tiempo disponible

    # Inicializar DataFrame para almacenar resultados
    # !!! NUEVO: Añadida columna AttractivePotential !!!
    results = DataFrame(
        ElectronEnergy = Float64[],
        Pressure = Float64[],
        MagneticField = Float64[],
        AttractivePotential = Float64[], # <-- Nueva columna
        FinalEfficiency = Float64[],
        AvgLifetime = Float64[],
        FinalConductivity = Float64[],
        Steps = Int[],
        ReachedTarget = Bool[],
        SimulationTime_us = Float64[],
        HeatingRate = Float64[]
    )

    # Realizar búsqueda en cuadrícula (4 bucles ahora)
    best_efficiency = -Inf # Inicia con -infinito para encontrar el máximo
    # !!! NUEVO: best_params ahora tiene 4 elementos !!!
    best_params = (0.0, 0.0, 0.0, 0.0) # (Energy, Pressure, Field, Potential)
    total_combinations = length(electron_energies) * length(pressures) * length(magnetic_fields) * length(attractive_potentials)
    count = 0

    for energy in electron_energies
        for pressure in pressures
            for field in magnetic_fields
                for potential in attractive_potentials # <-- NUEVO BUCLE INTERNO
                    count += 1
                    # !!! NUEVO: Mostrar el potencial actual !!!
                    println("\n($(count)/$(total_combinations)) Evaluando: E=$(energy)eV, P=$(pressure/1e6)MPa, B=$(field)T, V_attr=$(potential)V")

                    # --- Configuración para esta iteración ---
                    magnetic_field = [0.0, 0.0, field] # Campo B
                    # !!! NUEVO: Calcular campo E DENTRO del bucle de potencial !!!
                    if chamber_height <= 0.0
                        error("chamber_height debe ser positivo.")
                    end
                    electric_field = [0.0, 0.0, -potential / chamber_height] # Campo E

                    initial_air_density_n = calculate_air_density_n(pressure, initial_temperature)
                    initial_velocity = electron_velocity_from_energy(energy)
                    initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_velocity)
                    initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))

                    # --- Ejecutar Simulación ---
                    # La llamada a run_pic_simulation ya acepta electric_field
                    (temperatures_history, avg_temps_history, density_history,
                    energy_deposition_history, elastic_energy_deposition_history, final_step,
                    reached_target, accumulated_input_energy, efficiency_history,
                    avg_efficiency, detailed_data, avg_lifetime,
                    conductivity_history) = run_pic_simulation(
                        initial_positions, initial_velocities, initial_temperature_grid,
                        initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
                        magnetic_field,   # Campo B (variable)
                        electric_field,   # Campo E (variable ahora)
                        electron_charge,
                        electron_mass,
                        initial_velocity,
                        verbose=false, # Modo silencioso
                        max_steps_override=search_max_steps
                    )

                    # --- Calcular Métricas Adicionales ---
                    final_temperature = avg_temps_history[end]
                    simulation_time_s = final_step * dt
                    simulation_time_us = simulation_time_s * 1e6
                    heating_rate = 0.0
                    if simulation_time_us > 1e-9
                        heating_rate = (final_temperature - initial_temperature) / simulation_time_us
                    end
                    final_conductivity = isempty(conductivity_history) ? 0.0 : conductivity_history[end]

                    # --- Mostrar y Guardar Resultados ---
                    println("  Resultados -> Eficiencia: $(round(avg_efficiency, digits=2))%, Vida e⁻: $(round(avg_lifetime*1e9, digits=2)) ns, σ_fin: $(round(final_conductivity, sigdigits=3)) S/m, Pasos: $final_step")

                    # !!! NUEVO: Añadir 'potential' al push! !!!
                    push!(results, (
                        energy,
                        pressure,
                        field,
                        potential,           # <-- Guardar potencial
                        avg_efficiency,      # FinalEfficiency
                        avg_lifetime,        # AvgLifetime
                        final_conductivity,  # FinalConductivity
                        final_step,          # Steps
                        reached_target,      # ReachedTarget
                        simulation_time_us,  # SimulationTime_us
                        heating_rate         # HeatingRate
                    ))

                    # --- Actualizar Mejores Parámetros ---
                    if avg_efficiency > best_efficiency
                        best_efficiency = avg_efficiency
                        # !!! NUEVO: Guardar los 4 parámetros óptimos !!!
                        best_params = (energy, pressure, field, potential)
                        println("  ✨ ¡NUEVO MEJOR RESULTADO!")
                    end
                end # Bucle potential
            end # Bucle field
        end # Bucle pressure
    end # Bucle energy

    println("\n--- Búsqueda de Parámetros Completada ---")

    # Ordenar resultados por la eficiencia final (descendente)
    sort!(results, :FinalEfficiency, rev=true)

    # Mostrar resumen de mejores parámetros
    if best_efficiency > -Inf
        println("Mejores Parámetros Encontrados:")
        println("  Energía Electrón:      $(best_params[1]) eV")
        println("  Presión Inicial:       $(best_params[2]/1e6) MPa")
        println("  Campo Magnético:       $(best_params[3]) T")
        println("  Potencial Atractivo:   $(best_params[4]) V")
        println("  Mejor Eficiencia Promedio: $(round(best_efficiency, digits= 2))%")
    else
        println("No se encontraron resultados válidos en la búsqueda.")
    end

    # El retorno sigue siendo el DataFrame, la tupla de mejores params, y la mejor eficiencia
    # Pero best_params ahora es una tupla de 4 elementos
    return results, best_params, best_efficiency
end