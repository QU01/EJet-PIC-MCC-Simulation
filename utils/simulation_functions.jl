using Statistics
using Random
using DataFrames

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
    final_step = 0 # Initialize final_step here
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
    final_step = step # Assign the final value of step after the loop

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
           energy_deposition_history, elastic_energy_deposition_history, final_step, reached_target_temp,
           accumulated_input_energy, efficiency_history, avg_efficiency, detailed_data
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
    electron_energies = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 500.0, 750.0, 1000.0]  # eV
    pressures = [1e6, 2e6, 3e6, 4e6]  # Pa
    magnetic_fields = [0.5, 1.0, 1.5, 2.0]  # Tesla

    # Número reducido de pasos para la búsqueda rápida
    search_max_steps = 200

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