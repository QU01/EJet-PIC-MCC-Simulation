# ---------------------------------------------------------------------------
# FILE: simulation_functions.jl (Main simulation loop and parameter search)
# ---------------------------------------------------------------------------

using Statistics
using Random
using DataFrames
using CUDA

# --- Main Simulation Function ---
# This function now takes structs for parameters and data for better organization.
function run_pic_simulation(params, cpu_data, gpu_data; verbose=true)

    # --- Initialization ---
    use_gpu = params.use_gpu
    nx, ny, nz = size(params.initial_temperature_grid)
    
    # Determine if we are running on CPU or GPU and set up initial arrays accordingly
    if use_gpu
        # Move initial data to the GPU
        positions = CuArray(params.initial_positions)
        velocities = CuArray(params.initial_velocities)
        temperature_grid = CuArray(params.initial_temperature_grid)
        
        # Initialize grids on the GPU
        charge_density_grid = CUDA.zeros(Float64, nx, ny, nz)
    else
        # Keep initial data on the CPU
        positions = params.initial_positions
        velocities = params.initial_velocities
        temperature_grid = params.initial_temperature_grid
        
        # Initialize grids on the CPU
        charge_density_grid = zeros(Float64, nx, ny, nz)
    end

    # Initial field solve (can be on CPU or GPU depending on charge_density_grid type)
    potential_grid = solve_poisson_equation(charge_density_grid, params.x_cell_size, params.y_cell_size, params.z_cell_size, params.anode_voltage)
    Ex, Ey, Ez = calculate_electric_field_from_potential(potential_grid, params.x_cell_size, params.y_cell_size, params.z_cell_size)
    electric_field_grid = ElectricFieldGrid(Ex, Ey, Ez, potential_grid)

    # History trackers (mostly kept on CPU for simplicity, except for grids)
    avg_temps_history = [mean(to_cpu(temperature_grid))] # to_cpu handles both cases
    efficiency_history = Float64[]
    
    # Detailed data for analysis
    detailed_data = Dict(
        "electron_count" => Int[], # Empezar vacío
        "inelastic_energy_J" => Float64[],
        "elastic_energy_J" => Float64[],
        "input_energy_J" => Float64[]
    )
    
    # Animation history (optional, can consume a lot of memory)
    position_history = []
    potential_history = []

    # State variables
    current_time = 0.0
    step = 0
    accumulated_input_energy = 0.0
    electron_creation_times = Float64[]
    electron_lifetimes = Float64[]

    if verbose
        println("Starting PIC simulation on ", use_gpu ? "GPU" : "CPU", "...")
    end

    # --- Main Loop ---
    while avg_temps_history[end] < params.target_temperature && step < params.max_steps
        step += 1
        current_time += params.dt

        # --- 1. Inject New Electrons ---
        # Injection is always done on CPU, then data is moved to GPU if needed.
        new_positions_cpu, new_velocities_cpu = initialize_electrons(
            params.simulated_electrons_per_step, params.chamber_dims, params.initial_electron_velocity
        )
        
        # Combine old and new particles
        if use_gpu
            positions = vcat(positions, CuArray(new_positions_cpu))
            velocities = vcat(velocities, CuArray(new_velocities_cpu))
        else
            positions = vcat(positions, new_positions_cpu)
            velocities = vcat(velocities, new_velocities_cpu)
        end
        
        append!(electron_creation_times, fill(current_time, size(new_positions_cpu, 1)))
        
        # Calculate injected energy for this step
        step_input_energy = sum(electron_energy_from_velocity.(vec(sqrt.(sum(new_velocities_cpu.^2, dims=2))))) * params.particle_weight
        accumulated_input_energy += step_input_energy
        push!(detailed_data["input_energy_J"], step_input_energy)

        if verbose
            println("\n--- Step $(step), Time = $(round(current_time * 1e6, digits=3)) µs ---")
            println("Avg Temp: $(round(avg_temps_history[end], digits=1)) K, Active Electrons: $(size(positions, 1))")
        end

        # --- 2. Move Electrons (Boris Pusher) ---
        # This dispatcher function will call the correct CPU/GPU implementation.
        positions, velocities = move_electrons(
            positions, velocities, params.dt, params.magnetic_field, electric_field_grid,
            params.x_grid, params.y_grid, params.z_grid
        )

        # --- 3. Apply Boundary Conditions ---
        kept_mask_gpu = calculate_inside_chamber_mask(positions, params.chamber_dims)
        kept_mask_cpu = Array(kept_mask_gpu)  # Asegurar conversión explícita

        # Ahora todas las operaciones de filtrado se hacen en la CPU.
        if !all(kept_mask_cpu)
            lost_mask_cpu = .!kept_mask_cpu
            lost_creation_times = electron_creation_times[lost_mask_cpu]
            append!(electron_lifetimes, current_time .- lost_creation_times)
        end

        # Filtramos los arrays de GPU usando la máscara de GPU.
        positions = positions[kept_mask_gpu, :]
        velocities = velocities[kept_mask_gpu, :]

        # Filtramos el array de CPU usando la máscara de CPU.
        electron_creation_times = electron_creation_times[kept_mask_cpu]

        push!(detailed_data["electron_count"], size(positions, 1))

        # --- 4. Field Solve Step ---
        if step % params.field_update_interval == 0
            # a. Deposit charge from particles to grid
            charge_density_grid = calculate_charge_density(
                positions, params.particle_weight, params.x_grid, params.y_grid, params.z_grid, params.cell_volume
            )
            # b. Solve Poisson's equation for potential
            potential_grid = solve_poisson_equation(
                charge_density_grid, params.x_cell_size, params.y_cell_size, params.z_cell_size, params.anode_voltage
            )
            # c. Calculate electric field from potential
            Ex, Ey, Ez = calculate_electric_field_from_potential(
                potential_grid, params.x_cell_size, params.y_cell_size, params.z_cell_size
            )
            electric_field_grid = ElectricFieldGrid(Ex, Ey, Ez, potential_grid)
            
            if params.store_animation_data; push!(potential_history, to_cpu(potential_grid)); end
        end

        # --- 5. Monte Carlo Collisions ---
        if size(positions, 1) > 0
            # La función de colisión devuelve las nuevas velocidades y las transferencias de energía.
            # El tipo de `new_velocities` dependerá de si se usó la CPU o la GPU.
            new_velocities, inelastic_transfer, elastic_transfer = monte_carlo_collision(
                positions, velocities, params.initial_air_density_n, params.dt, cpu_data.air_composition, gpu_data
            )

            # Actualizamos el array de velocidades del bucle principal.
            velocities = new_velocities

            # Energy limiter (se aplica al array de velocidades actualizado, ya sea de CPU o GPU)
            limit_electron_energy!(velocities, params.min_energy_eV, params.max_energy_eV)

            # --- 6. Deposit Energy and Update Temperature ---
            # `positions` no se modifica en la colisión, así que usamos el array filtrado de antes.
            # `inelastic_transfer` y `elastic_transfer` son arrays de CPU, `deposit_energy` los moverá a la GPU si es necesario.
            inelastic_energy_grid = deposit_energy(positions, inelastic_transfer .* params.particle_weight, params.x_grid, params.y_grid, params.z_grid)
            elastic_energy_grid = deposit_energy(positions, elastic_transfer .* params.particle_weight, params.x_grid, params.y_grid, params.z_grid)

            temperature_grid = update_temperature_grid(
                temperature_grid, inelastic_energy_grid, elastic_energy_grid, params.initial_air_density_n, params.cell_volume
            )

            push!(detailed_data["inelastic_energy_J"], sum(inelastic_transfer) * params.particle_weight)
            push!(detailed_data["elastic_energy_J"], sum(elastic_transfer) * params.particle_weight)
        else
            # Si no hay electrones, empujamos ceros para mantener los historiales alineados
            push!(detailed_data["inelastic_energy_J"], 0.0)
            push!(detailed_data["elastic_energy_J"], 0.0)
        end

        # --- 7. Diagnostics and History ---
        current_avg_temp = mean(to_cpu(temperature_grid))
        push!(avg_temps_history, current_avg_temp)
        
        # Calculate step efficiency (optional, can be slow)
        # ... (logic for efficiency calculation) ...

        if params.store_animation_data && step % 10 == 0
            push!(position_history, to_cpu(positions))
        end
        
        # --- 8. Check Stop Condition ---
        if current_avg_temp >= params.target_temperature
            if verbose; println("\n🏁 Target temperature reached at step $(step)!"); end
            break
        end
    end # End of main loop

    # --- Post-Simulation Analysis ---
    final_step = step
    reached_target_temp = avg_temps_history[end] >= params.target_temperature
    
    avg_lifetime = isempty(electron_lifetimes) ? 0.0 : mean(electron_lifetimes)
    avg_efficiency = isempty(efficiency_history) ? 0.0 : mean(filter(isfinite, efficiency_history))
    
    # Final conductivity calculation
    final_density_grid = calculate_grid_density(positions, params.x_grid, params.y_grid, params.z_grid)
    avg_e_density = mean(to_cpu(final_density_grid)) / params.cell_volume
    final_pressure = params.initial_air_density_n * K_B * avg_temps_history[end]
    plasma_conductivity = calculate_plasma_conductivity(
        max(1e-5, avg_e_density), avg_temps_history[end], final_pressure, norm(params.magnetic_field)
    )

    if verbose
        println("\n--- Simulation Finished ---")
        println("Total steps: $(final_step)")
        println("Final avg temperature: $(round(avg_temps_history[end], digits=1)) K")
    end

    # Return results in a structured way
    return (
        final_step = final_step,
        reached_target_temp = reached_target_temp,
        accumulated_input_energy = accumulated_input_energy,
        avg_temps_history = avg_temps_history,
        efficiency_history = efficiency_history,
        avg_efficiency = avg_efficiency,
        avg_electron_lifetime = avg_lifetime,
        plasma_conductivity = plasma_conductivity,
        final_density_grid = final_density_grid,
        final_temperature_grid = temperature_grid,
        detailed_data = detailed_data,
        position_history = position_history,
        potential_history = potential_history
    )
end

# --- Parameter Search Function ---
function parameter_search(search_params, base_params, cpu_data, gpu_data)
    println("--- Starting Parameter Search ---")
    
    results_list = []
    total_combinations = length(search_params.energies) * length(search_params.pressures) * length(search_params.fields) * length(search_params.voltages)
    count = 0

    for energy in search_params.energies, pressure in search_params.pressures, field in search_params.fields, voltage in search_params.voltages
        count += 1
        println("\n($(count)/$(total_combinations)) Evaluating: E=$(energy)eV, P=$(pressure/1e6)MPa, B=$(field)T, V=$(voltage)V")

        # --- CORRECCIÓN AQUÍ ---
        # 1. Calcular los parámetros dependientes para esta iteración
        initial_air_density_n = calculate_air_density_n(pressure, base_params.initial_temperature)
        initial_electron_velocity = electron_velocity_from_energy(energy)
        
        # 2. Crear los arrays de estado inicial para esta iteración
        #    (Empezamos con 0 electrones, se inyectan en el primer paso)
        initial_positions, initial_velocities = initialize_electrons(
            0, base_params.chamber_dims, initial_electron_velocity
        )
        initial_temperature_grid = fill(
            base_params.initial_temperature, 
            (length(base_params.x_grid)-1, length(base_params.y_grid)-1, length(base_params.z_grid)-1)
        )

        # 3. Crear el NamedTuple `run_params` completo, incluyendo los nuevos campos
        run_params = merge(base_params, (
            electron_injection_energy_eV = energy,
            initial_pressure = pressure,
            magnetic_field = [0.0, 0.0, field],
            anode_voltage = voltage,
            max_steps = search_params.max_steps,
            field_update_interval = search_params.field_update_interval,
            
            # Añadir los campos que faltaban:
            initial_air_density_n = initial_air_density_n,
            initial_electron_velocity = initial_electron_velocity,
            initial_positions = initial_positions,
            initial_velocities = initial_velocities,
            initial_temperature_grid = initial_temperature_grid
        ))
        
        # 4. Ahora la llamada a run_pic_simulation recibirá un `params` completo
        sim_results = run_pic_simulation(run_params, cpu_data, gpu_data, verbose=false)

        # ... (el resto de la función sigue igual) ...
        push!(results_list, (
            ElectronEnergy = energy,
            Pressure = pressure,
            MagneticField = field,
            AnodeVoltage = voltage,
            FinalEfficiency = sim_results.avg_efficiency,
            AvgLifetime = sim_results.avg_electron_lifetime,
            FinalTemp = sim_results.avg_temps_history[end]
        ))
    end
    
    # ... (el resto de la función sigue igual) ...
    results_df = DataFrame(results_list)
    sort!(results_df, :FinalEfficiency, rev=true)
    
    println("\n--- Parameter Search Complete ---")
    if !isempty(results_df)
        best = first(results_df)
        println("Best Parameters Found:")
        println("  Energy: $(best.ElectronEnergy) eV, Pressure: $(best.Pressure/1e6) MPa, Field: $(best.MagneticField) T, Voltage: $(best.AnodeVoltage) V")
        println("  Best Avg Efficiency: $(round(best.FinalEfficiency, digits=2))%")
    end
    
    return results_df
end

# --- Plasma Conductivity Calculation ---
function calculate_plasma_conductivity(electron_density, temperature_e, pressure, magnetic_field_strength)
    # (This function remains mostly the same, just ensure it uses the global constants)
    e = abs(ELECTRON_CHARGE)
    m_e = ELECTRON_MASS
    n_gas = pressure / (K_B * temperature_e)
    E_avg_eV = (3/2) * K_B * temperature_e / e
    
    # This part needs access to the CPU cross-section functions
    total_cs = 0.0
    for gas_name in keys(air_composition_cpu)
        gas_info = air_composition_cpu[gas_name]
        total_cs += gas_info["fraction"] * gas_info["total_cross_section_func"](E_avg_eV)
    end
    
    v_th = sqrt(3 * K_B * temperature_e / m_e)
    ν_en = n_gas * total_cs * v_th
    ω_ce = (e * magnetic_field_strength) / m_e
    
    σ = (e^2 * electron_density * ν_en) / (m_e * (ν_en^2 + ω_ce^2))
    return σ
end