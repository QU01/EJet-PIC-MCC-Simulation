using Dates
using DataFrames
using CSV

function calculate_energy_based_efficiency(initial_temperature, final_temperature, initial_air_density_n, cell_volume, total_cells, accumulated_input_energy)
    # Calcular energía interna inicial y final del gas (ideal)
    initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n * cell_volume * total_cells
    final_internal_energy = (3/2) * k_b * final_temperature * initial_air_density_n * cell_volume * total_cells
    delta_internal_energy = final_internal_energy - initial_internal_energy
    
    # Calcular eficiencia
    if accumulated_input_energy > 0
        efficiency = (delta_internal_energy / accumulated_input_energy) * 100
    else
        efficiency = 0.0
    end
    
    return efficiency, delta_internal_energy, initial_internal_energy, final_internal_energy
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
        final_temperature = avg_temps_history_julia[end]
        energy_based_efficiency, delta_U, initial_U, final_U = calculate_energy_based_efficiency(
            initial_temperature, final_temperature, initial_air_density_n, cell_volume, TOTAL_CELLS, accumulated_input_energy
        )
        println("\n--- Eficiencia Basada en Energía Interna ---")
        println("Energía Interna Inicial del Gas: $(initial_U) J")
        println("Energía Interna Final del Gas: $(final_U) J")
        println("Cambio en Energía Interna (ΔU): $(delta_U) J")
        println("Energía Total de Electrones (Input): $(accumulated_input_energy) J")
        println("Eficiencia Térmica Global: $(round(energy_based_efficiency, digits=2)) %")
    end
    println("\nReporte guardado en: $(filename)")
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