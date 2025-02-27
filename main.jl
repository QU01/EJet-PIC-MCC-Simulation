include("utils/constants.jl")
include("utils/cross_sections.jl")
include("utils/particles.jl")
include("utils/collisions.jl")
include("utils/grid_functions.jl")
include("utils/simulation_functions.jl")
include("utils/plotting_functions.jl")
include("utils/reporting_functions.jl")

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
electron_injection_energy_eV = 200
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
savefig(p1, "plots/temperature_vs_time.png")
display(p1)

# Graficar Eficiencia vs Tiempo
p4 = plot_efficiency_vs_time(time_points[2:end], efficiency_history_julia) # Time points adjusted to match efficiency history length
savefig(p4, "plots/efficiency_vs_time.png")
display(p4)

final_density_grid = density_history_julia[end]
final_temperature_grid = temperatures_history_julia[end]
z_slice_index = num_z_cells ÷ 2

# Heatmap de Densidad (slice)
p2 = heatmap_density_slice(x_grid, y_grid, final_density_grid, z_slice_index)
savefig(p2, "plots/density_heatmap.png")
display(p2)

# Heatmap de Temperatura (slice)
p3 = heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, z_slice_index)
savefig(p3, "plots/temperature_heatmap.png")
display(p3)

# --- Generar Reporte Final ---
generate_report("plots/simulation_report.txt", initial_temperature, initial_pressure, electron_injection_energy_eV, magnetic_field_strength, dt, simulation_time, simulated_electrons_per_step, efficiency_simulation, increase_internal_energy, total_input_energy, final_step, reached_target_temp, avg_temps_history_julia, avg_efficiency_julia) # Pass avg_efficiency_julia which is now the main efficiency

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