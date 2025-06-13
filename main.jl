# --- Main.jl ---

# Incluir todos los módulos necesarios
include("utils/constants.jl")
include("utils/cross_sections.jl")
include("utils/particles.jl")
include("utils/collisions.jl")
include("utils/grid_functions.jl")
include("utils/electric_fields.jl")
include("utils/simulation_functions.jl")
include("utils/plotting_functions.jl")
include("utils/reporting_functions.jl")

# Dependencias externas
using Plots # O cualquier backend que uses (e.g., GR)
using DataFrames
using CSV
using Dates
using Statistics
# using Polynomials # Si todavía es necesario en alguna parte

# --- Parámetros de la Cámara ---
chamber_width = 0.1   # m
chamber_length = 0.1  # m
chamber_height = 0.001  # m
num_x_cells = 2
num_y_cells = 2
num_z_cells = 500
TOTAL_CELLS = Int(num_x_cells * num_y_cells * num_z_cells)

# --- Malla Rectangular ---
x_grid = LinRange(0, chamber_width, num_x_cells + 1)
y_grid = LinRange(0, chamber_length, num_y_cells + 1)
z_grid = LinRange(0, chamber_height, num_z_cells + 1)
x_cell_size = x_grid[2] - x_grid[1]
y_cell_size = y_grid[2] - y_grid[1]
z_cell_size = z_grid[2] - z_grid[1]
cell_volume = x_cell_size * y_cell_size * z_cell_size

anode_voltage = 1000.0  # Voltaje del ánodo en Voltios

# Different update intervals for parameter search vs main simulation
parameter_search_update_interval = 10  # For parameter search
main_simulation_update_interval = 10    # For main simulation

# --- Parámetros de Simulación PIC ---
initial_temperature = 800.0     # K
target_temperature = 22000.0    # K
dt = 1e-13                   # s (timestep)
simulated_electrons_per_step = 500 # Número de macropartículas inyectadas por paso
physical_electrons_per_step = 5e14 # Número físico de electrones representados por inyección
particle_weight = physical_electrons_per_step / simulated_electrons_per_step # Peso de cada macropartícula
max_steps = 10000                # Límite de pasos para la simulación final

# --- Parámetros Variables Iniciales (se sobrescribirán con los óptimos) ---
electron_injection_energy_eV = 50.0 # eV (valor inicial de ejemplo)
initial_pressure = 1e6              # Pa (valor inicial de ejemplo)
magnetic_field_strength = 0.5       # T (valor inicial de ejemplo)

# --- Parámetros del Campo Eléctrico Atractivo ---
attractive_potential_V = 50000.0 # Voltaje en z=chamber_height (relativo a z=0). Positivo atrae electrones.
if chamber_height <= 0.0
    error("chamber_height debe ser positivo para calcular el campo eléctrico.")
end
# Campo E uniforme: E = -∇V. Para V(z) = (V_attr / Lz) * z => E = [0, 0, -V_attr / Lz]
electric_field = [0.0, 0.0, -attractive_potential_V / chamber_height]
println("Campo Eléctrico Atractivo Aplicado (Ez): $(round(electric_field[3], digits=2)) V/m (Potencial: $(attractive_potential_V) V)")

# --- Crear directorio de plots si no existe ---
isdir("plots") || mkdir("plots")
isdir("simulation_data") || mkdir("simulation_data") # Para CSVs

# --- Ejecutar Búsqueda de Parámetros ---
println("\n--- Iniciando Búsqueda de Parámetros Óptimos ---")

results_df, best_params, best_efficiency = parameter_search(x_grid, y_grid, z_grid, anode_voltage, parameter_search_update_interval)

# --- Mostrar Mejores Parámetros ---
println("\n--- Mejores Parámetros Encontrados ---")
println("Energía de Electrón: $(best_params[1]) eV")
println("Presión: $(best_params[2]/1e6) MPa")
println("Campo Magnético: $(best_params[3]) T")
println("Potencial Atractivo: $(best_params[4]) V") # <-- Mostrar el potencial óptimo
println("Eficiencia Promedio Óptima: $(round(best_efficiency, digits=2)) %")

# --- Mostrar Top 5 Resultados ---
println("\n--- Top 5 Combinaciones de Parámetros (Ordenadas por Eficiencia) ---")
if !isempty(results_df)
    first_rows = first(results_df, min(5, nrow(results_df)))
    required_cols = [:ElectronEnergy, :Pressure, :MagneticField, :AttractivePotential, :FinalEfficiency]
    if all(col -> col in names(first_rows), required_cols)
        for i in 1:nrow(first_rows)
            row = first_rows[i, :]
            println("Rank $i: E=$(row.ElectronEnergy) eV, P=$(row.Pressure/1e6) MPa, B=$(row.MagneticField) T, V_attr=$(row.AttractivePotential) V, Eficiencia=$(round(row.FinalEfficiency, digits=2)) %")
        end
    else
        println("Advertencia: Faltan columnas esperadas en results_df para mostrar el top 5.")
        println("Columnas presentes: $(names(first_rows))")
    end
else
    println("No se generaron resultados en la búsqueda de parámetros.")
end


# --- Guardar Resultados Óptimos en Archivo ---
optim_report_filename = "parametros_optimos.txt"
try
    open(optim_report_filename, "w") do io
        println(io, "--- Resultados de Búsqueda de Parámetros ---")
        println(io, "Fecha y Hora: $(Dates.now())")
        println(io, "\n--- Mejores Parámetros ---")
        println(io, "Energía de Electrón: $(best_params[1]) eV")
        println(io, "Presión: $(best_params[2]/1e6) MPa")
        println(io, "Campo Magnético: $(best_params[3]) T")
        println(io, "Potencial Atractivo: $(best_params[4]) V")
        println(io, "Eficiencia Promedio Óptima: $(round(best_efficiency, digits=2)) %")

        if !isempty(results_df) && all(col -> col in names(results_df), required_cols)
            println(io, "\n--- Top 10 Combinaciones de Parámetros ---")
            for i in 1:min(10, nrow(results_df))
                row = results_df[i, :]
                println(io, "Rank $i: E=$(row.ElectronEnergy) eV, P=$(row.Pressure/1e6) MPa, B=$(row.MagneticField) T, V_attr=$(row.AttractivePotential) V, Eficiencia=$(round(row.FinalEfficiency, digits=2)) %")
            end
        end
    end
    println("\nResultados óptimos guardados en $(optim_report_filename)")
catch e
    println("Error al guardar resultados óptimos: $e")
end

# --- Ejecutar Simulación Completa con Parámetros Óptimos ---
println("\n--- Ejecutando Simulación Completa con Parámetros Óptimos y Campo E ---")
electron_injection_energy_eV = best_params[1]
initial_pressure = best_params[2]
magnetic_field_strength = best_params[3]
optimal_anode_voltage = best_params[4]

# Actualizar variables globales y configurar para la simulación final
magnetic_field = [0.0, 0.0, magnetic_field_strength] # Campo B óptimo

initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature) # Densidad con P óptima
initial_electron_velocity = electron_velocity_from_energy(electron_injection_energy_eV) # Velocidad con E óptima
initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
initial_air_density_n_value = Float64(initial_air_density_n)

println("Ejecutando con E=$(electron_injection_energy_eV)eV, P=$(initial_pressure/1e6)MPa, B=$(magnetic_field_strength)T, V_anode=$(optimal_anode_voltage)V")

# Asegúrate que la función run_pic_simulation retorna la tupla en este orden
(temperatures_history_julia, avg_temps_history_julia, density_history_julia,
 energy_deposition_history_julia, elastic_energy_deposition_history_julia, final_step, reached_target_temp,
 accumulated_input_energy, efficiency_history_julia, avg_efficiency_julia, detailed_data, avg_electron_lifetime, conductivity_history,
 final_charge_density_grid, final_electric_field_grid, final_potential_grid, electric_field_history, potential_history,
 position_history) = run_pic_simulation(
    initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
    magnetic_field,       # Campo B óptimo
    electron_charge, electron_mass, initial_electron_velocity,
    x_grid, y_grid, z_grid, optimal_anode_voltage, main_simulation_update_interval,
    verbose=true,
    max_steps_override=max_steps
)

println(length(potential_history))

# --- Calcular Métricas Finales ---
efficiency_simulation = avg_efficiency_julia # Usar la eficiencia promedio como principal

# Calcular eficiencia basada en energía interna (referencia)
total_input_energy = accumulated_input_energy
initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n_value * cell_volume * TOTAL_CELLS
final_temperature_avg = avg_temps_history_julia[end]
final_internal_energy = (3/2) * k_b * final_temperature_avg * initial_air_density_n_value * cell_volume * TOTAL_CELLS
increase_internal_energy = final_internal_energy - initial_internal_energy
efficiency_simulation_total_energy = 0.0
if total_input_energy > 1e-20
    efficiency_simulation_total_energy = (increase_internal_energy / total_input_energy) * 100
end

# Calcular conductividad final (usando T y P estimados al final)
final_density_grid_raw = density_history_julia[end] # Conteo por celda
final_electron_density_avg = mean(final_density_grid_raw) / cell_volume # Densidad numérica promedio
final_pressure_est = initial_air_density_n_value * k_b * final_temperature_avg # Presión estimada final
# Asegurar valores seguros para el cálculo de conductividad
safe_final_e_density = max(final_electron_density_avg, 1e-5)
safe_final_temp = max(final_temperature_avg, 1.0)
safe_final_pressure = max(final_pressure_est, 1e-3)

plasma_conductivity = calculate_plasma_conductivity(safe_final_e_density, safe_final_temp,
                                                   safe_final_pressure, # Usar presión estimada final
                                                   magnetic_field_strength)

# --- Resultados y Visualización ---
if final_step > 0
    time_points = LinRange(0, final_step * dt * 1e6, length(avg_temps_history_julia)) # µs
    # Ajustar time_points para eficiencia (que tiene un punto menos)
    time_points_eff = time_points[2:end]
    if length(time_points_eff) != length(efficiency_history_julia)
         println("Advertencia: Discrepancia en longitud de tiempo y eficiencia. Ajustando...")
         min_len = min(length(time_points_eff), length(efficiency_history_julia))
         time_points_eff = time_points_eff[1:min_len]
         efficiency_history_julia = efficiency_history_julia[1:min_len]
    end


    # Graficar Temperatura Promedio vs Tiempo
    p1 = plot_temperature_vs_time(time_points, avg_temps_history_julia, target_temperature)
    savefig(p1, "plots/temperature_vs_time.png")
    display(p1)

    # Graficar Eficiencia vs Tiempo
    if !isempty(efficiency_history_julia)
        p4 = plot_efficiency_vs_time(time_points_eff, efficiency_history_julia)
        savefig(p4, "plots/efficiency_vs_time.png")
        display(p4)
    else
        println("No hay datos de historial de eficiencia para graficar.")
    end

    # Graficar Eficiencia vs Tiempo de Vida (de la búsqueda de parámetros)
    if !isempty(results_df)
        println("\nGenerando gráfico de eficiencia vs tiempo de vida...")
        p5 = plot_efficiency_vs_lifetime(results_df)
        savefig(p5, "plots/efficiency_vs_lifetime.png")
        display(p5)
    else
         println("No hay datos de búsqueda de parámetros para gráfico eficiencia vs lifetime.")
    end


    # Heatmaps (Asegurarse que los grids finales no estén vacíos)
    final_density_grid = density_history_julia[end] # Ya es el grid de densidad
    final_temperature_grid = temperatures_history_julia[end] # Ya es el grid de temperatura
    if all(size(final_density_grid) .> 0) && all(size(final_temperature_grid) .> 0)
        z_slice_index = max(1, min(num_z_cells, num_z_cells ÷ 2)) # Índice seguro para el slice
        y_slice_index = 1

        # Heatmap de Densidad (slice)
        p2 = heatmap_density_slice(x_grid, z_grid, final_density_grid, y_slice_index)
        savefig(p2, "plots/density_heatmap.png")
        display(p2)

        # Heatmap de Temperatura (slice)
        p3 = heatmap_temperature_slice(x_grid, z_grid, final_temperature_grid, y_slice_index)
        savefig(p3, "plots/temperature_heatmap.png")
        display(p3)
        
        # --- Nuevos plots para campos eléctricos ---
        # Plot de Densidad de Carga
        p6 = plot_charge_density_slice(x_grid, z_grid, final_charge_density_grid, y_slice_index)
        savefig(p6, "plots/charge_density_slice.png")
        display(p6)
        
        # Compute y slice index for X-Z plane
        y_slice_index = max(1, min(num_y_cells, num_y_cells ÷ 2))
        
        # Plot de Potencial Eléctrico en plano X-Z
        p7 = plot_potential_xz_slice(x_grid, z_grid, final_potential_grid, y_slice_index)
        savefig(p7, "plots/potential_slice.png")
        display(p7)
        
        # Plot de Vectores de Campo Eléctrico
        p8 = plot_electric_field_vectors(x_grid, z_grid,
                                        final_electric_field_grid.Ex,
                                        final_electric_field_grid.Ez,
                                        y_slice_index)
        savefig(p8, "plots/electric_field_vectors.png")
        display(p8)
        
        # --- Memory Optimization: Reduce animation frames and add monitoring ---
        max_frames = 50  # Limit animation frames to prevent OOM
        # --- Generate Potential Animation (reduced frames) ---
        if !isempty(potential_history)
            println("Generando animación reducida del potencial (max $max_frames frames)...")
            animate_potential_slice(potential_history, x_grid, z_grid, y_slice_index;
                                     fps=5, filename="plots/potential_animation.gif", max_frames=max_frames)
        end
        
        # --- Generate Electron Positions Animation (reduced frames) ---
        if !isempty(position_history)
            println("Generando animación reducida de posiciones (max $max_frames frames)...")
            animate_electron_positions(position_history, chamber_width, chamber_height;
                                     filename="plots/electron_positions_animation.gif", fps=5, max_frames=max_frames)
        end
        
        # --- Memory Usage Report ---
        println("\n--- Uso de Memoria Post-Plotting ---")
        println("Longitud de historial eléctrico: ", length(electric_field_history))
        println("Longitud de historial potencial: ", length(potential_history))
        println("Longitud de historial posiciones: ", length(position_history))
    else
        println("No se pudieron generar heatmaps porque los grids finales están vacíos.")
    end

else
    println("La simulación final no completó ningún paso. No se generarán gráficos.")
    time_points = [0.0] # Para evitar errores posteriores
    # Asignar valores por defecto para evitar errores en el reporte/exportación
    efficiency_history_julia = []
    final_density_grid = zeros(num_x_cells, num_y_cells, num_z_cells)
    final_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
end

# --- Generar Reporte Final ---
report_filename = "plots/simulation_report.txt"
try
    # Asume que generate_report toma estos argumentos
    generate_report(report_filename, initial_temperature, initial_pressure,
                   electron_injection_energy_eV, magnetic_field_strength, dt,
                   final_step * dt, # Pasar tiempo total simulado
                   simulated_electrons_per_step,
                   efficiency_simulation, # Eficiencia promedio
                   increase_internal_energy,
                   total_input_energy, final_step, reached_target_temp,
                   avg_temps_history_julia, # Pasar historial completo
                   avg_efficiency_julia, # Eficiencia promedio
                   avg_electron_lifetime, plasma_conductivity,
                   attractive_potential_V) # <-- Añadir potencial al reporte
    println("\nReporte final guardado en $(report_filename)")
catch e
    println("Error al generar el reporte final: $e")
    # Considera imprimir el stacktrace: Base.showerror(stdout, e, catch_backtrace())
end


# --- Preparar Parámetros para Exportación ---
simulation_params = Dict(
    "initial_temperature" => initial_temperature,
    "target_temperature" => target_temperature,
    "initial_pressure_Pa" => initial_pressure,
    "electron_energy_eV" => electron_injection_energy_eV,
    "magnetic_field_T" => magnetic_field_strength,
    "electric_field_Ez_Vm" => electric_field[3], # Ez correspondiente al óptimo V
    "dt_s" => dt,
    "chamber_width_m" => chamber_width,
    "chamber_length_m" => chamber_length,
    "chamber_height_m" => chamber_height,
    "final_avg_temp_K" => avg_temps_history_julia[end],
    "avg_efficiency_percent" => avg_efficiency_julia,
    "avg_electron_lifetime_ns" => avg_electron_lifetime * 1e9,
    "final_conductivity_Sm" => plasma_conductivity,
    "final_step" => final_step,
    "reached_target_temp" => reached_target_temp
)


# --- Exportar Datos a CSV ---
report_filename = "plots/simulation_report.txt"
try
    # Asume que generate_report ahora acepta attractive_potential_V
    generate_report(report_filename, initial_temperature, initial_pressure,
                   electron_injection_energy_eV, magnetic_field_strength, dt,
                   final_step * dt, # Tiempo total simulado
                   simulated_electrons_per_step,
                   avg_efficiency_julia, # Eficiencia promedio
                   increase_internal_energy,
                   total_input_energy, final_step, reached_target_temp,
                   avg_temps_history_julia, # Historial completo Temp
                   avg_efficiency_julia, # Eficiencia promedio (de nuevo?)
                   avg_electron_lifetime, plasma_conductivity) # <-- Pasar potencial óptimo
    println("\nReporte final guardado en $(report_filename)")
catch e
    println("Error al generar el reporte final: $e")
end


# --- Exportar Datos a CSV (Actualizado) ---
csv_export_filename_base = "simulation_data/final_sim" # Prefijo para archivos CSV
try
    # Asume que export_simulation_data_to_csv ahora maneja los 4 best_params
    # y el simulation_params actualizado
    export_simulation_data_to_csv(
        time_points,
        avg_temps_history_julia,
        efficiency_history_julia,
        final_density_grid,
        final_temperature_grid,
        x_grid, y_grid, z_grid,
        simulation_params,      # Diccionario actualizado
        detailed_data,
        results_df,             # DataFrame ahora incluye AttractivePotential
        best_params,            # Tupla ahora tiene 4 elementos
        csv_export_filename_base
    )
     println("Datos detallados exportados a CSV en la carpeta 'simulation_data'")
catch e
     println("Error al exportar datos a CSV: $e")
end


println("\nSimulación completada. Resultados, plots y reporte guardados.")