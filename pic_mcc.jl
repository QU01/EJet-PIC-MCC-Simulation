using Interpolations
using Plots
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Dates
using DataFrames
using WriteVTK  

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


function export_vtk(step, positions, velocities, temperature_grid, density_grid, x_grid, y_grid, z_grid, output_dir="vtk_output")
    mkpath(output_dir)
    
    # 1. Exportar malla estructurada (temperatura y densidad)
    x_coords = collect(x_grid)
    y_coords = collect(y_grid)
    z_coords = collect(z_grid)
    vtk_grid_file = joinpath(output_dir, "grid_data_$(step)")
    vtkfile = vtk_grid(vtk_grid_file, x_coords, y_coords, z_coords)
    vtkfile["temperature", VTKCellData()] = temperature_grid
    vtkfile["electron_density", VTKCellData()] = density_grid
    vtk_save(vtkfile)
    
    # 2. Exportar posiciones de electrones como VERTICES (POLYDATA .vtp)
    if !isempty(positions)
        vtk_points_file = joinpath(output_dir, "electron_positions_$(step)")
        
        # Convertir posiciones a matriz 3xN (dim x num_points)
        points = collect(positions')  # Transponer y convertir a Matrix
        
        # Crear celdas: un vértice por electrón
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:size(positions, 1)]
        
        # Crear archivo VTK (automáticamente .vtp por usar VTK_VERTEX)
        vtk = vtk_grid(vtk_points_file, points, cells)
        
        # Añadir velocidades como datos de punto
        if !isempty(velocities)
            velocity_magnitude = sqrt.(sum(velocities.^2, dims=2))[:]  # Vector 1D
            vtk["velocity", VTKPointData()] = collect(velocities')     # Matriz 3xN
            vtk["velocity_magnitude", VTKPointData()] = velocity_magnitude
        end
        
        vtk_save(vtk)
    end
    
    println("Datos VTK exportados para el paso $(step)")
end
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
function initialize_electrons(num_electrons, chamber_width, chamber_length, initial_velocity)
    rng = MersenneTwister(0)
    x_positions = rand(rng, num_electrons) * chamber_width
    y_positions = rand(rng, num_electrons) * chamber_length
    z_positions = zeros(num_electrons)

    vx = zeros(num_electrons)
    vy = zeros(num_electrons)
    vz = fill(initial_velocity, num_electrons)

    positions = hcat(x_positions, y_positions, z_positions)
    velocities = hcat(vx, vy, vz)
    return positions, velocities
end

# --- Movimiento de Electrones con Fuerza de Lorentz (Corregido) ---
function move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
    force = zeros(size(velocities))
    for i in 1:size(velocities, 1)
        force[i, :] = lorentz_force(velocities[i, :], magnetic_field, electron_charge)
    end
    acceleration = force ./ electron_mass
    new_velocities = velocities .+ (acceleration .* dt)
    new_positions = positions .+ (new_velocities .* dt)
    return new_positions, new_velocities
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

    v_magnitudes = sqrt.(sum(velocities.^2, dims=2))
    E_e_eV = electron_energy_from_velocity.(v_magnitudes) ./ 1.60218e-19
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)
    total_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    # --- Nuevos parámetros de pérdida de energía ---
    excitation_energy_loss_eV = 3.0
    excitation_energy_loss_joules = excitation_energy_loss_eV * 1.60218e-19
    ionization_secondary_electron_fraction = 0.5

    # Initialize secondary_electron_ke here to ensure it's always defined
    secondary_electron_ke = zeros(num_particles) # Initialize as zeros array

    # Iterar sobre los gases
    for gas_name in keys(air_composition)
        gas_info = air_composition[gas_name]
        mass = gas_info["mass"]
        gas_fraction = gas_info["fraction"]
        ionization_energy_eV = gas_info["ionization_energy_eV"]
        ionization_energy_joules = ionization_energy_eV * 1.60218e-19
        total_cross_section_func = gas_info["total_cross_section_func"]
        ionization_cross_section_func = gas_info["ionization_cross_section_func"]

        total_cross_section = total_cross_section_func.(E_e_eV)
        ionization_cross_section = ionization_cross_section_func.(E_e_eV)
        excitation_cross_section = estimate_excitation_cross_section.(
            total_cross_section, ionization_cross_section, E_e_eV, ionization_energy_eV
        )

        collision_probs_total = 1 .- exp.(-(air_density_n * gas_fraction) .* total_cross_section .* v_magnitudes .* dt)
        rand_numbers_total = rand(rng, num_particles)
        collide_total = rand_numbers_total .< collision_probs_total
        collided_flags = collided_flags .| collide_total

        ionize_prob = ifelse.(collide_total, ionization_cross_section ./ total_cross_section, 0.0)
        rand_numbers_ionize = rand(rng, num_particles)
        collide_ionize = (rand_numbers_ionize .< ionize_prob) .& collide_total .& (E_e_eV .> ionization_energy_eV)

        excite_prob = ifelse.(collide_total .& .!collide_ionize, excitation_cross_section ./ total_cross_section, 0.0)
        rand_numbers_excite = rand(rng, num_particles)
        collide_excite = (rand_numbers_excite .< excite_prob) .& collide_total .& .!collide_ionize

        collide_elastic = collide_total .& .!collide_ionize .& .!collide_excite

        v_rand_direction = randn(rng, num_particles, 3)
        v_rand_norm = v_rand_direction ./ sqrt.(sum(v_rand_direction.^2, dims=2))

        excess_energy = max.(0, E_e_joules .- ionization_energy_joules)
        secondary_electron_ke_step = 0.1 .* excess_energy # Define step-local secondary_electron_ke
        secondary_electron_ke = ifelse.(collide_ionize, secondary_electron_ke_step, secondary_electron_ke) # Update global secondary_electron_ke only if ionize

        ionization_loss = ifelse.(collide_ionize, ionization_energy_joules .+ secondary_electron_ke_step, 0.0) # Use step-local variable

        excitation_loss = ifelse.(collide_excite, excitation_energy_loss_joules, 0.0)

        fraction_energy_elastic = (2 * electron_mass / mass)
        elastic_loss_step = ifelse.(collide_elastic, fraction_energy_elastic .* E_e_joules, 0.0)
        elastic_energy_transfer .+= elastic_loss_step

        energy_loss_step = ionization_loss .+ excitation_loss .+ elastic_loss_step
        total_energy_transfer .+= energy_loss_step

        energy_transfer_eV_step = energy_loss_step ./ 1.60218e-19
        collision_energy_transfers_eV = ifelse.(collide_total, energy_transfer_eV_step, collision_energy_transfers_eV)

        if simulated_electrons_per_step > 0
            velocities_secondary = v_rand_norm .* sqrt.(2 .* secondary_electron_ke ./ electron_mass) # Use the global, potentially updated, secondary_electron_ke
            velocities_new = ifelse.(reshape(collide_ionize, :, 1), velocities_secondary, velocities_new)
        end

        new_v_magnitude = sqrt.(2 .* max.(0.0, E_e_joules .- energy_loss_step) ./ electron_mass)
        velocities_scattered = v_rand_norm .* new_v_magnitude
        velocities_new = ifelse.(reshape(collide_total, :, 1), velocities_scattered, velocities_new)
    end

    return positions, velocities_new, rng, total_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer
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
        println(io, "Eficiencia de la Simulación: $(efficiency_simulation) %")
        println(io, "Eficiencia Promedio por Paso: $(avg_efficiency_julia) %") # Added average step efficiency to report
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

# --- Función Principal de Simulación (modificada para calcular eficiencia por paso) ---
function run_pic_simulation(initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass; verbose=true, vtk_export=true, vtk_frequency=5)

    positions_history = [initial_positions]
    velocities_history = [initial_velocities]
    temperatures_history = [initial_temperature_grid]
    avg_temps_history = [Statistics.mean(initial_temperature_grid)]
    density_history = [calculate_grid_density(initial_positions, x_grid, y_grid, z_grid)]
    energy_deposition_history = [zeros(size(initial_temperature_grid))]
    elastic_energy_deposition_history = [zeros(size(initial_temperature_grid))]
    efficiency_history = Float64[] # Para almacenar la eficiencia en cada paso

    positions = initial_positions
    velocities = initial_velocities
    temperature_grid = initial_temperature_grid
    rng = MersenneTwister(0)

    current_time = 0.0
    step = 0
    reached_target_temp = false
    accumulated_input_energy = 0.0

    # Para calcular la energía interna inicial
    initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n * cell_volume * TOTAL_CELLS
    current_internal_energy = initial_internal_energy

    # Exportar estado inicial si se solicita
    if vtk_export
        density_grid = calculate_grid_density(positions, x_grid, y_grid, z_grid)
        export_vtk(0, positions, velocities, temperature_grid, density_grid, x_grid, y_grid, z_grid)
    end

    while avg_temps_history[end] < target_temperature && step < max_steps
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

        if verbose
            println("Energía Cinética de Nuevos Electrones (Input de este paso): $(step_input_energy) J")
        end

        positions = vcat(positions, new_positions)
        velocities = vcat(velocities, new_velocities)

        # Guardar energía interna antes de las colisiones
        previous_internal_energy = current_internal_energy

        # Movimiento de electrones y condiciones de frontera
        positions, velocities = move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
        positions, velocities = apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)

        # Simulación de colisiones
        positions, velocities, rng, energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer = monte_carlo_collision(
            positions, velocities, initial_air_density_n, dt, rng, 1.0
        )

        # Depositar energía en la malla y actualizar temperatura
        energy_deposition_grid_step = deposit_energy(positions, energy_transfer * particle_weight, x_grid, y_grid, z_grid)
        elastic_energy_deposition_grid_step = deposit_energy(positions, elastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)

        temperature_grid = update_temperature_grid(temperature_grid, energy_deposition_grid_step, elastic_energy_deposition_grid_step, initial_air_density_n, cell_volume)

        # Calcular la nueva energía interna del gas
        current_internal_energy = (3/2) * k_b * Statistics.mean(temperature_grid) * initial_air_density_n * cell_volume * TOTAL_CELLS
        energy_transferred_to_gas = current_internal_energy - previous_internal_energy

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

        # Exportar datos VTK periódicamente
        if vtk_export && (step % vtk_frequency == 0)
            density_grid = calculate_grid_density(positions, x_grid, y_grid, z_grid)
            export_vtk(step, positions, velocities, temperature_grid, density_grid, x_grid, y_grid, z_grid)
        end

        if avg_temps_history[end] >= target_temperature
            reached_target_temp = true
            if verbose
                println("Temperatura objetivo alcanzada en el paso $(step)!")
            end
            break
        end
    end

    # Exportar datos finales
    if vtk_export
        density_grid = calculate_grid_density(positions, x_grid, y_grid, z_grid)
        export_vtk(step, positions, velocities, temperature_grid, density_grid, x_grid, y_grid, z_grid)
    end

    avg_efficiency = isempty(efficiency_history) ? 0.0 : Statistics.mean(efficiency_history)
    if verbose
        println("\nEficiencia promedio a lo largo de la simulación: $(avg_efficiency) %")
    end

    return temperatures_history, avg_temps_history, density_history,
           energy_deposition_history, elastic_energy_deposition_history, step, reached_target_temp,
           accumulated_input_energy, efficiency_history, avg_efficiency
end


# --- Función para Ejecutar Simulación y Estimar Eficiencia ---
function estimate_efficiency(electron_injection_energy_eV, initial_pressure, magnetic_field_strength)
    magnetic_field = [0.0, 0.0, magnetic_field_strength]

    initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature)
    initial_electron_velocity = electron_velocity_from_energy(electron_injection_energy_eV)
    initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
    initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
    initial_air_density_n_value = Float64(initial_air_density_n)

    (temperatures_history_julia, avg_temps_history_julia, density_history_julia,
     energy_deposition_history_julia, elastic_energy_deposition_history_julia, final_step, reached_target_temp, accumulated_input_energy, efficiency_history_julia, avg_efficiency_julia) = run_pic_simulation(
        initial_positions, initial_velocities, initial_temperature_grid,
        initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
         magnetic_field, electron_charge, electron_mass, verbose=false, vtk_export=false
    )

    electron_energy_joules = electron_injection_energy_eV * 1.60218e-19
    total_electrons_injected = physical_electrons_per_step * final_step
    total_input_energy = electron_energy_joules * total_electrons_injected

    initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n_value * cell_volume * TOTAL_CELLS
    final_internal_energy = (3/2) * k_b * avg_temps_history_julia[end] * initial_air_density_n_value * cell_volume * TOTAL_CELLS
    increase_internal_energy = final_internal_energy - initial_internal_energy

    efficiency_simulation = (increase_internal_energy / total_input_energy) * 100

    return efficiency_simulation, avg_temps_history_julia[end], final_step, reached_target_temp, avg_efficiency_julia
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
dt = 1e-9
simulation_time = 1e-7 # No longer used for stopping condition
simulated_electrons_per_step = Int(1e4)
physical_electrons_per_step = 5e17
particle_weight = physical_electrons_per_step / simulated_electrons_per_step
max_steps = Int(100) # Reducido para diagnóstico
electron_injection_energy_eV = 100.0
initial_pressure = 2e6 # Increased pressure for better collision rate
magnetic_field_strength = 1 # Added magnetic field for confinement


initial_air_density_n = calculate_air_density_n(initial_pressure, initial_temperature)
initial_electron_velocity = electron_velocity_from_energy(electron_injection_energy_eV)
initial_positions, initial_velocities = initialize_electrons(0, chamber_width, chamber_length, initial_electron_velocity)
initial_temperature_grid = fill(initial_temperature, (num_x_cells, num_y_cells, num_z_cells))
initial_air_density_n_value = Float64(initial_air_density_n)
magnetic_field = [0.0, 0.0, magnetic_field_strength] # Define magnetic_field here in global scope


# --- Ejecutar Simulación PIC-MCC con Multiples Electrones ---
(temperatures_history_julia, avg_temps_history_julia, density_history_julia,
 energy_deposition_history_julia, elastic_energy_deposition_history_julia, final_step, reached_target_temp, accumulated_input_energy, efficiency_history_julia, avg_efficiency_julia) = run_pic_simulation(
    initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n_value, air_composition, dt, simulated_electrons_per_step,
     magnetic_field, electron_charge, electron_mass, verbose=true, vtk_export=true, vtk_frequency=5
)

# --- Cálculo de Eficiencia Energética Final (Opcional - ya tienes la eficiencia promedio por paso) ---
total_input_energy = accumulated_input_energy # Usar accumulated_input_energy directamente
initial_internal_energy = (3/2) * k_b * initial_temperature * initial_air_density_n_value * cell_volume * TOTAL_CELLS
final_internal_energy = (3/2) * k_b * avg_temps_history_julia[end] * initial_air_density_n_value * cell_volume * TOTAL_CELLS
increase_internal_energy = final_internal_energy - initial_internal_energy
efficiency_simulation = (increase_internal_energy / total_input_energy) * 100

println("\n--- Resultados de la Simulación con Multiples Electrones ---")
println("Eficiencia de la Simulación Final: $(efficiency_simulation) %")
println("Eficiencia Promedio por Paso: $(avg_efficiency_julia) %") # Print average step efficiency
println("Energía Total Introducida por Electrones: $(total_input_energy) J")
println("Aumento Total de Energía Interna del Aire: $(increase_internal_energy) J")

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
generate_report("simulation_report.txt", initial_temperature, initial_pressure, electron_injection_energy_eV, magnetic_field_strength, dt, simulation_time, simulated_electrons_per_step, efficiency_simulation, increase_internal_energy, total_input_energy, final_step, reached_target_temp, avg_temps_history_julia, avg_efficiency_julia) # Pass avg_efficiency_julia to report

println("\nSimulación completada. Resultados, plots y reporte guardados.")