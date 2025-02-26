using Statistics
using StatsBase

# --- Funciones Auxiliares (Sin Cambios Mayores) ---
function calculate_air_density_n(pressure, temperature)
    return pressure / (k_b * temperature)
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