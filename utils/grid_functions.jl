# ---------------------------------------------------------------------------
# ARCHIVO: grid_functions.jl (Funciones de deposición y actualización de malla)
# ---------------------------------------------------------------------------

using Statistics
using StatsBase: fit, Histogram, Weights

# GPU: Importar CUDA y la bandera de uso
using CUDA
# Asumimos que USE_GPU y constantes como K_B están definidas en otros archivos.

# ===========================================================================
# KERNELS DE GPU
# ===========================================================================

# GPU: Kernel genérico para depositar una cantidad por partícula en la malla.
# Usa operaciones atómicas para evitar conflictos.
# Sirve tanto para contar partículas (densidad) como para sumar energía.
function deposit_quantity_kernel!(grid_to_update, positions, quantity_per_particle,
                                  x_grid, y_grid, z_grid)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > size(positions, 1); return; end

    # Posición de la partícula
    px, py, pz = positions[i, 1], positions[i, 2], positions[i, 3]
    
    # Cantidad a depositar para esta partícula
    quantity = quantity_per_particle[i]

    # Encontrar índices de la celda (Nearest Grid Point)
    # Asume una malla uniforme.
    dx = x_grid[2] - x_grid[1]
    dy = y_grid[2] - y_grid[1]
    dz = z_grid[2] - z_grid[1]
    ix = clamp(floor(Int, px / dx) + 1, 1, size(grid_to_update, 1))
    iy = clamp(floor(Int, py / dy) + 1, 1, size(grid_to_update, 2))
    iz = clamp(floor(Int, pz / dz) + 1, 1, size(grid_to_update, 3))

    # GPU: Operación atómica para sumar de forma segura la cantidad a la celda.
    CUDA.atomic_add!(pointer(grid_to_update, (ix-1)*stride(grid_to_update,1) + (iy-1)*stride(grid_to_update,2) + (iz-1)*stride(grid_to_update,3) + 1), quantity)

    return
end

# GPU: Kernel para actualizar la malla de temperatura.
# Es una operación simple tipo "map", cada hilo trabaja en una celda.
function update_temperature_kernel!(new_temp_grid, old_temp_grid,
                                    inelastic_energy_grid, elastic_energy_grid,
                                    n_air_cell)
    i, j, k = threadIdx().x, threadIdx().y, blockIdx().x
    nx, ny, nz = size(new_temp_grid)

    if i <= nx && j <= ny && k <= nz
        # Factor de conversión de energía a temperatura
        energy_to_temp_factor = (2.0 / (3.0 * K_B)) / n_air_cell
        
        delta_T_inelastic = energy_to_temp_factor * inelastic_energy_grid[i, j, k]
        delta_T_elastic = energy_to_temp_factor * elastic_energy_grid[i, j, k]
        
        new_temp_grid[i, j, k] = old_temp_grid[i, j, k] + delta_T_inelastic + delta_T_elastic
    end
    
    return
end


# ===========================================================================
# FUNCIONES DE ALTO NIVEL (DESPACHADORAS)
# ===========================================================================

# --- Cálculo de la Densidad en la Malla ---
function calculate_grid_density(positions, x_grid, y_grid, z_grid)
    nx, ny, nz = length(x_grid)-1, length(y_grid)-1, length(z_grid)-1
    num_particles = size(positions, 1)

    if USE_GPU && num_particles > 0
        # Versión GPU
        density_grid_d = CUDA.zeros(Float64, nx, ny, nz)
        positions_d = CuArray(positions)
        
        # Para contar partículas, la "cantidad" a depositar es 1.0 para cada una.
        ones_d = CUDA.ones(Float64, num_particles)
        
        x_grid_d, y_grid_d, z_grid_d = CuArray(x_grid), CuArray(y_grid), CuArray(z_grid)

        threads = 256
        blocks = cld(num_particles, threads)
        @cuda threads=threads blocks=blocks deposit_quantity_kernel!(
            density_grid_d, positions_d, ones_d,
            x_grid_d, y_grid_d, z_grid_d
        )
        return density_grid_d # Devuelve un CuArray
    else
        # Versión CPU
        if num_particles == 0
            return zeros(Float64, nx, ny, nz)
        end
        bins = (collect(x_grid), collect(y_grid), collect(z_grid))
        h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), bins)
        return h.weights
    end
end

# --- Depositar Energía en la Malla ---
function deposit_energy(positions, energy_transfer, x_grid, y_grid, z_grid)
    nx, ny, nz = length(x_grid)-1, length(y_grid)-1, length(z_grid)-1
    num_particles = size(positions, 1)

    if USE_GPU && num_particles > 0
        # Versión GPU
        energy_grid_d = CUDA.zeros(Float64, nx, ny, nz)
        positions_d = CuArray(positions)
        energy_transfer_d = CuArray(energy_transfer)
        
        x_grid_d, y_grid_d, z_grid_d = CuArray(x_grid), CuArray(y_grid), CuArray(z_grid)

        threads = 256
        blocks = cld(num_particles, threads)
        @cuda threads=threads blocks=blocks deposit_quantity_kernel!(
            energy_grid_d, positions_d, energy_transfer_d,
            x_grid_d, y_grid_d, z_grid_d
        )
        return energy_grid_d # Devuelve un CuArray
    else
        # Versión CPU
        if num_particles == 0
            return zeros(Float64, nx, ny, nz)
        end
        bins = (collect(x_grid), collect(y_grid), collect(z_grid))
        w = Weights(energy_transfer)
        h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), w, bins)
        return h.weights
    end
end

# --- Actualización de la Temperatura en la Malla ---
function update_temperature_grid(temperature_grid, energy_deposition_grid, elastic_energy_deposition_grid, n_air, cell_volume)
    if USE_GPU && isa(temperature_grid, CuArray)
        # Versión GPU
        nx, ny, nz = size(temperature_grid)
        new_temp_grid_d = CUDA.zeros(Float64, nx, ny, nz)
        n_air_cell = n_air * cell_volume
        
        threads = (16, 16) # Bloques 2D
        blocks = nz # Bloques en la dirección Z
        @cuda threads=threads blocks=blocks update_temperature_kernel!(
            new_temp_grid_d, temperature_grid,
            energy_deposition_grid, elastic_energy_deposition_grid,
            n_air_cell
        )
        return new_temp_grid_d # Devuelve el nuevo CuArray de temperatura
    else
        # Versión CPU
        n_air_cell = n_air * cell_volume
        # La operación con punto (.) ya es eficiente en la CPU para arrays
        delta_T_inelastic = (2.0 / (3.0 * K_B)) .* (energy_deposition_grid ./ n_air_cell)
        delta_T_elastic = (2.0 / (3.0 * K_B)) .* (elastic_energy_deposition_grid ./ n_air_cell)
        new_temperature_grid = temperature_grid .+ delta_T_inelastic .+ delta_T_elastic
        return new_temperature_grid
    end
end


# ===========================================================================
# FUNCIONES AUXILIARES (SIN CAMBIOS)
# ===========================================================================

# Esta función no necesita una versión GPU ya que opera con escalares.
function calculate_air_density_n(pressure, temperature)
    return pressure / (K_B * temperature)
end