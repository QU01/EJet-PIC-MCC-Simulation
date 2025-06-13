# ---------------------------------------------------------------------------
# ARCHIVO: electric_fields.jl (Solucionador de campo de Poisson y utilidades)
# ---------------------------------------------------------------------------

using LinearAlgebra
using Statistics

# GPU: Importar CUDA y la bandera de uso
using CUDA
# Asumimos que USE_GPU se define en otro archivo (ej. collisions.jl) y es accesible
# Si no, descomenta la siguiente línea:
# const USE_GPU = CUDA.functional()

# --- Constantes Físicas ---
const ε_0 = 8.854187817e-12  # F/m
# Asumimos que ELECTRON_CHARGE se define en otro archivo (ej. air_properties.jl)
# Si no, descomenta la siguiente línea:
# const ELECTRON_CHARGE = -1.60218e-19

# ===========================================================================
# ESTRUCTURAS Y FUNCIONES HÍBRIDAS (CPU/GPU)
# ===========================================================================

# GPU: Modificamos la estructura para que sea paramétrica.
# Puede contener arrays de CPU (Array{T,3}) o de GPU (CuArray{T,3}).
struct ElectricFieldGrid{T<:AbstractArray}
    Ex::T
    Ey::T
    Ez::T
    potential::T
end

# GPU: Kernel para depositar la carga de las partículas en la malla.
# Usa operaciones atómicas para manejar colisiones de escritura.
function deposit_charge_kernel!(charge_density_grid, positions, particle_weight,
                                x_grid, y_grid, z_grid, cell_volume)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > size(positions, 1); return; end

    charge_per_particle = -ELECTRON_CHARGE * particle_weight / cell_volume

    # Posición de la partícula
    px, py, pz = positions[i, 1], positions[i, 2], positions[i, 3]

    # Encontrar índices de la celda (Cloud-in-Cell de orden cero - Nearest Grid Point)
    # Nota: searchsortedlast no está en la GPU, así que hacemos una búsqueda simple.
    # Esto asume una malla uniforme.
    dx, dy, dz = x_grid[2] - x_grid[1], y_grid[2] - y_grid[1], z_grid[2] - z_grid[1]
    ix = clamp(floor(Int, px / dx) + 1, 1, size(charge_density_grid, 1))
    iy = clamp(floor(Int, py / dy) + 1, 1, size(charge_density_grid, 2))
    iz = clamp(floor(Int, pz / dz) + 1, 1, size(charge_density_grid, 3))

    # GPU: Operación atómica. Suma de forma segura el valor a la celda de la malla,
    # incluso si otro hilo intenta escribir en la misma celda al mismo tiempo.
    CUDA.atomic_add!(pointer(charge_density_grid, (ix-1)*stride(charge_density_grid,1) + (iy-1)*stride(charge_density_grid,2) + (iz-1)*stride(charge_density_grid,3) + 1), charge_per_particle)

    return
end

# GPU: Kernel para una iteración del solucionador de Poisson (Gauss-Seidel Red-Black).
# El patrón Red-Black evita conflictos de datos al actualizar puntos vecinos.
function poisson_step_kernel!(V, V_old, charge_density, factors, ω, is_red_pass::Bool)
    i, j, k = threadIdx().x, threadIdx().y, blockIdx().x
    nx, ny, nz = size(V)

    # Solo trabajar en puntos internos
    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        # Patrón Red-Black: solo actualiza si el color del hilo coincide con el pase
        if ( (i+j+k) % 2 == 0 ) == is_red_pass
            dx2_inv, dy2_inv, dz2_inv, denominator_inv = factors

            V_new = (
                (V[i+1,j,k] + V[i-1,j,k]) * dx2_inv +
                (V[i,j+1,k] + V[i,j-1,k]) * dy2_inv +
                (V[i,j,k+1] + V[i,j,k-1]) * dz2_inv -
                charge_density[i,j,k] / ε_0
            ) * denominator_inv

            # Sobre-relajación
            V[i,j,k] = (1-ω) * V_old[i,j,k] + ω * V_new
        end
    end
    return
end

# GPU: Kernel para aplicar las condiciones de frontera de Neumann.
function apply_neumann_bc_kernel!(V)
    idx, dim = threadIdx().x, blockIdx().x
    nx, ny, nz = size(V)
    
    if dim == 1 && idx <= ny*nz # Frontera X
        V[1, (idx-1)%ny + 1, (idx-1)÷ny + 1] = V[2, (idx-1)%ny + 1, (idx-1)÷ny + 1]
        V[nx, (idx-1)%ny + 1, (idx-1)÷ny + 1] = V[nx-1, (idx-1)%ny + 1, (idx-1)÷ny + 1]
    elseif dim == 2 && idx <= nx*nz # Frontera Y
        V[(idx-1)%nx + 1, 1, (idx-1)÷nx + 1] = V[(idx-1)%nx + 1, 2, (idx-1)÷nx + 1]
        V[(idx-1)%nx + 1, ny, (idx-1)÷nx + 1] = V[(idx-1)%nx + 1, ny-1, (idx-1)÷nx + 1]
    end
    return
end


# GPU: Kernel para calcular el campo eléctrico a partir del potencial.
function calculate_efield_kernel!(Ex, Ey, Ez, V, cell_sizes)
    i, j, k = threadIdx().x, threadIdx().y, blockIdx().x
    nx, ny, nz = size(V)
    inv_2dx, inv_2dy, inv_2dz = cell_sizes

    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        Ex[i,j,k] = -(V[i+1,j,k] - V[i-1,j,k]) * inv_2dx
        Ey[i,j,k] = -(V[i,j+1,k] - V[i,j-1,k]) * inv_2dy
        Ez[i,j,k] = -(V[i,j,k+1] - V[i,j,k-1]) * inv_2dz
    end
    return
end

# ===========================================================================
# FUNCIONES DE ALTO NIVEL (DESPACHADORAS)
# ===========================================================================

function calculate_charge_density(positions, particle_weight, x_grid, y_grid, z_grid, cell_volume)
    nx, ny, nz = length(x_grid)-1, length(y_grid)-1, length(z_grid)-1
    
    if USE_GPU && size(positions, 1) > 0
        # Versión GPU
        charge_density_d = CUDA.zeros(Float64, nx, ny, nz)
        positions_d = CuArray(positions)
        
        # Pasamos los grids de la CPU a la GPU. Son pequeños, no es un gran coste.
        x_grid_d, y_grid_d, z_grid_d = CuArray(x_grid), CuArray(y_grid), CuArray(z_grid)

        threads = 256
        blocks = cld(size(positions, 1), threads)
        @cuda threads=threads blocks=blocks deposit_charge_kernel!(
            charge_density_d, positions_d, particle_weight,
            x_grid_d, y_grid_d, z_grid_d, cell_volume
        )
        return charge_density_d # Devolvemos el CuArray
    else
        # Versión CPU
        # (Asume que calculate_grid_density existe en otro archivo)
        electron_density = calculate_grid_density(positions, x_grid, y_grid, z_grid)
        charge_density = -ELECTRON_CHARGE * electron_density * particle_weight / cell_volume
        return charge_density # Devolvemos el Array de CPU
    end
end

function solve_poisson_equation(charge_density_grid, x_cell_size, y_cell_size, z_cell_size,
                                V_anode; max_iterations=1000, tolerance=1e-5)
    
    if USE_GPU && isa(charge_density_grid, CuArray)
        # Versión GPU
        nx, ny, nz = size(charge_density_grid)
        V_d = CUDA.zeros(Float64, nx, ny, nz)
        V_d[:, :, end] .= V_anode # Condición de frontera Dirichlet
        
        factors_d = CuArray([1.0/x_cell_size^2, 1.0/y_cell_size^2, 1.0/z_cell_size^2, 
                             1.0 / (2.0 * (1/x_cell_size^2 + 1/y_cell_size^2 + 1/z_cell_size^2))])
        ω = 1.8

        threads = (16, 16) # 2D threads por bloque
        blocks = nz # Bloques en la dirección Z

        for iter in 1:max_iterations
            V_old_d = copy(V_d)
            
            # Pase Rojo
            @cuda threads=threads blocks=blocks poisson_step_kernel!(V_d, V_old_d, charge_density_grid, factors_d, ω, true)
            # Pase Negro
            @cuda threads=threads blocks=blocks poisson_step_kernel!(V_d, V_old_d, charge_density_grid, factors_d, ω, false)

            # Aplicar fronteras de Neumann
            @cuda threads=max(ny*nz, nx*nz) blocks=2 apply_neumann_bc_kernel!(V_d)
            
            # Comprobación de convergencia (costosa, hacerla cada N iteraciones)
            if iter % 20 == 0
                max_change = maximum(abs.(V_d .- V_old_d))
                if max_change < tolerance
                    # println("Poisson GPU convergió en $iter iteraciones.")
                    break
                end
            end
        end
        return V_d # Devolvemos el CuArray
    else
        # Versión CPU
        return solve_poisson_equation_cpu(charge_density_grid, x_cell_size, y_cell_size, z_cell_size, V_anode; max_iterations=max_iterations, tolerance=tolerance)
    end
end

function calculate_electric_field_from_potential(V, x_cell_size, y_cell_size, z_cell_size)
    if USE_GPU && isa(V, CuArray)
        # Versión GPU
        nx, ny, nz = size(V)
        Ex_d = CUDA.zeros(Float64, nx, ny, nz)
        Ey_d = CUDA.zeros(Float64, nx, ny, nz)
        Ez_d = CUDA.zeros(Float64, nx, ny, nz)
        
        cell_sizes_d = CuArray([1.0/(2*x_cell_size), 1.0/(2*y_cell_size), 1.0/(2*z_cell_size)])
        
        threads = (16, 16)
        blocks = nz
        @cuda threads=threads blocks=blocks calculate_efield_kernel!(Ex_d, Ey_d, Ez_d, V, cell_sizes_d)
        
        # Extrapolar a fronteras (se puede hacer con otro kernel, pero por simplicidad lo hacemos en la CPU si es necesario)
        # O, más simple, el kernel de movimiento de partículas puede manejar los bordes.
        # Aquí, simplemente copiamos los valores del interior, que es lo que hacía el código de CPU.
        @cuda threads=max(ny*nz, nx*nz) blocks=2 apply_neumann_bc_kernel!(Ex_d)
        @cuda threads=max(ny*nz, nx*nz) blocks=2 apply_neumann_bc_kernel!(Ey_d)
        # La frontera Z se maneja de forma similar si es necesario.
        
        return Ex_d, Ey_d, Ez_d # Devolvemos CuArrays
    else
        # Versión CPU
        return calculate_electric_field_from_potential_cpu(V, x_cell_size, y_cell_size, z_cell_size)
    end
end


# ===========================================================================
# IMPLEMENTACIONES ORIGINALES PARA CPU
# ===========================================================================

# (Aquí se asume que `calculate_grid_density` y `lorentz_force` están definidos en otro lugar)

function solve_poisson_equation_cpu(charge_density_grid, x_cell_size, y_cell_size, z_cell_size,
                                    V_anode; max_iterations=10000, tolerance=1e-6)
    nx, ny, nz = size(charge_density_grid)
    V = zeros(nx, ny, nz)
    V[:, :, 1] .= 0.0
    V[:, :, end] .= V_anode
    
    dx2_inv = 1.0 / x_cell_size^2
    dy2_inv = 1.0 / y_cell_size^2
    dz2_inv = 1.0 / z_cell_size^2
    denominator = 2.0 * (dx2_inv + dy2_inv + dz2_inv)
    ω = 1.8
    
    for iter in 1:max_iterations
        V_old = copy(V)
        max_change = 0.0
        for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
            V_new = ((V[i+1,j,k] + V[i-1,j,k]) * dx2_inv +
                     (V[i,j+1,k] + V[i,j-1,k]) * dy2_inv +
                     (V[i,j,k+1] + V[i,j,k-1]) * dz2_inv -
                     charge_density_grid[i,j,k] / ε_0) / denominator
            V[i,j,k] = (1-ω) * V[i,j,k] + ω * V_new
            max_change = max(max_change, abs(V[i,j,k] - V_old[i,j,k]))
        end
        V[1,:,:] = V[2,:,:]; V[end,:,:] = V[end-1,:,:]
        V[:,1,:] = V[:,2,:]; V[:,end,:] = V[:,end-1,:]
        if max_change < tolerance; break; end
    end
    return V
end

function calculate_electric_field_from_potential_cpu(V, x_cell_size, y_cell_size, z_cell_size)
    nx, ny, nz = size(V)
    Ex, Ey, Ez = zeros(nx,ny,nz), zeros(nx,ny,nz), zeros(nx,ny,nz)
    
    for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        Ex[i,j,k] = -(V[i+1,j,k] - V[i-1,j,k]) / (2 * x_cell_size)
        Ey[i,j,k] = -(V[i,j+1,k] - V[i,j-1,k]) / (2 * y_cell_size)
        Ez[i,j,k] = -(V[i,j,k+1] - V[i,j,k-1]) / (2 * z_cell_size)
    end
    
    Ex[1,:,:] = Ex[2,:,:]; Ex[end,:,:] = Ex[end-1,:,:]
    Ey[:,1,:] = Ey[:,2,:]; Ey[:,end,:] = Ey[:,end-1,:]
    Ez[:,:,1] = Ez[:,:,2]; Ez[:,:,end] = Ez[:,:,end-1]
    
    return Ex, Ey, Ez
end

# La interpolación y el cálculo de fuerza no necesitan una versión GPU separada
# si se integran en el kernel de movimiento de partículas.
# Por ahora, las dejamos como funciones de CPU.

function interpolate_electric_field(position, E_grid::ElectricFieldGrid, x_grid, y_grid, z_grid)
    # Esta función asume que los datos están en la CPU.
    # Se necesitará una versión "device" para usarla dentro de un kernel de movimiento.
    x, y, z = position
    i = searchsortedlast(x_grid, x)
    j = searchsortedlast(y_grid, y)
    k = searchsortedlast(z_grid, z)
    
    i = clamp(i, 1, length(x_grid)-1)
    j = clamp(j, 1, length(y_grid)-1)
    k = clamp(k, 1, length(z_grid)-1)
    
    return [E_grid.Ex[i,j,k], E_grid.Ey[i,j,k], E_grid.Ez[i,j,k]]
end

function total_force_on_electron(velocity, electric_field, magnetic_field, charge)
    # (Asume que `lorentz_force` está definido en otro lugar)
    return charge * electric_field + lorentz_force(velocity, magnetic_field, charge)
end