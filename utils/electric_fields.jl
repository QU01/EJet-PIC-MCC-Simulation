using LinearAlgebra
using Statistics

# Constante de permitividad del vacío
const ε_0 = 8.854187817e-12  # F/m

# Estructura para almacenar el campo eléctrico en la malla
struct ElectricFieldGrid
    Ex::Array{Float64, 3}
    Ey::Array{Float64, 3}
    Ez::Array{Float64, 3}
    potential::Array{Float64, 3}
end

"""
Calcular el potencial eléctrico en la malla usando diferencias finitas
con condiciones de frontera de Dirichlet
"""
function solve_poisson_equation(charge_density_grid, x_cell_size, y_cell_size, z_cell_size,
                               V_anode; max_iterations=10000, tolerance=1e-6)
    nx, ny, nz = size(charge_density_grid)
    
    # Inicializar potencial
    V = zeros(nx, ny, nz)
    
    # Condiciones de frontera Dirichlet para z
    V[:, :, 1] .= 0.0       # Cátodo (z=0)
    V[:, :, end] .= V_anode  # Ánodo (z=nz)
    
    # Factores para la ecuación de Poisson discretizada
    dx2_inv = 1.0 / x_cell_size^2
    dy2_inv = 1.0 / y_cell_size^2
    dz2_inv = 1.0 / z_cell_size^2
    denominator = 2.0 * (dx2_inv + dy2_inv + dz2_inv)
    
    # Método de Gauss-Seidel con sobre-relajación
    ω = 1.8  # Factor de sobre-relajación
    
    for iter in 1:max_iterations
        V_old = copy(V)
        max_change = 0.0
        
        # Iterar sobre puntos internos
        for k in 2:nz-1
            for j in 2:ny-1
                for i in 2:nx-1
                    # Aproximación de diferencias finitas
                    V_new = (
                        (V[i+1,j,k] + V[i-1,j,k]) * dx2_inv +
                        (V[i,j+1,k] + V[i,j-1,k]) * dy2_inv +
                        (V[i,j,k+1] + V[i,j,k-1]) * dz2_inv -
                        charge_density_grid[i,j,k] / ε_0
                    ) / denominator
                    
                    # Sobre-relajación
                    V[i,j,k] = (1-ω) * V[i,j,k] + ω * V_new
                    
                    # Verificar convergencia
                    change = abs(V[i,j,k] - V_old[i,j,k])
                    max_change = max(max_change, change)
                end
            end
        end
        
        # Condiciones de frontera Neumann en las paredes laterales
        # (sin flujo de campo eléctrico a través de las paredes)
        V[1,:,:] = V[2,:,:]
        V[end,:,:] = V[end-1,:,:]
        V[:,1,:] = V[:,2,:]
        V[:,end,:] = V[:,end-1,:]
        
        # Verificar convergencia
        if max_change < tolerance
            println("Ecuación de Poisson convergió en $iter iteraciones")
            break
        end
        
        if iter == max_iterations
            @warn "Ecuación de Poisson no convergió completamente"
        end
    end
    
    # Debug check: verify anode voltage is uniform
    anode_values = unique(V[:, :, end])
    if length(anode_values) != 1 || anode_values[1] != V_anode
        @warn "Anode voltage is not uniform! Expected $V_anode, got values: $anode_values"
    end

    return V
end

"""
Calcular el campo eléctrico a partir del potencial usando gradiente
E = -∇V
"""
function calculate_electric_field_from_potential(V, x_cell_size, y_cell_size, z_cell_size)
    nx, ny, nz = size(V)
    Ex = zeros(nx, ny, nz)
    Ey = zeros(nx, ny, nz)
    Ez = zeros(nx, ny, nz)
    
    # Calcular gradientes usando diferencias centradas
    for k in 2:nz-1
        for j in 2:ny-1
            for i in 2:nx-1
                Ex[i,j,k] = -(V[i+1,j,k] - V[i-1,j,k]) / (2 * x_cell_size)
                Ey[i,j,k] = -(V[i,j+1,k] - V[i,j-1,k]) / (2 * y_cell_size)
                Ez[i,j,k] = -(V[i,j,k+1] - V[i,j,k-1]) / (2 * z_cell_size)
            end
        end
    end
    
    # Extrapolar a las fronteras
    Ex[1,:,:] = Ex[2,:,:]
    Ex[end,:,:] = Ex[end-1,:,:]
    Ey[:,1,:] = Ey[:,2,:]
    Ey[:,end,:] = Ey[:,end-1,:]
    Ez[:,:,1] = Ez[:,:,2]
    Ez[:,:,end] = Ez[:,:,end-1]
    
    return Ex, Ey, Ez
end

"""
Calcular la densidad de carga en la malla a partir de las posiciones de electrones
"""
function calculate_charge_density(positions, particle_weight, x_grid, y_grid, z_grid, cell_volume)
    # Usar la función existente de densidad y convertir a densidad de carga
    electron_density = calculate_grid_density(positions, x_grid, y_grid, z_grid)
    
    # Densidad de carga = -e * densidad de electrones * peso de partícula / volumen de celda
    charge_density = -electron_charge * electron_density * particle_weight / cell_volume
    
    return charge_density
end

"""
Interpolar el campo eléctrico en una posición específica
"""
function interpolate_electric_field(position, Ex_grid, Ey_grid, Ez_grid, 
                                   x_grid, y_grid, z_grid)
    x, y, z = position
    
    # Encontrar índices de la celda
    i = searchsortedlast(x_grid, x)
    j = searchsortedlast(y_grid, y)
    k = searchsortedlast(z_grid, z)
    
    # Asegurar que estamos dentro de los límites
    i = clamp(i, 1, length(x_grid)-1)
    j = clamp(j, 1, length(y_grid)-1)
    k = clamp(k, 1, length(z_grid)-1)
    
    # Interpolación trilineal simple
    # (Para mayor precisión, se podría implementar interpolación más sofisticada)
    Ex_interp = Ex_grid[i,j,k]
    Ey_interp = Ey_grid[i,j,k]
    Ez_interp = Ez_grid[i,j,k]
    
    return [Ex_interp, Ey_interp, Ez_interp]
end

"""
Calcular la fuerza total sobre un electrón (Lorentz + Eléctrica)
"""
function total_force_on_electron(velocity, electric_field, magnetic_field, charge)
    # F = q(E + v × B)
    lorentz_force_component = lorentz_force(velocity, magnetic_field, charge)
    electric_force_component = charge * electric_field
    
    return electric_force_component + lorentz_force_component
end