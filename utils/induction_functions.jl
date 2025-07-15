# ---------------------------------------------------------------------------
# FILE: induction_functions.jl (Handles variable B-field and induced E-field)
# ---------------------------------------------------------------------------

using LinearAlgebra
using Base:@kwdef

# GPU: Importar CUDA para poder crear arrays en la GPU
using CUDA

# Asumimos que μ₀ está definido en constants.jl
# Si no, descomentar:
# const μ₀ = 4π * 1e-7

"""
    SolenoidParameters

Estructura para contener los parámetros de un solenoide ideal.
"""
@kwdef struct SolenoidParameters
    current_amplitude::Float64 # Amperios (I₀)
    frequency::Float64         # Hertz (f)
    num_turns::Int             # Número de vueltas
    length::Float64            # Longitud del solenoide (m)
    resistance::Float64 = 10.0 # Default resistance (Ω)
end

"""
    calculate_magnetic_field(solenoid, time)

Calcula el campo magnético axial B_z(t) de un solenoide ideal en un instante `t`.
"""
function calculate_magnetic_field(solenoid::SolenoidParameters, time::Float64)
    ω = 2 * π * solenoid.frequency
    current = solenoid.current_amplitude * sin(ω * time)
    
    # Campo magnético de un solenoide ideal (μ₀ * n * I)
    n = solenoid.num_turns / solenoid.length
    b_z = μ₀ * n * current
    
    return [0.0, 0.0, b_z]
end

"""
    calculate_induced_electric_field_grid(solenoid, x_grid, y_grid, z_grid, time, use_gpu)

Calcula la malla del campo eléctrico inducido E_inducido(r, t) en coordenadas cartesianas.
"""
function calculate_induced_electric_field_grid(solenoid::SolenoidParameters, x_grid, y_grid, z_grid, time::Float64, use_gpu::Bool)
    nx, ny, nz = length(x_grid) - 1, length(y_grid) - 1, length(z_grid) - 1
    
    # Calcular la derivada del campo magnético dB_z/dt
    ω = 2 * π * solenoid.frequency
    n = solenoid.num_turns / solenoid.length
    dbz_dt = μ₀ * n * ω * solenoid.current_amplitude * cos(ω * time)
    
    # Crear arrays para los componentes del campo eléctrico
    # La decisión de usar CPU o GPU se basa en el flag `use_gpu`
    if use_gpu
        Ex_induced = CUDA.zeros(Float64, nx, ny, nz)
        Ey_induced = CUDA.zeros(Float64, nx, ny, nz)
        Ez_induced = CUDA.zeros(Float64, nx, ny, nz) # Siempre será cero
    else
        Ex_induced = zeros(Float64, nx, ny, nz)
        Ey_induced = zeros(Float64, nx, ny, nz)
        Ez_induced = zeros(Float64, nx, ny, nz) # Siempre será cero
    end

    # Calcular el campo en el centro de cada celda
    for k in 1:nz, j in 1:ny, i in 1:nx
        # Coordenadas del centro de la celda (i, j)
        x = x_grid[i] + (x_grid[i+1] - x_grid[i]) / 2
        y = y_grid[j] + (y_grid[j+1] - y_grid[j]) / 2
        
        # E_inducido_x = (y / 2) * (dB_z / dt)
        # E_inducido_y = -(x / 2) * (dB_z / dt)
        Ex_induced[i, j, k] =  (y / 2) * dbz_dt
        Ey_induced[i, j, k] = -(x / 2) * dbz_dt
    end
    
    return Ex_induced, Ey_induced, Ez_induced
end

"""
    calculate_fields(solenoid, x_grid, y_grid, z_grid, time, use_gpu)

Función principal del módulo. Devuelve el campo magnético B(t) y la malla del
campo eléctrico inducido (Ex, Ey, Ez).
"""
function calculate_fields(solenoid::SolenoidParameters, x_grid, y_grid, z_grid, time::Float64, use_gpu::Bool)
    # 1. Calcular el campo magnético variable en el tiempo
    b_field_vector = calculate_magnetic_field(solenoid, time)
    
    # 2. Calcular la malla del campo eléctrico inducido
    Ex_induced, Ey_induced, Ez_induced = calculate_induced_electric_field_grid(
        solenoid, x_grid, y_grid, z_grid, time, use_gpu
    )
    
    return b_field_vector, Ex_induced, Ey_induced, Ez_induced
end

