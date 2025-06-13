# GPU: Importamos CUDA para poder crear los arrays de la GPU (CuArray).
using CUDA

# --- Constantes Físicas Globales ---
# No necesitan cambios, son valores escalares que se pueden usar en cualquier parte.
const K_B = 1.380649e-23
const AMU = 1.66054e-27
const ELECTRON_MASS = 9.109e-31
const ELECTRON_CHARGE = -1.60218e-19
const ε_0 = 8.854187817e-12


# ===========================================================================
# DATOS PARA LA CPU (FORMATO ORIGINAL)
# ===========================================================================

# CPU: Mantenemos el diccionario original. Es ideal para la legibilidad y para
# la implementación en la CPU, que puede manejar estructuras de datos complejas.
const air_composition_cpu = Dict(
    "N2" => Dict{String, Any}(
        "mass" => 28.0134 * AMU,
        "fraction" => 0.7808,
        "ionization_energy_eV" => 15.6
        # Las funciones de interpolación se añadirán en el script principal
    ),
    "O2" => Dict{String, Any}(
        "mass" => 31.9988 * AMU,
        "fraction" => 0.2095,
        "ionization_energy_eV" => 12.07
    ),
)


# ===========================================================================
# DATOS Y ESTRUCTURAS PARA LA GPU
# ===========================================================================

# GPU: Definimos una estructura para contener todos los datos de los gases y
# las secciones eficaces que necesita el kernel de la GPU.
# Esto organiza los datos y facilita pasarlos como un solo argumento.
# Los campos serán CuArray (arrays en la memoria de la GPU).
struct GPUGasData
    # Propiedades de los gases (índice 1=N2, 2=O2)
    masses::CuArray{Float64, 1}
    fractions::CuArray{Float64, 1}
    ion_energies::CuArray{Float64, 1}

    # Datos de sección eficaz para N2
    n2_E::CuArray{Float64, 1}
    n2_total_cs::CuArray{Float64, 1}
    n2_ion_cs::CuArray{Float64, 1}

    # Datos de sección eficaz para O2
    o2_E::CuArray{Float64, 1}
    o2_total_cs::CuArray{Float64, 1}
    o2_ion_cs::CuArray{Float64, 1}
end

# GPU: Función de utilidad para preparar y transferir los datos a la GPU.
# Esta función se debe llamar UNA SOLA VEZ al inicio de la simulación.
# Toma los arrays de la CPU y los convierte en la estructura GPUGasData.
function setup_gpu_gas_data(use_gpu_flag::Bool, n2_data, o2_data)
    # Si la GPU no está disponible, no hacemos nada y devolvemos `nothing`.
    if !use_gpu_flag
        return nothing
    end

    println("✓ Preparando y transfiriendo datos de gases a la GPU...")

    # Aplanamos los datos del diccionario en arrays simples.
    # El orden es importante y debe ser consistente (aquí: 1=N2, 2=O2).
    masses_cpu = [air_composition_cpu["N2"]["mass"], air_composition_cpu["O2"]["mass"]]
    fractions_cpu = [air_composition_cpu["N2"]["fraction"], air_composition_cpu["O2"]["fraction"]]
    ion_energies_cpu = [air_composition_cpu["N2"]["ionization_energy_eV"], air_composition_cpu["O2"]["ionization_energy_eV"]]

    # Creamos la instancia de la estructura, transfiriendo cada array a la GPU con CuArray().
    gpu_data = GPUGasData(
        CuArray(masses_cpu),
        CuArray(fractions_cpu),
        CuArray(ion_energies_cpu),
        CuArray(n2_data.energy_eV),
        CuArray(n2_data.total_cs),
        CuArray(n2_data.ion_cs),
        CuArray(o2_data.energy_eV),
        CuArray(o2_data.total_cs),
        CuArray(o2_data.ion_cs)
    )

    return gpu_data
end