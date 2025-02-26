using Dates
using DataFrames
using CSV

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