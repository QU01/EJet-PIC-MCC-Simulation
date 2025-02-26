using Interpolations

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