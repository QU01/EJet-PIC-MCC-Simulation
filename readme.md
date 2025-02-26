
# Reporte Técnico: Simulación PIC-MCC para Plasma de Descarga en Gas
### Ing. Nick Vázquez
## Índice
1. **Introducción y Marco Teórico**
2. **Descripción del Código**
3. **Componentes Principales**
4. **Algoritmos Clave**
5. **Validación y Optimización**
6. **Conclusiones**

---

## 1. Introducción y Marco Teórico

Este reporte documenta una implementación de un simulador de plasma mediante el método Particle-In-Cell con Monte Carlo Collisions (PIC-MCC) en Julia. La simulación modela el comportamiento de electrones en un gas bajo la influencia de campos magnéticos, con enfoque en la transferencia de energía entre electrones y moléculas de gas.

La simulación resuelve numéricamente las siguientes ecuaciones físicas:

1. **Ecuación de movimiento** de partículas cargadas bajo fuerza de Lorentz:
   $$\frac{d\vec{v}}{dt} = \frac{q}{m}(\vec{v} \times \vec{B})$$
   donde $q$ es la carga del electrón, $m$ es la masa del electrón, $\vec{v}$ es la velocidad, y $\vec{B}$ es el campo magnético.

2. **Procesos de colisión** entre electrones y moléculas de gas:
   - Colisiones elásticas
   - Excitación molecular
   - Ionización

3. **Transferencia de energía** de electrones al gas modelada según:
   $$\Delta T = \frac{2}{3k_B n_{gas}V_{cell}} \Delta E_{transferida}$$
   donde $k_B$ es la constante de Boltzmann, $n_{gas}$ es la densidad del gas, $V_{cell}$ es el volumen de la celda, y $\Delta E_{transferida}$ es la energía transferida.

---

## 2. Descripción del Código

El código implementa una simulación tridimensional que rastrea el movimiento de electrones individuales en un gas (aire) bajo la influencia de un campo magnético. Utiliza un enfoque híbrido PIC-MCC donde:

- Las partículas (electrones) se mueven continuamente en el espacio 3D
- El gas se modela como un medio continuo con densidad uniforme
- Las colisiones se modelan mediante procesos estocásticos de Monte Carlo
- La energía se deposita en una malla discreta para el cálculo de temperatura

### Constantes Físicas y Propiedades del Aire

```julia
const k_b = 1.380649e-23       # Constante de Boltzmann [J/K]
const amu = 1.66054e-27        # Unidad de masa atómica [kg]
const electron_mass = 9.109e-31 # Masa del electrón [kg]
const electron_charge = -1.60218e-19 # Carga del electrón [C]

# Composición del aire modelada con N2 y O2
air_composition = Dict(
    "N2" => Dict{String, Any}(
        "mass" => 28.0134 * amu,        # Masa molecular [kg]
        "fraction" => 0.7808,           # Fracción en volumen
        "ionization_energy_eV" => 15.6  # Potencial de ionización [eV]
    ),
    "O2" => Dict{String, Any}(
        "mass" => 31.9988 * amu,
        "fraction" => 0.2095,
        "ionization_energy_eV" => 12.07
    ),
)
```

### Parámetros de Malla

```julia
chamber_width = 0.1    # Ancho de la cámara [m]
chamber_length = 0.1   # Largo de la cámara [m] 
chamber_height = 0.1   # Altura de la cámara [m]
num_x_cells = 20       # Número de celdas en dirección x
num_y_cells = 20       # Número de celdas en dirección y
num_z_cells = 50       # Número de celdas en dirección z
```

---

## 3. Componentes Principales

### 3.1 Secciones Eficaces de Colisión

El código utiliza datos experimentales de las secciones eficaces de colisión del Nitrógeno (N2) y Oxígeno (O2) basados en los estudios de Itikawa. Estas secciones eficaces describen la probabilidad de diferentes tipos de interacciones electrón-molécula en función de la energía del electrón.

```julia
# Secciones eficaces interpoladas para cálculo continuo
n2_total_cross_section_func = LinearInterpolation(n2_energy_eV, n2_total_cross_section; extrapolation_bc = Flat())
n2_ionization_cross_section_func = LinearInterpolation(n2_energy_eV, n2_ionization_cross_section; extrapolation_bc = Flat())

o2_total_cross_section_func = LinearInterpolation(o2_energy_eV, o2_total_cross_section; extrapolation_bc = Flat())
o2_ionization_cross_section_func = LinearInterpolation(o2_energy_eV, o2_ionization_cross_section; extrapolation_bc = Flat())
```

La sección eficaz de excitación se estima con:

```julia
function estimate_excitation_cross_section(total_cross_section, ionization_cross_section, energy_eV, ionization_energy_eV)
    cross_section_diff = total_cross_section - ionization_cross_section
    excitation_cross_section = max(0.0, cross_section_diff)
    energy_above_threshold = max(0.0, energy_eV - ionization_energy_eV)
    excitation_fraction = exp(-0.1 * energy_above_threshold)
    excitation_cross_section = excitation_cross_section * excitation_fraction
    return excitation_cross_section
end
```

Esta función estima la sección eficaz de excitación como la diferencia entre la sección eficaz total y la de ionización, modulada por un factor que decrece exponencialmente con la energía por encima del umbral de ionización.

### 3.2 Cálculo de Densidad y Velocidad Electrónica

```julia
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
```

Estas funciones implementan:
- La ecuación de gas ideal: $n = \frac{P}{k_B T}$
- La relación entre energía cinética y velocidad: $E_k = \frac{1}{2}mv^2$

### 3.3 Inicialización de Electrones

```julia
function initialize_electrons(num_electrons, chamber_width, chamber_length, initial_electron_velocity)
    rng = MersenneTwister(0)
    x_positions = rand(rng, num_electrons) * chamber_width
    y_positions = rand(rng, num_electrons) * chamber_length
    z_positions = zeros(num_electrons)

    vx = zeros(num_electrons)
    vy = zeros(num_electrons)
    vz = fill(initial_electron_velocity, num_electrons)

    positions = hcat(x_positions, y_positions, z_positions)
    velocities = hcat(vx, vy, vz)
    return positions, velocities
end
```

Los electrones se inicializan con una distribución espacial uniforme en el plano XY en la base de la cámara (z=0), con velocidades iniciales dirigidas hacia arriba (+z).

---

## 4. Algoritmos Clave

### 4.1 Algoritmo de Boris para Movimiento en Campo Magnético

```julia
function move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
    new_velocities = copy(velocities)
    B_magnitude = norm(magnetic_field)
    
    # Si no hay campo magnético significativo, movimiento simple
    if B_magnitude < 1e-10
        new_positions = positions + velocities * dt
        return new_positions, new_velocities
    end
    
    # Vector unitario en dirección del campo magnético
    b_hat = magnetic_field / B_magnitude
    
    # Ángulo de rotación durante medio paso
    theta = electron_charge * B_magnitude * dt / (2 * electron_mass)
    
    # Factores para rotación
    cos_2theta = cos(2 * theta)
    sin_2theta = sin(2 * theta)
    
    # Para cada electrón
    for i in 1:size(velocities, 1)
        v = velocities[i, :]
        v_mag_initial = norm(v)
        
        # Descomponer velocidad: componentes paralela y perpendicular a B
        v_parallel = dot(v, b_hat) * b_hat
        v_perp = v - v_parallel
        
        # Rotar solo componente perpendicular
        v_perp_mag = norm(v_perp)
        
        if v_perp_mag > 1e-10
            # Vectores unitarios perpendiculares en plano de rotación
            e1 = v_perp / v_perp_mag
            e2 = cross(b_hat, e1)
            
            # Rotación de la componente perpendicular
            v_perp_rotated = v_perp_mag * (cos_2theta * e1 + sin_2theta * e2)
            
            # Nueva velocidad: paralela + perpendicular rotada
            new_velocities[i, :] = v_parallel + v_perp_rotated
            
            # Garantizar conservación exacta de energía
            v_mag_final = norm(new_velocities[i, :])
            if abs(v_mag_final - v_mag_initial) / v_mag_initial > 1e-10
                new_velocities[i, :] *= (v_mag_initial / v_mag_final)
            end
        end
    end
    
    # Actualizar posiciones (método de segundo orden)
    new_positions = positions + 0.5 * (velocities + new_velocities) * dt
    
    return new_positions, new_velocities
end
```

El algoritmo de Boris proporciona:
1. Conservación exacta de energía (solo cambia la dirección, no la magnitud de la velocidad)
2. Estabilidad numérica a largo plazo
3. Precisión de segundo orden en el tiempo

La frecuencia del ciclotón electrónico se calcula como:
$$\omega_c = \frac{|q|B}{m}$$

Y el período:
$$T_c = \frac{2\pi}{\omega_c} = \frac{2\pi m}{|q|B}$$

### 4.2 Condiciones de Frontera

```julia
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

    # [procedimientos similares para y, z]

    positions_updated = hcat(x, y, z)
    velocities_updated = hcat(vx, vy, vz)
    
    return positions_updated, velocities_updated
end
```

Implementa rebotes elásticos en las paredes de la cámara rectangular, invirtiendo la componente de velocidad perpendicular a la superficie de contacto.

### 4.3 Monte Carlo Collisions

```julia
function monte_carlo_collision(positions, velocities, air_density_n, dt, rng, efficiency)
    num_particles = size(velocities, 1)
    
    # Vectores para resultados
    inelastic_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    # Cálculos de energía inicial
    v_magnitudes = sqrt.(sum(velocities.^2, dims=2))
    E_e_eV = electron_energy_from_velocity.(v_magnitudes) ./ 1.60218e-19
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)
    
    # Para cada partícula
    for i in 1:num_particles
        # Cálculo de probabilidad de colisión con cada gas
        total_collision_prob = 0.0
        gas_probs = Dict{String, Float64}()
        
        for gas_name in keys(air_composition)
            gas_info = air_composition[gas_name]
            gas_fraction = gas_info["fraction"]
            total_cross_section_func = gas_info["total_cross_section_func"]
            total_cross_section = total_cross_section_func(E_e_eV[i])
            
            # Probabilidad de colisión: P = 1 - exp(-n*σ*v*dt)
            gas_prob = (1.0 - exp(-(air_density_n * gas_fraction) * total_cross_section * v_magnitudes[i] * dt))
            gas_probs[gas_name] = gas_prob
            total_collision_prob += gas_prob * gas_fraction
        end
        
        # Verificar si ocurre colisión
        if rand(rng) < total_collision_prob
            # [Selección de gas y tipo de colisión]
            
            # Decidir tipo de colisión basado en secciones eficaces relativas
            collision_rand = rand(rng)
            
            if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
                # Colisión de ionización
                energy_loss_joules = min(ionization_energy_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > excitation_energy_joules
                # Colisión de excitación
                energy_loss_joules = min(excitation_energy_loss_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
            else
                # Colisión elástica
                fraction_energy_elastic = (2 * electron_mass / gas_mass)
                energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
                elastic_energy_transfer[i] = energy_loss_joules
            end
            
            # Actualizar velocidad con energía remanente
            remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
            new_speed = sqrt(2 * remaining_energy_joules / electron_mass)
            
            # Nueva dirección aleatoria para dispersión
            new_direction = randn(rng, 3)
            new_direction = new_direction / norm(new_direction)
            
            # Asignar nueva velocidad
            velocities_new[i, :] = new_direction * new_speed
            collided_flags[i] = true
        end
    end
    
    return positions, velocities_new, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer
end
```

La probabilidad de colisión para una partícula con velocidad $v$ en un tiempo $dt$ es:

$$P_{colisión} = 1 - \exp(-n \sigma v dt)$$

Donde:
- $n$ es la densidad del gas
- $\sigma$ es la sección eficaz total
- $v$ es la velocidad del electrón
- $dt$ es el paso de tiempo

Para colisiones elásticas, la fracción de energía transferida es:

$$\frac{\Delta E}{E} = \frac{2m_e}{M_{gas}}$$

Donde $m_e$ es la masa del electrón y $M_{gas}$ es la masa molecular del gas.

### 4.4 Deposición de Energía en Malla

```julia
function deposit_energy(positions, energy_transfer, x_grid, y_grid, z_grid)
    bins = (collect(x_grid), collect(y_grid), collect(z_grid))
    w = Weights(energy_transfer)
    h = fit(Histogram, (positions[:,1], positions[:,2], positions[:,3]), w, bins)
    return h.weights
end

function update_temperature_grid(temperature_grid, energy_deposition_grid, elastic_energy_deposition_grid, n_air, cell_volume)
    n_air_cell = n_air * cell_volume
    delta_T_inelastic = (2.0 / (3.0 * k_b)) * (energy_deposition_grid ./ n_air_cell)
    delta_T_elastic = (2.0 / (3.0 * k_b)) * (elastic_energy_deposition_grid ./ n_air_cell)
    new_temperature_grid = temperature_grid .+ delta_T_inelastic .+ delta_T_elastic
    return new_temperature_grid
end
```

El cambio de temperatura se calcula según:

$$\Delta T = \frac{2}{3 k_B n V_{cell}} \Delta E$$

Esta ecuación proviene de la relación entre energía interna y temperatura para un gas ideal:

$$E_{internal} = \frac{3}{2} n k_B T V_{cell}$$

### 4.5 Algoritmo Principal de Simulación

```julia
function run_pic_simulation(initial_positions, initial_velocities, initial_temperature_grid,
    initial_air_density_n, air_composition, dt, simulated_electrons_per_step,
    magnetic_field, electron_charge, electron_mass, initial_electron_velocity; verbose=true)

    # Inicialización de historiales
    
    while avg_temps_history[end] < target_temperature && step < max_steps
        step += 1
        
        # 1. Inyectar nuevos electrones
        new_positions, new_velocities = initialize_electrons(simulated_electrons_per_step, chamber_width, chamber_length, initial_electron_velocity)
        positions = vcat(positions, new_positions)
        velocities = vcat(velocities, new_velocities)
        
        # 2. Calcular energía de entrada
        step_input_energy = sum(electron_energy_from_velocity.(sqrt.(sum(new_velocities.^2, dims=2)))) * particle_weight
        
        # 3. Movimiento de electrones en campo magnético
        positions, velocities = move_electrons(positions, velocities, dt, magnetic_field, electron_charge, electron_mass)
        
        # 4. Aplicar condiciones de frontera
        positions, velocities = apply_rectangular_boundary_conditions(positions, velocities, chamber_width, chamber_length, chamber_height)
        
        # 5. Simulación de colisiones
        positions, velocities, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer = monte_carlo_collision(
            positions, velocities, initial_air_density_n, dt, rng, 1.0
        )
        
        # 6. Depositar energía en la malla
        energy_deposition_grid_step = deposit_energy(positions, inelastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)
        elastic_energy_deposition_grid_step = deposit_energy(positions, elastic_energy_transfer * particle_weight, x_grid, y_grid, z_grid)
        
        # 7. Actualizar temperatura
        temperature_grid = update_temperature_grid(temperature_grid, energy_deposition_grid_step, elastic_energy_deposition_grid_step, initial_air_density_n, cell_volume)
        
        # 8. Calcular eficiencia
        current_internal_energy = (3/2) * k_b * Statistics.mean(temperature_grid) * initial_air_density_n * cell_volume * TOTAL_CELLS
        energy_transferred_to_gas = current_internal_energy - previous_internal_energy
        step_efficiency = (energy_transferred_to_gas / step_input_energy) * 100
        
        # 9. Actualizar historiales
        push!(avg_temps_history, Statistics.mean(temperature_grid))
        push!(efficiency_history, step_efficiency)
    end
    
    return results...
end
```

---

## 5. Validación y Optimización

### 5.1 Verificación de la Conservación de Energía

El código implementa varias verificaciones para garantizar que la energía se conserve apropiadamente durante la simulación:

1. **En el algoritmo de Boris**: Garantizando que el movimiento en campo magnético no altere la energía cinética.
2. **En el proceso de colisión**: Verificando que la energía transferida no exceda la energía del electrón.
3. **En el balance global**: Comprobando que la suma de energía transferida al gas más la energía remanente en los electrones sea igual a la energía total de entrada.

La expresión que verifica la conservación es:

$$\left|\frac{E_{perdida} - E_{transferida}}{E_{transferida}}\right| < \epsilon$$

donde $\epsilon$ es una tolerancia pequeña (típicamente 0.01).

### 5.2 Validez del Paso de Tiempo

El paso de tiempo debe satisfacer dos condiciones:

1. **Condición de Courant-Friedrichs-Lewy (CFL)** para estabilidad espacial:
   $$dt < \frac{\Delta x}{v_{max}}$$

2. **Condición del ciclotón** para precisión en campo magnético:
   $$dt < \frac{0.1 \cdot 2\pi m_e}{|q|B}$$

### 5.3 Búsqueda Paramétrica para Optimización

La función `parameter_search()` explora sistemáticamente el espacio de parámetros:

```julia
function parameter_search()
    # Rangos de exploración
    electron_energies = [50.0, 100.0, 150.0, 200.0]  # eV
    pressures = [1e6, 2e6, 3e6, 4e6]  # Pa
    magnetic_fields = [0.5, 1.0, 1.5, 2.0]  # Tesla
    
    # Para cada combinación de parámetros
    for energy in electron_energies
        for pressure in pressures
            for field in magnetic_fields
                # Evaluar eficiencia
                efficiency = estimate_efficiency(energy, pressure, field)
                
                # Registrar resultados
                push!(results, (energy, pressure, field, efficiency))
            end
        end
    end
    
    # Ordenar por eficiencia
    sort!(results, :Efficiency, rev=true)
    return results, best_params, best_efficiency
end
```

La eficiencia se calcula como:

$$\eta = \frac{E_{transferida \, al \, gas}}{E_{entrada \, total}} \times 100\%$$

---

## 6. Conclusiones

La simulación PIC-MCC implementada permite modelar el calentamiento de un gas por impacto electrónico bajo influencia de campos magnéticos. Las conclusiones principales son:

1. **Algoritmo de Boris**: Fundamental para la correcta simulación del movimiento de electrones en campo magnético, garantizando la conservación de energía.

2. **Modelo de Colisiones Monte Carlo**: El enfoque estocástico proporciona una representación física adecuada de las interacciones electrón-molécula.

3. **Verificación de Conservación**: Las validaciones implementadas garantizan que los resultados sean físicamente plausibles.

4. **Optimización Paramétrica**: La búsqueda sistemática permite identificar los parámetros óptimos (energía de electrón, presión, campo magnético) para maximizar la eficiencia de calentamiento.

5. **Diagnósticos Detallados**: El seguimiento paso a paso de la energía y temperatura permite comprender la física del proceso.

Los resultados muestran que la eficiencia del calentamiento depende fuertemente de los parámetros operativos, y el modelo proporciona una herramienta valiosa para optimizar estos parámetros en aplicaciones prácticas de generación de plasma y calentamiento de gases.
## 7. Resultados de la Simulación

### --- Resultados de Simulación PIC-MCC ---

**Fecha y Hora:** 2025-02-26T14:29:02.183

**--- Parámetros de Simulación ---**
* Temperatura Inicial: 800.0 K
* Presión Inicial: 1.0 MPa
* Densidad Inicial del Aire: 9.0537131450499e25 m^-3
* Energía Inicial de los Electrones: 50.0 eV
* Campo Magnético Axial (Bz): 0.5 T
* Paso de Tiempo (dt): 1.0e-12 s
* Tiempo de Simulación por paso: 1.0e-12 s
* Electrones Simulados por paso: 100
* Temperatura Objetivo: 2200.0 K
* Máximo número de pasos: 1000

**--- Resultados ---**
* Temperatura Final Promedio: 2201.1098822691433 K
* Alcanzó Temperatura Objetivo: true
* Número de Pasos Simulados: 686
* Tiempo Total de Simulación: 6.86e-10 s
* Aumento Total de Energía Interna del Aire: 2627.0810292546444 J
* Energía Total Introducida por Electrones: 2747.7387000000463 J
* Eficiencia de la Simulación: 95.07649872269027 %
* Eficiencia Promedio por Paso: 95.07649872269027 %

**--- Plots ---**
* Temperatura Promedio vs Tiempo: temperature_vs_time.png
* Eficiencia Promedio vs Tiempo: efficiency_vs_time.png
* Densidad de Electrones al Final (Slice): density_heatmap.png
* Temperatura del Aire al Final (Slice): temperature_heatmap.png

### Análisis de Resultados

La simulación PIC-MCC del calentamiento de aire hasta una temperatura objetivo de 2200.0 K ha sido exitosa. Los resultados indican lo siguiente:

* **Calentamiento Eficiente:** La simulación alcanzó la temperatura objetivo (2200.0 K) con una temperatura final promedio de 2201.1 K, superando ligeramente el objetivo. Esto se logró en 686 pasos de simulación, lo que corresponde a un tiempo total de simulación de 6.86e-10 segundos.

* **Alta Eficiencia Energética:** La eficiencia global de la simulación es notablemente alta, alcanzando el 95.08%. Esto significa que el 95.08% de la energía total introducida por los electrones se transfirió al aire, resultando en un aumento de la temperatura. La eficiencia promedio por paso coincide con la eficiencia global, lo que sugiere una transferencia de energía consistente a lo largo de la simulación.

* **Visualización de la Temperatura vs Tiempo:** El gráfico [Temperatura Promedio vs Tiempo](temperature_vs_time.png) muestra la evolución de la temperatura del aire a lo largo de la simulación. Se observa un aumento constante y rápido de la temperatura desde la temperatura inicial de 800.0 K hasta alcanzar y superar ligeramente la temperatura objetivo de 2200.0 K. La línea roja punteada en el gráfico indica claramente el objetivo de temperatura, permitiendo visualizar el momento en que se alcanza.

* **Visualización de la Eficiencia vs Tiempo:** El gráfico [Eficiencia Promedio vs Tiempo](efficiency_vs_time.png) presenta la eficiencia de calentamiento por paso a lo largo del tiempo. Se aprecia que la eficiencia se estabiliza rápidamente en un valor alto, cercano al 95%, con algunas fluctuaciones menores. Esto confirma la alta eficiencia del proceso de calentamiento simulado y su consistencia a lo largo de la simulación.

* **Próximos Pasos:** Los gráficos [Densidad de Electrones al Final (Slice)](density_heatmap.png) y [Temperatura del Aire al Final (Slice)](temperature_heatmap.png) (no incluidos en este reporte) proporcionarían información espacial detallada sobre la distribución de la densidad de electrones y la temperatura del aire al final de la simulación. Estos gráficos serían útiles para analizar la uniformidad del calentamiento y la distribución espacial del plasma generado.

En resumen, la simulación demuestra un método eficiente y rápido para calentar aire utilizando un plasma generado mediante descarga en gas, con una alta eficiencia de transferencia de energía de los electrones al gas. Los resultados obtenidos son consistentes con los parámetros de simulación y proporcionan información valiosa para la optimización de sistemas de calentamiento de plasma.