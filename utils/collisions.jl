using Random

# ---  Modelo de Excitación (Estimación) ---
function estimate_excitation_cross_section(total_cross_section, ionization_cross_section, energy_eV, ionization_energy_eV)
    cross_section_diff = total_cross_section - ionization_cross_section
    excitation_cross_section = max(0.0, cross_section_diff)
    energy_above_threshold = max(0.0, energy_eV - ionization_energy_eV)
    excitation_fraction = exp(-0.1 * energy_above_threshold)
    excitation_cross_section = excitation_cross_section * excitation_fraction
    return excitation_cross_section
end

# ---  Función de Colisión Monte Carlo (¡¡¡MODIFICADA!!!) ---
function monte_carlo_collision(positions, velocities, air_density_n, dt, rng, efficiency)
    num_particles = size(velocities, 1)
    if num_particles == 0
        return positions, velocities, rng, zeros(num_particles), zeros(Bool, num_particles), zeros(num_particles), zeros(num_particles)
    end

    # Vectores para resultados - ESENCIAL: inicializar a cero
    inelastic_energy_transfer = zeros(num_particles)
    elastic_energy_transfer = zeros(num_particles)
    velocities_new = copy(velocities)
    collided_flags = zeros(Bool, num_particles)
    collision_energy_transfers_eV = zeros(num_particles)

    # Cálculos de energía para diagnóstico
    v_magnitudes = sqrt.(sum(velocities.^2, dims=2))
    E_e_eV = electron_energy_from_velocity.(v_magnitudes) ./ 1.60218e-19  # Convertir J a eV
    E_e_joules = electron_energy_from_velocity.(v_magnitudes)

    total_initial_energy_eV = sum(E_e_eV)
    total_initial_energy_J = sum(E_e_joules)
    """
    println("Diagnóstico Inicial:")
    println("  Número de electrones: $(num_particles)")
    println("  Media de energía de electrones: $(mean(E_e_eV)) eV")
    println("  Energía máxima de electrones: $(maximum(E_e_eV)) eV")
    println("  Energía mínima de electrones: $(minimum(E_e_eV)) eV")
    println("  Energía TOTAL inicial: $(total_initial_energy_eV) eV ($(total_initial_energy_J) J)")
    """
    # Constantes físicas
    excitation_energy_loss_eV = 3.0
    excitation_energy_loss_joules = excitation_energy_loss_eV * 1.60218e-19

    # Para cada partícula
    for i in 1:num_particles
        # IMPORTANTE: La energía disponible para este electrón
        electron_energy_joules = E_e_joules[i]

        # Si el electrón no tiene suficiente energía, saltar
        if electron_energy_joules < 1e-25
            continue
        end

        # Probabilidad total de colisión
        total_collision_prob = 0.0
        gas_probs = Dict{String, Float64}()

        for gas_name in keys(air_composition)
            gas_info = air_composition[gas_name]
            gas_fraction = gas_info["fraction"]
            total_cross_section_func = gas_info["total_cross_section_func"]
            total_cross_section = total_cross_section_func(E_e_eV[i])

            # Probabilidad de colisión por densidad, sección eficaz y tiempo
            gas_prob = (1.0 - exp(-(air_density_n * gas_fraction) * total_cross_section * v_magnitudes[i] * dt))
            gas_probs[gas_name] = gas_prob
            total_collision_prob += gas_prob * gas_fraction
        end

        # Limitar probabilidad por seguridad numérica
        total_collision_prob = min(total_collision_prob, 0.95)

        # Decidir si colisiona
        if rand(rng) < total_collision_prob
            # Seleccionar gas para colisión (ponderado por fracción)
            selected_gas = ""
            rand_val = rand(rng)
            cumulative_prob = 0.0

            for gas_name in keys(air_composition)
                gas_fraction = air_composition[gas_name]["fraction"]
                gas_prob = gas_probs[gas_name] * gas_fraction / total_collision_prob

                cumulative_prob += gas_prob
                if rand_val <= cumulative_prob
                    selected_gas = gas_name
                    break
                end
            end

            if selected_gas == ""
                selected_gas = last(keys(air_composition))
            end

            # Propiedades del gas seleccionado
            gas_info = air_composition[selected_gas]
            mass = gas_info["mass"]
            ionization_energy_eV = gas_info["ionization_energy_eV"]
            ionization_energy_joules = ionization_energy_eV * 1.60218e-19
            total_cross_section_func = gas_info["total_cross_section_func"]
            ionization_cross_section_func = gas_info["ionization_cross_section_func"]

            # Secciones eficaces
            total_cross_section = total_cross_section_func(E_e_eV[i])
            ionization_cross_section = ionization_cross_section_func(E_e_eV[i])
            excitation_cross_section = estimate_excitation_cross_section(
                total_cross_section, ionization_cross_section, E_e_eV[i], ionization_energy_eV)
            elastic_cross_section = max(0.0, total_cross_section - ionization_cross_section - excitation_cross_section)

            # Probabilidades relativas
            p_total = ionization_cross_section + excitation_cross_section + elastic_cross_section
            if p_total > 0
                p_ionize = ionization_cross_section / p_total
                p_excite = excitation_cross_section / p_total
                p_elastic = elastic_cross_section / p_total
            else
                p_elastic = 1.0
                p_ionize = 0.0
                p_excite = 0.0
            end

            # Decidir tipo de colisión
            collision_rand = rand(rng)
            energy_loss_joules = 0.0
            collision_type = "Elástica"

            # CRÍTICO: NO PUEDE TRANSFERIR MÁS ENERGÍA DE LA QUE TIENE
            max_transfer = 0.95 * electron_energy_joules

            # IONIZACIÓN (si tiene suficiente energía)
            if collision_rand < p_ionize && electron_energy_joules > ionization_energy_joules
                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(ionization_energy_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Ionización"

            # EXCITACIÓN (si tiene suficiente energía)
            elseif collision_rand < (p_ionize + p_excite) && electron_energy_joules > excitation_energy_loss_joules
                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(excitation_energy_loss_joules, max_transfer)
                inelastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Excitación"

            # ELÁSTICA
            else
                # Fracción teórica de transferencia elástica (ecuación física)
                fraction_energy_elastic = (2 * electron_mass / mass)

                # Transferencia limitada al valor mínimo
                energy_loss_joules = min(fraction_energy_elastic * electron_energy_joules, max_transfer)
                elastic_energy_transfer[i] = energy_loss_joules
                collision_type = "Elástica"
            end

            # CRUCIAL: Actualizar velocidad con energía remanente
            remaining_energy_joules = max(1e-30, electron_energy_joules - energy_loss_joules)
            new_speed = sqrt(2 * remaining_energy_joules / electron_mass)

            # Nueva dirección aleatoria para dispersión
            new_direction = randn(rng, 3)
            new_direction_norm = sqrt(sum(new_direction.^2))
            if new_direction_norm > 0
                new_direction = new_direction ./ new_direction_norm
            else
                new_direction = [0.0, 0.0, 1.0]
            end

            # Actualizar velocidad
            velocities_new[i, :] = new_direction .* new_speed

            # Registrar colisión
            collided_flags[i] = true
            collision_energy_transfers_eV[i] = energy_loss_joules / 1.60218e-19

            # Diagnóstico para primeras partículas
            """
            if i <= 3
                println("Colisión #$i con gas $selected_gas:")
                println("  E inicial = $(E_e_eV[i]) eV ($(electron_energy_joules) J)")
                println("  Tipo colisión: $collision_type")
                println("  Pérdida energía = $(energy_loss_joules/1.60218e-19) eV ($(energy_loss_joules) J)")
                println("  E final = $(remaining_energy_joules/1.60218e-19) eV ($(remaining_energy_joules) J)")
                println("  Ratio transferencia/inicial = $(energy_loss_joules/electron_energy_joules * 100)%")
            end
            """
        end
    end

    # Diagnóstico final con verificación de conservación de energía
    new_v_magnitudes = sqrt.(sum(velocities_new.^2, dims=2))
    new_E_e_eV = electron_energy_from_velocity.(new_v_magnitudes) ./ 1.60218e-19
    new_E_e_joules = electron_energy_from_velocity.(new_v_magnitudes)

    # Sumas totales para verificación
    inelastic_sum_J = sum(inelastic_energy_transfer)
    elastic_sum_J = sum(elastic_energy_transfer)
    inelastic_sum_eV = inelastic_sum_J / 1.60218e-19
    elastic_sum_eV = elastic_sum_J / 1.60218e-19

    total_final_energy_J = sum(new_E_e_joules)
    total_final_energy_eV = sum(new_E_e_eV)

    energy_lost_J = total_initial_energy_J - total_final_energy_J
    energy_lost_eV = total_initial_energy_eV - total_final_energy_eV

    energy_transferred_J = inelastic_sum_J + elastic_sum_J
    energy_transferred_eV = inelastic_sum_eV + elastic_sum_eV
    """
    println("Diagnóstico Final:")
    println("  Media de energía después de colisiones: $(mean(new_E_e_eV)) eV")
    println("  Energía máxima después de colisiones: $(maximum(new_E_e_eV)) eV")
    println("  Energía TOTAL final: $(total_final_energy_eV) eV ($(total_final_energy_J) J)")
    println("  Pérdida de energía total: $(energy_lost_eV) eV ($(energy_lost_J) J)")
    println("  Energía transferida inelástica total: $(inelastic_sum_eV) eV ($(inelastic_sum_J) J)")
    println("  Energía transferida elástica total: $(elastic_sum_eV) eV ($(elastic_sum_J) J)")
    println("  Energía transferida total: $(energy_transferred_eV) eV ($(energy_transferred_J) J)")
    """
    # VERIFICACIÓN ESENCIAL: La energía transferida debe ser igual (o muy cercana) a la energía perdida
    conservation_error = abs(energy_lost_J - energy_transferred_J) / max(1e-30, energy_transferred_J)
    if conservation_error > 0.01
       println("⚠️ ERROR DE CONSERVACIÓN: Diferencia entre energía perdida y transferida: $(energy_lost_eV - energy_transferred_eV) eV")
    else
        println("✓ Conservación de energía verificada (error relativo = $(conservation_error*100)%).")
    end

    return positions, velocities_new, rng, inelastic_energy_transfer, collided_flags, collision_energy_transfers_eV, elastic_energy_transfer
end